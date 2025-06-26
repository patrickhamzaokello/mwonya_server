package main

import (
	"container/list"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type AudioTrack struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	Artist       string    `json:"artist"`
	Duration     int       `json:"duration"` // in seconds
	Status       string    `json:"status"`   // "processing", "ready", "error"
	CreatedAt    time.Time `json:"createdAt"`
	ProcessingProgress float64 `json:"processingProgress"`
}

type ProcessingStatus struct {
	TrackID    string
	Quality    string
	Progress   float64
	SegmentCount int
	TotalSegments int
	Ready      bool
}

// LRU Cache implementation
type CacheItem struct {
	key   string
	value []byte
	size  int
}

type LRUCache struct {
	capacity   int
	currentSize int
	maxSize    int64 // Max memory in bytes
	items      map[string]*list.Element
	evictList  *list.List
	mu         sync.RWMutex
}

func NewLRUCache(capacity int, maxSizeBytes int64) *LRUCache {
	return &LRUCache{
		capacity:   capacity,
		maxSize:    maxSizeBytes,
		items:      make(map[string]*list.Element),
		evictList:  list.New(),
	}
}

func (c *LRUCache) Get(key string) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[key]; ok {
		c.evictList.MoveToFront(elem)
		return elem.Value.(*CacheItem).value, true
	}
	return nil, false
}

func (c *LRUCache) Put(key string, value []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()

	size := len(value)
	
	// Remove existing item if present
	if elem, ok := c.items[key]; ok {
		c.evictList.MoveToFront(elem)
		oldItem := elem.Value.(*CacheItem)
		c.currentSize -= oldItem.size
		elem.Value = &CacheItem{key: key, value: value, size: size}
		c.currentSize += size
		return
	}

	// Add new item
	item := &CacheItem{key: key, value: value, size: size}
	elem := c.evictList.PushFront(item)
	c.items[key] = elem
	c.currentSize += size

	// Evict if necessary
	c.evictIfNeeded()
}

func (c *LRUCache) evictIfNeeded() {
	for (len(c.items) > c.capacity || int64(c.currentSize) > c.maxSize) && c.evictList.Len() > 0 {
		elem := c.evictList.Back()
		if elem != nil {
			c.evictList.Remove(elem)
			item := elem.Value.(*CacheItem)
			delete(c.items, item.key)
			c.currentSize -= item.size
		}
	}
}

// Rate limiter with cleanup
type RateLimiter struct {
	requests map[string][]time.Time
	mu       sync.RWMutex
	cleanup  time.Duration
}

func NewRateLimiter(cleanup time.Duration) *RateLimiter {
	rl := &RateLimiter{
		requests: make(map[string][]time.Time),
		cleanup:  cleanup,
	}
	
	// Start cleanup goroutine
	go rl.cleanupRoutine()
	return rl
}

func (rl *RateLimiter) Allow(key string, maxRequests int, window time.Duration) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	requests := rl.requests[key]
	
	// Filter out old requests
	cutoff := now.Add(-window)
	validRequests := requests[:0]
	for _, req := range requests {
		if req.After(cutoff) {
			validRequests = append(validRequests, req)
		}
	}
	
	if len(validRequests) >= maxRequests {
		rl.requests[key] = validRequests
		return false
	}
	
	validRequests = append(validRequests, now)
	rl.requests[key] = validRequests
	return true
}

func (rl *RateLimiter) cleanupRoutine() {
	ticker := time.NewTicker(rl.cleanup)
	defer ticker.Stop()
	
	for range ticker.C {
		rl.mu.Lock()
		cutoff := time.Now().Add(-rl.cleanup * 2)
		for key, requests := range rl.requests {
			if len(requests) == 0 || requests[len(requests)-1].Before(cutoff) {
				delete(rl.requests, key)
			}
		}
		rl.mu.Unlock()
	}
}

type StreamingServer struct {
	audioDir         string
	uploadDir        string
	tracks           map[string]*AudioTrack
	segmentCache     *LRUCache
	playlistCache    *LRUCache
	rateLimiter      *RateLimiter
	processingStatus map[string]*ProcessingStatus
	mu               sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
}

func NewStreamingServer(audioDir, uploadDir string) *StreamingServer {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &StreamingServer{
		audioDir:         audioDir,
		uploadDir:        uploadDir,
		tracks:           make(map[string]*AudioTrack),
		segmentCache:     NewLRUCache(2000, 500*1024*1024), // 500MB max
		playlistCache:    NewLRUCache(1000, 50*1024*1024),  // 50MB max
		rateLimiter:      NewRateLimiter(5 * time.Minute),
		processingStatus: make(map[string]*ProcessingStatus),
		ctx:              ctx,
		cancel:           cancel,
	}
}

func (s *StreamingServer) loadTracks() {
	if _, err := os.Stat(s.audioDir); os.IsNotExist(err) {
		return
	}

	entries, err := os.ReadDir(s.audioDir)
	if err != nil {
		log.Printf("Error reading audio directory: %v", err)
		return
	}

	for _, entry := range entries {
		if entry.IsDir() {
			trackID := entry.Name()
			masterPlaylistPath := filepath.Join(s.audioDir, trackID, "playlist.m3u8")
			if _, err := os.Stat(masterPlaylistPath); err == nil {
				duration := s.getTrackDurationFromPlaylist(trackID)
				s.tracks[trackID] = &AudioTrack{
					ID:       trackID,
					Title:    strings.ReplaceAll(trackID, "_", " "),
					Artist:   "Unknown Artist",
					Duration: duration,
					Status:   "ready",
					CreatedAt: time.Now(),
					ProcessingProgress: 100.0,
				}
				log.Printf("Loaded existing track: %s", trackID)
			}
		}
	}
}

func (s *StreamingServer) getTrackDurationFromPlaylist(trackID string) int {
	playlistPath := filepath.Join(s.audioDir, trackID, "med", "playlist.m3u8")
	data, err := os.ReadFile(playlistPath)
	if err != nil {
		return 180
	}

	lines := strings.Split(string(data), "\n")
	duration := 0.0
	for _, line := range lines {
		if strings.HasPrefix(line, "#EXTINF:") {
			var segDuration float64
			fmt.Sscanf(line, "#EXTINF:%f,", &segDuration)
			duration += segDuration
		}
	}
	return int(duration)
}

func (s *StreamingServer) generateMasterPlaylist(trackID string) string {
	basePath := fmt.Sprintf("/stream/%s", trackID)
	return fmt.Sprintf(`#EXTM3U
#EXT-X-VERSION:6

#EXT-X-STREAM-INF:BANDWIDTH=48000,CODECS="mp4a.40.2",RESOLUTION=1x1
%s/low/playlist.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=96000,CODECS="mp4a.40.2",RESOLUTION=1x1
%s/med/playlist.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=192000,CODECS="mp4a.40.2",RESOLUTION=1x1
%s/high/playlist.m3u8
`, basePath, basePath, basePath)
}

func (s *StreamingServer) handleMasterPlaylist(w http.ResponseWriter, r *http.Request) {
	trackID := strings.TrimPrefix(r.URL.Path, "/stream/")
	trackID = strings.TrimSuffix(trackID, "/playlist.m3u8")

	s.mu.RLock()
	track, exists := s.tracks[trackID]
	s.mu.RUnlock()

	if !exists {
		http.Error(w, "Track not found", http.StatusNotFound)
		return
	}

	// Check cache first
	cacheKey := fmt.Sprintf("master_%s", trackID)
	if data, found := s.playlistCache.Get(cacheKey); found {
		s.writePlaylistResponse(w, data)
		return
	}

	// Set headers
	w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Cache-Control", "max-age=30") // Short cache for master playlist
	w.Header().Set("ETag", fmt.Sprintf(`"%s_%d"`, trackID, track.CreatedAt.Unix()))

	// Check if-none-match
	if match := r.Header.Get("If-None-Match"); match != "" {
		if strings.Contains(match, fmt.Sprintf(`"%s_%d"`, trackID, track.CreatedAt.Unix())) {
			w.WriteHeader(http.StatusNotModified)
			return
		}
	}

	var data []byte
	masterPlaylistPath := filepath.Join(s.audioDir, trackID, "playlist.m3u8")
	if fileData, err := os.ReadFile(masterPlaylistPath); err == nil {
		data = fileData
	} else {
		playlist := s.generateMasterPlaylist(trackID)
		data = []byte(playlist)
	}

	// Cache the playlist
	s.playlistCache.Put(cacheKey, data)
	
	w.Write(data)
	log.Printf("Served master playlist for track: %s (%s)", track.Title, trackID)
}

func (s *StreamingServer) writePlaylistResponse(w http.ResponseWriter, data []byte) {
	w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Write(data)
}

func (s *StreamingServer) handleQualityPlaylist(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 4 {
		http.Error(w, "Invalid path", http.StatusBadRequest)
		return
	}

	trackID := parts[1]
	quality := parts[2]

	s.mu.RLock()
	track, exists := s.tracks[trackID]
	s.mu.RUnlock()

	if !exists {
		http.Error(w, "Track not found", http.StatusNotFound)
		return
	}

	// Check cache first
	cacheKey := fmt.Sprintf("playlist_%s_%s", trackID, quality)
	if data, found := s.playlistCache.Get(cacheKey); found {
		s.writePlaylistResponse(w, data)
		return
	}

	playlistPath := filepath.Join(s.audioDir, trackID, quality, "playlist.m3u8")
	
	data, err := os.ReadFile(playlistPath)
	if err != nil {
		// If track is still processing, return a partial playlist
		if track.Status == "processing" {
			data = s.generatePartialPlaylist(trackID, quality)
			if data == nil {
				http.Error(w, "Playlist not ready", http.StatusNotFound)
				return
			}
		} else {
			http.Error(w, "Playlist not found", http.StatusNotFound)
			return
		}
	}

	// Cache the playlist (shorter cache for processing tracks)
	cacheTime := 300 // 5 minutes
	if track.Status == "processing" {
		cacheTime = 10 // 10 seconds for processing tracks
	}
	
	w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Cache-Control", fmt.Sprintf("max-age=%d", cacheTime))
	w.Header().Set("ETag", fmt.Sprintf(`"%s_%s_%d"`, trackID, quality, track.CreatedAt.Unix()))

	s.playlistCache.Put(cacheKey, data)
	w.Write(data)
	log.Printf("Served %s quality playlist for: %s", quality, track.Title)
}

func (s *StreamingServer) generatePartialPlaylist(trackID, quality string) []byte {
	outputDir := filepath.Join(s.audioDir, trackID, quality)
	
	// Find existing segments
	entries, err := os.ReadDir(outputDir)
	if err != nil {
		return nil
	}

	var segments []string
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".ts") {
			segments = append(segments, entry.Name())
		}
	}

	if len(segments) == 0 {
		return nil
	}

	// Generate partial playlist
	playlist := "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n"
	
	for _, segment := range segments {
		playlist += "#EXTINF:4.0,\n"
		playlist += segment + "\n"
	}

	return []byte(playlist)
}

func (s *StreamingServer) handleSegment(w http.ResponseWriter, r *http.Request) {
	clientIP := strings.Split(r.RemoteAddr, ":")[0]
	
	// Rate limiting - allow 30 requests per 10 seconds per IP
	if !s.rateLimiter.Allow(clientIP, 30, 10*time.Second) {
		http.Error(w, "Rate limited", http.StatusTooManyRequests)
		return
	}

	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 4 {
		http.Error(w, "Invalid segment path", http.StatusBadRequest)
		return
	}

	trackID := parts[1]
	quality := parts[2]
	segmentFile := parts[3]

	cacheKey := fmt.Sprintf("%s/%s/%s", trackID, quality, segmentFile)

	// Check cache
	if cachedData, exists := s.segmentCache.Get(cacheKey); exists {
		s.writeSegmentResponse(w, r, cachedData)
		return
	}

	// Load from file
	filePath := filepath.Join(s.audioDir, trackID, quality, segmentFile)
	data, err := os.ReadFile(filePath)
	if err != nil {
		http.Error(w, "Segment not found", http.StatusNotFound)
		return
	}

	// Cache the segment
	s.segmentCache.Put(cacheKey, data)
	s.writeSegmentResponse(w, r, data)
}

func (s *StreamingServer) writeSegmentResponse(w http.ResponseWriter, r *http.Request, data []byte) {
	// Support range requests for better seeking
	w.Header().Set("Content-Type", "video/MP2T")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "max-age=31536000") // 1 year cache for segments
	w.Header().Set("Accept-Ranges", "bytes")
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	
	// Add ETag for better caching
	etag := fmt.Sprintf(`"%x"`, len(data))
	w.Header().Set("ETag", etag)
	
	// Handle conditional requests
	if match := r.Header.Get("If-None-Match"); match == etag {
		w.WriteHeader(http.StatusNotModified)
		return
	}

	// Handle range requests
	if rangeHeader := r.Header.Get("Range"); rangeHeader != "" {
		s.handleRangeRequest(w, r, data, rangeHeader)
		return
	}

	w.Write(data)
}

func (s *StreamingServer) handleRangeRequest(w http.ResponseWriter, r *http.Request, data []byte, rangeHeader string) {
	// Simple range request implementation
	ranges := strings.TrimPrefix(rangeHeader, "bytes=")
	parts := strings.Split(ranges, "-")
	
	if len(parts) != 2 {
		http.Error(w, "Invalid range", http.StatusRequestedRangeNotSatisfiable)
		return
	}

	start, _ := strconv.Atoi(parts[0])
	end := len(data) - 1
	if parts[1] != "" {
		end, _ = strconv.Atoi(parts[1])
	}

	if start >= len(data) || end >= len(data) || start > end {
		http.Error(w, "Invalid range", http.StatusRequestedRangeNotSatisfiable)
		return
	}

	w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, len(data)))
	w.Header().Set("Content-Length", strconv.Itoa(end-start+1))
	w.WriteHeader(http.StatusPartialContent)
	w.Write(data[start : end+1])
}

func (s *StreamingServer) handleTrackList(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	var tracks []*AudioTrack
	for _, track := range s.tracks {
		tracks = append(tracks, track)
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "max-age=30")
	json.NewEncoder(w).Encode(tracks)
}

func (s *StreamingServer) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(100 << 20) // 100MB
	if err != nil {
		http.Error(w, "Error parsing form", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("audio")
	if err != nil {
		http.Error(w, "Error getting file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	trackID := r.FormValue("trackId")
	title := r.FormValue("title")
	artist := r.FormValue("artist")

	if trackID == "" {
		trackID = fmt.Sprintf("track_%d", time.Now().Unix())
	}
	if title == "" {
		title = strings.TrimSuffix(header.Filename, filepath.Ext(header.Filename))
	}
	if artist == "" {
		artist = "Unknown Artist"
	}

	uploadPath := filepath.Join(s.uploadDir, header.Filename)
	dst, err := os.Create(uploadPath)
	if err != nil {
		http.Error(w, "Error saving file", http.StatusInternalServerError)
		return
	}
	defer dst.Close()

	_, err = io.Copy(dst, file)
	if err != nil {
		http.Error(w, "Error copying file", http.StatusInternalServerError)
		return
	}

	log.Printf("📁 Uploaded: %s -> %s", header.Filename, uploadPath)

	// Add track to memory immediately with processing status
	s.mu.Lock()
	s.tracks[trackID] = &AudioTrack{
		ID:                 trackID,
		Title:              title,
		Artist:             artist,
		Duration:           0, // Will be updated after processing
		Status:             "processing",
		CreatedAt:          time.Now(),
		ProcessingProgress: 0.0,
	}
	s.mu.Unlock()

	// Process audio in background with progressive streaming
	go func() {
		err := s.processAudioFileProgressive(uploadPath, trackID, title, artist)
		if err != nil {
			log.Printf("❌ Error processing %s: %v", trackID, err)
			s.mu.Lock()
			s.tracks[trackID].Status = "error"
			s.mu.Unlock()
		} else {
			log.Printf("✅ Successfully processed: %s", trackID)
		}
	}()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":  true,
		"trackId":  trackID,
		"message":  "Upload successful, processing audio...",
		"filename": header.Filename,
	})
}

func (s *StreamingServer) processAudioFileProgressive(inputPath, trackID, title, artist string) error {
	trackDir := filepath.Join(s.audioDir, trackID)
	os.MkdirAll(filepath.Join(trackDir, "low"), 0755)
	os.MkdirAll(filepath.Join(trackDir, "med"), 0755)
	os.MkdirAll(filepath.Join(trackDir, "high"), 0755)

	log.Printf("🎵 Processing audio: %s", inputPath)

	duration, err := s.getAudioDuration(inputPath)
	if err != nil {
		return fmt.Errorf("failed to get duration: %v", err)
	}

	// Update track with duration
	s.mu.Lock()
	s.tracks[trackID].Duration = duration
	s.mu.Unlock()

	// Process qualities in parallel
	qualities := []struct {
		name    string
		bitrate string
	}{
		{"low", "48k"},
		{"med", "96k"},
		{"high", "192k"},
	}

	var wg sync.WaitGroup
	errors := make(chan error, len(qualities))

	for _, q := range qualities {
		wg.Add(1)
		go func(quality struct {
			name    string
			bitrate string
		}) {
			defer wg.Done()
			err := s.processQuality(inputPath, trackID, quality.name, quality.bitrate)
			if err != nil {
				errors <- fmt.Errorf("failed to process %s quality: %v", quality.name, err)
			}
		}(q)
	}

	// Wait for all qualities to complete
	go func() {
		wg.Wait()
		close(errors)
	}()

	// Check for errors
	for err := range errors {
		if err != nil {
			return err
		}
	}

	// Generate master playlist
	masterPlaylistPath := filepath.Join(trackDir, "playlist.m3u8")
	masterPlaylistContent := s.generateMasterPlaylist(trackID)
	if err := os.WriteFile(masterPlaylistPath, []byte(masterPlaylistContent), 0644); err != nil {
		return fmt.Errorf("failed to write master playlist: %v", err)
	}

	// Update track status
	s.mu.Lock()
	s.tracks[trackID].Status = "ready"
	s.tracks[trackID].ProcessingProgress = 100.0
	s.mu.Unlock()

	return nil
}

func (s *StreamingServer) processQuality(inputPath, trackID, quality, bitrate string) error {
	outputDir := filepath.Join(s.audioDir, trackID, quality)
	playlistPath := filepath.Join(outputDir, "playlist.m3u8")

	cmd := exec.Command("ffmpeg", "-i", inputPath,
		"-c:a", "aac",
		"-b:a", bitrate,
		"-f", "hls",
		"-hls_time", "4",                    // 4-second segments for faster start
		"-hls_list_size", "0",
		"-hls_segment_filename", filepath.Join(outputDir, "segment_%03d.ts"),
		"-hls_playlist_type", "vod",
		"-hls_flags", "independent_segments", // Better for seeking
		"-y",
		playlistPath,
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg failed: %v - %s", err, output)
	}

	log.Printf("✅ Generated %s quality segments for %s", quality, trackID)
	return nil
}

func (s *StreamingServer) getAudioDuration(inputPath string) (int, error) {
	cmd := exec.Command("ffprobe", "-v", "quiet", "-show_entries",
		"format=duration", "-of", "csv=p=0", inputPath)

	output, err := cmd.Output()
	if err != nil {
		return 180, nil
	}

	duration := strings.TrimSpace(string(output))
	if duration == "" {
		return 180, nil
	}

	var durationFloat float64
	fmt.Sscanf(duration, "%f", &durationFloat)
	return int(durationFloat), nil
}

func (s *StreamingServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	
	s.mu.RLock()
	trackCount := len(s.tracks)
	processingCount := 0
	for _, track := range s.tracks {
		if track.Status == "processing" {
			processingCount++
		}
	}
	s.mu.RUnlock()

	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":           "healthy",
		"time":             time.Now().Format(time.RFC3339),
		"totalTracks":      trackCount,
		"processingTracks": processingCount,
		"cacheStats": map[string]interface{}{
			"segmentCacheSize":  len(s.segmentCache.items),
			"playlistCacheSize": len(s.playlistCache.items),
		},
	})
}

func (s *StreamingServer) handleCORS(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Range")
	w.WriteHeader(http.StatusOK)
}

func (s *StreamingServer) Shutdown() {
	s.cancel()
}

func main() {
	audioDir := os.Getenv("AUDIO_DIR")
	if audioDir == "" {
		audioDir = "./processed_audio"
	}

	uploadDir := os.Getenv("UPLOAD_DIR")
	if uploadDir == "" {
		uploadDir = "./uploads"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Create directories
	os.MkdirAll(audioDir, 0755)
	os.MkdirAll(uploadDir, 0755)
	os.MkdirAll("./static", 0755)

	server := NewStreamingServer(audioDir, uploadDir)
	server.loadTracks()

	// Routes
	http.HandleFunc("/stream/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			server.handleCORS(w, r)
			return
		}

		path := r.URL.Path

		switch {
		case strings.HasSuffix(path, "/playlist.m3u8") && strings.Count(path, "/") == 3:
			server.handleMasterPlaylist(w, r)
		case strings.HasSuffix(path, "/playlist.m3u8"):
			server.handleQualityPlaylist(w, r)
		case strings.HasSuffix(path, ".ts"):
			server.handleSegment(w, r)
		default:
			http.Error(w, "Invalid stream path", http.StatusBadRequest)
		}
	})

	http.HandleFunc("/api/tracks", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			server.handleCORS(w, r)
			return
		}
		server.handleTrackList(w, r)
	})

	http.HandleFunc("/api/upload", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			server.handleCORS(w, r)
			return
		}
		server.handleUpload(w, r)
	})

	http.HandleFunc("/health", server.handleHealth)

	// Serve static files (the web interface)
	http.Handle("/", http.FileServer(http.Dir("./static/")))

	log.Printf("🚀 Starting adaptive audio streaming server on port %s", port)
	log.Printf("📁 Audio directory: %s", audioDir)
	log.Printf("📤 Upload directory: %s", uploadDir)
	log.Printf("🎵 Stream URL format: http://localhost:%s/stream/TRACK_ID/playlist.m3u8", port)
	log.Printf("📱 Web interface: http://localhost:%s", port)

	log.Fatal(http.ListenAndServe(":"+port, nil))
}