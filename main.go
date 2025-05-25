package main

import (
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
	ID       string `json:"id"`
	Title    string `json:"title"`
	Artist   string `json:"artist"`
	Duration int    `json:"duration"` // in seconds
}

type StreamingServer struct {
	audioDir    string
	uploadDir   string
	tracks      map[string]*AudioTrack
	segmentCache map[string][]byte // In-memory cache for hot segments
	rateLimiter  map[string]time.Time // Simple rate limiting
	mu          sync.RWMutex
}

func NewStreamingServer(audioDir, uploadDir string) *StreamingServer {
	return &StreamingServer{
		audioDir:     audioDir,
		uploadDir:    uploadDir,
		tracks:       make(map[string]*AudioTrack),
		segmentCache: make(map[string][]byte),
		rateLimiter:  make(map[string]time.Time),
	}
}

// Load track metadata (you'd typically get this from your database)
func (s *StreamingServer) loadTracks() {
	// Example tracks - replace with your database logic
	s.tracks["track1"] = &AudioTrack{
		ID:       "track1",
		Title:    "Sample Song",
		Artist:   "Sample Artist",
		Duration: 180,
	}
	// Add more tracks...
}

// Generate master playlist for a track
func (s *StreamingServer) generateMasterPlaylist(trackID string) string {
	return fmt.Sprintf(`#EXTM3U
#EXT-X-VERSION:3

#EXT-X-STREAM-INF:BANDWIDTH=40000,CODECS="opus"
%s/low/playlist.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=80000,CODECS="opus"
%s/med/playlist.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=120000,CODECS="opus"
%s/high/playlist.m3u8
`, trackID, trackID, trackID)
}

// Generate quality-specific playlist
func (s *StreamingServer) generateQualityPlaylist(trackID, quality string, duration int) string {
	segments := duration / 10 // 10-second segments
	if duration%10 != 0 {
		segments++
	}

	playlist := `#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-PLAYLIST-TYPE:VOD
`

	for i := 0; i < segments; i++ {
		playlist += fmt.Sprintf("#EXTINF:10.0,\nsegment_%03d.webm\n", i)
	}

	playlist += "#EXT-X-ENDLIST\n"
	return playlist
}

// Handle master playlist requests
func (s *StreamingServer) handleMasterPlaylist(w http.ResponseWriter, r *http.Request) {
	trackID := strings.TrimPrefix(r.URL.Path, "/stream/")
	trackID = strings.TrimSuffix(trackID, "/playlist.m3u8")

	track, exists := s.tracks[trackID]
	if !exists {
		http.Error(w, "Track not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "max-age=3600")

	playlist := s.generateMasterPlaylist(trackID)
	w.Write([]byte(playlist))

	log.Printf("Served master playlist for track: %s (%s)", track.Title, trackID)
}

// Handle quality-specific playlist requests
func (s *StreamingServer) handleQualityPlaylist(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 4 {
		http.Error(w, "Invalid path", http.StatusBadRequest)
		return
	}

	trackID := parts[1]
	quality := parts[2] // low, med, high

	track, exists := s.tracks[trackID]
	if !exists {
		http.Error(w, "Track not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "max-age=3600")

	playlist := s.generateQualityPlaylist(trackID, quality, track.Duration)
	w.Write([]byte(playlist))

	log.Printf("Served %s quality playlist for: %s", quality, track.Title)
}

// Handle audio segment requests with caching and rate limiting
func (s *StreamingServer) handleSegment(w http.ResponseWriter, r *http.Request) {
	// Simple rate limiting per IP
	clientIP := r.RemoteAddr
	s.mu.Lock()
	if lastRequest, exists := s.rateLimiter[clientIP]; exists {
		if time.Since(lastRequest) < 100*time.Millisecond {
			s.mu.Unlock()
			http.Error(w, "Rate limited", http.StatusTooManyRequests)
			return
		}
	}
	s.rateLimiter[clientIP] = time.Now()
	s.mu.Unlock()

	// Extract path: /stream/trackID/quality/segment_001.webm
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 4 {
		http.Error(w, "Invalid segment path", http.StatusBadRequest)
		return
	}

	trackID := parts[1]
	quality := parts[2]
	segmentFile := parts[3]

	// Create cache key
	cacheKey := fmt.Sprintf("%s/%s/%s", trackID, quality, segmentFile)

	// Check cache first
	s.mu.RLock()
	if cachedData, exists := s.segmentCache[cacheKey]; exists {
		s.mu.RUnlock()
		
		w.Header().Set("Content-Type", "audio/webm")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Cache-Control", "max-age=86400")
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Length", strconv.Itoa(len(cachedData)))
		
		w.Write(cachedData)
		log.Printf("Served cached segment: %s", cacheKey)
		return
	}
	s.mu.RUnlock()

	// Build file path
	filePath := filepath.Join(s.audioDir, trackID, quality, segmentFile)

	// Check if file exists
	file, err := os.Open(filePath)
	if err != nil {
		http.Error(w, "Segment not found", http.StatusNotFound)
		return
	}
	defer file.Close()

	// Read file data
	data, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Error reading segment", http.StatusInternalServerError)
		return
	}

	// Cache the segment (limit cache size to prevent memory issues)
	s.mu.Lock()
	if len(s.segmentCache) < 1000 { // Max 1000 cached segments
		s.segmentCache[cacheKey] = data
	}
	s.mu.Unlock()

	// Set appropriate headers
	w.Header().Set("Content-Type", "audio/webm")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Cache-Control", "max-age=86400")
	w.Header().Set("Accept-Ranges", "bytes")
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))

	// Serve the data
	w.Write(data)

	log.Printf("Served segment: %s (cached for future requests)", cacheKey)
}

// API endpoint to get track list
func (s *StreamingServer) handleTrackList(w http.ResponseWriter, r *http.Request) {
	var tracks []*AudioTrack
	for _, track := range s.tracks {
		tracks = append(tracks, track)
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(tracks)
}

// Handle file upload and processing
func (s *StreamingServer) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form (max 50MB)
	err := r.ParseMultipartForm(50 << 20)
	if err != nil {
		http.Error(w, "Error parsing form", http.StatusBadRequest)
		return
	}

	// Get the uploaded file
	file, header, err := r.FormFile("audio")
	if err != nil {
		http.Error(w, "Error getting file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Get track metadata from form
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

	// Save uploaded file
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

	// Process audio in background
	go func() {
		err := s.processAudioFile(uploadPath, trackID, title, artist)
		if err != nil {
			log.Printf("❌ Error processing %s: %v", trackID, err)
		} else {
			log.Printf("✅ Successfully processed: %s", trackID)
		}
	}()

	// Respond immediately
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":  true,
		"trackId":  trackID,
		"message":  "Upload successful, processing audio...",
		"filename": header.Filename,
	})
}

// Process uploaded audio file into adaptive segments
func (s *StreamingServer) processAudioFile(inputPath, trackID, title, artist string) error {
	// Create output directories
	trackDir := filepath.Join(s.audioDir, trackID)
	os.MkdirAll(filepath.Join(trackDir, "low"), 0755)
	os.MkdirAll(filepath.Join(trackDir, "med"), 0755)
	os.MkdirAll(filepath.Join(trackDir, "high"), 0755)

	log.Printf("🎵 Processing audio: %s", inputPath)

	// Get audio duration first
	duration, err := s.getAudioDuration(inputPath)
	if err != nil {
		return fmt.Errorf("failed to get duration: %v", err)
	}

	// FFmpeg command to create all qualities at once
	cmd := exec.Command("ffmpeg", "-i", inputPath,
		// Low quality (32kbps Opus)
		"-map", "0:a", "-c:a", "libopus", "-b:a", "32k",
		"-f", "segment", "-segment_time", "10",
		"-segment_list", filepath.Join(trackDir, "low", "playlist.m3u8"),
		"-segment_list_flags", "+live",
		filepath.Join(trackDir, "low", "segment_%03d.webm"),

		// Medium quality (64kbps Opus)
		"-map", "0:a", "-c:a", "libopus", "-b:a", "64k",
		"-f", "segment", "-segment_time", "10",
		"-segment_list", filepath.Join(trackDir, "med", "playlist.m3u8"),
		"-segment_list_flags", "+live",
		filepath.Join(trackDir, "med", "segment_%03d.webm"),

		// High quality (96kbps Opus)
		"-map", "0:a", "-c:a", "libopus", "-b:a", "96k",
		"-f", "segment", "-segment_time", "10",
		"-segment_list", filepath.Join(trackDir, "high", "playlist.m3u8"),
		"-segment_list_flags", "+live",
		filepath.Join(trackDir, "high", "segment_%03d.webm"),
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg failed: %v - %s", err, output)
	}

	
    // ✅ Generate master playlist after FFmpeg succeeds
    masterPlaylistPath := filepath.Join(trackDir, "playlist.m3u8")
    masterPlaylistContent := s.generateMasterPlaylist(trackID)
    
    if err := os.WriteFile(masterPlaylistPath, []byte(masterPlaylistContent), 0644); err != nil {
        return fmt.Errorf("failed to write master playlist: %v", err)
    }

	// Add track to memory
	s.mu.Lock()
	s.tracks[trackID] = &AudioTrack{
		ID:       trackID,
		Title:    title,
		Artist:   artist,
		Duration: duration,
	}
	s.mu.Unlock()

	log.Printf("✅ Audio processing complete: %s (%d seconds)", trackID, duration)
	return nil
}

// Get audio duration using ffprobe
func (s *StreamingServer) getAudioDuration(inputPath string) (int, error) {
	cmd := exec.Command("ffprobe", "-v", "quiet", "-show_entries",
		"format=duration", "-of", "csv=p=0", inputPath)
	
	output, err := cmd.Output()
	if err != nil {
		return 180, nil // Default to 3 minutes if can't detect
	}

	duration := strings.TrimSpace(string(output))
	if duration == "" {
		return 180, nil
	}

	// Parse duration (it's in seconds as float)
	var durationFloat float64
	fmt.Sscanf(duration, "%f", &durationFloat)
	return int(durationFloat), nil
}

// Health check endpoint
func (s *StreamingServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
		"time":   time.Now().Format(time.RFC3339),
	})
}

func main() {
	audioDir := os.Getenv("AUDIO_DIR")
	if audioDir == "" {
		audioDir = "./audio" // Default directory
	}

	uploadDir := os.Getenv("UPLOAD_DIR")
	if uploadDir == "" {
		uploadDir = "./uploads" // Default upload directory
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Create directories if they don't exist
	os.MkdirAll(audioDir, 0755)
	os.MkdirAll(uploadDir, 0755)
	os.MkdirAll("./static", 0755)

	server := NewStreamingServer(audioDir, uploadDir)
	server.loadTracks()

	// Routes
	http.HandleFunc("/stream/", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path

		switch {
		case strings.HasSuffix(path, "/playlist.m3u8") && strings.Count(path, "/") == 2:
			// Master playlist: /stream/trackID/playlist.m3u8
			server.handleMasterPlaylist(w, r)
		case strings.HasSuffix(path, "/playlist.m3u8"):
			// Quality playlist: /stream/trackID/quality/playlist.m3u8
			server.handleQualityPlaylist(w, r)
		case strings.HasSuffix(path, ".webm"):
			// Audio segment: /stream/trackID/quality/segment_001.webm
			server.handleSegment(w, r)
		default:
			http.Error(w, "Invalid stream path", http.StatusBadRequest)
		}
	})

	http.HandleFunc("/api/tracks", server.handleTrackList)
	http.HandleFunc("/api/upload", server.handleUpload)
	http.HandleFunc("/health", server.handleHealth)

	// Serve static files (your web app)
	http.Handle("/", http.FileServer(http.Dir("./static/")))

	log.Printf("🚀 Starting adaptive audio streaming server on port %s", port)
	log.Printf("📁 Audio directory: %s", audioDir)
	log.Printf("📤 Upload directory: %s", uploadDir)
	log.Printf("🎵 Example stream URL: http://localhost:%s/stream/track1/playlist.m3u8", port)
	log.Printf("📱 Web interface: http://localhost:%s", port)

	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// Directory structure should be:
// audio/
//   track1/
//     low/
//       playlist.m3u8
//       segment_000.webm
//       segment_001.webm
//       ...
//     med/
//       playlist.m3u8  
//       segment_000.webm
//       ...
//     high/
//       playlist.m3u8
//       segment_000.webm
//       ...