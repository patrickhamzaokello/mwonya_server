let currentTrackId = null;
let hls = null; // HLS.js instance

async function loadTracks() {
    try {
        const response = await fetch('/api/tracks');
        const tracks = await response.json();
        
        const tracksList = document.getElementById('tracksList');
        
        if (tracks.length === 0) {
            tracksList.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">♪</div>
                    <div>No tracks uploaded yet</div>
                </div>
            `;
            return;
        }

        tracksList.innerHTML = tracks.map(track => `
            <div class="track-item ${currentTrackId === track.id ? 'active' : ''}" 
                 onclick="playTrack('${track.id}', '${track.title}')">
                <div class="track-name">${track.title}</div>
                <div class="track-meta">
                    <span>${track.duration || '0:00'} • ${track.size || '0 MB'}</span>
                    <span class="track-status ${currentTrackId === track.id ? 'status-playing' : 'status-ready'}">
                        ${currentTrackId === track.id ? '▶ Playing' : '✓ Ready'}
                    </span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading tracks:', error);
        tracksList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <div>Error loading tracks</div>
            </div>
        `;
    }
}

function playTrack(trackId, title) {
    const audio = document.getElementById('audioPlayer');
    const streamUrl = `/stream/${trackId}/playlist.m3u8`;
    
    // Destroy previous HLS instance if exists
    if (hls) {
        hls.destroy();
    }
    
    // Update previous track status
    if (currentTrackId) {
        const prevTrackItem = document.querySelector(`.track-item[onclick*="${currentTrackId}"]`);
        if (prevTrackItem) {
            prevTrackItem.classList.remove('active');
            const statusSpan = prevTrackItem.querySelector('.track-status');
            if (statusSpan) {
                statusSpan.className = 'track-status status-ready';
                statusSpan.textContent = '✓ Ready';
            }
        }
    }
    
    // Update current track
    currentTrackId = trackId;
    const currentTrackItem = document.querySelector(`.track-item[onclick*="${trackId}"]`);
    if (currentTrackItem) {
        currentTrackItem.classList.add('active');
        const statusSpan = currentTrackItem.querySelector('.track-status');
        if (statusSpan) {
            statusSpan.className = 'track-status status-playing';
            statusSpan.textContent = '▶ Playing';
        }
    }
    
    if (Hls.isSupported()) {
        // Create new HLS instance
        hls = new Hls();
        hls.loadSource(streamUrl);
        hls.attachMedia(audio);
        hls.on(Hls.Events.MANIFEST_PARSED, () => {
            audio.play().catch(e => {
                console.error('Auto-play failed:', e);
                // Show play button or handle error
            });
        });
        
        hls.on(Hls.Events.ERROR, (event, data) => {
            console.error('HLS error:', data);
            if (data.fatal) {
                switch(data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR:
                        console.error('Network error, trying to recover');
                        hls.startLoad();
                        break;
                    case Hls.ErrorTypes.MEDIA_ERROR:
                        console.error('Media error, trying to recover');
                        hls.recoverMediaError();
                        break;
                    default:
                        console.error('Unrecoverable error');
                        initAudioFallback(audio, streamUrl);
                        break;
                }
            }
        });
    } else if (audio.canPlayType('application/vnd.apple.mpegurl')) {
        // Native HLS support (Safari)
        audio.src = streamUrl;
        audio.addEventListener('loadedmetadata', () => {
            audio.play().catch(e => {
                console.error('Auto-play failed:', e);
            });
        });
    } else {
        console.error('HLS not supported in this browser');
        initAudioFallback(audio, streamUrl);
    }
    
    document.getElementById('currentTrack').textContent = title;
    document.getElementById('currentTrack').classList.add('playing');
    document.getElementById('playbackStatus').textContent = 'Playing';
    
    console.log(`🎵 Playing: ${title} (${streamUrl})`);
}

function initAudioFallback(audioElement, streamUrl) {
    // Fallback for unsupported browsers - try to play directly (may not work for HLS)
    console.warn('Falling back to direct audio playback');
    audioElement.src = streamUrl;
    audioElement.play().catch(e => {
        console.error('Fallback playback failed:', e);
        alert('Error: Your browser doesnt support HLS streaming. Try Safari or Chrome with hls.js.');
    });
}

async function uploadAudio() {
    const fileInput = document.getElementById('audioUpload');
    const uploadBtn = document.querySelector('.upload-btn');
    const uploadText = document.getElementById('upload-text');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an audio file');
        return;
    }
    
    // Show uploading state
    uploadBtn.disabled = true;
    uploadText.innerHTML = '<span class="loading"></span>Uploading...';
    
    const formData = new FormData();
    formData.append('audio', file);
    formData.append('trackId', `track_${Date.now()}`);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            uploadText.textContent = '✓ Uploaded!';
            setTimeout(() => {
                uploadText.textContent = 'Upload Audio';
                uploadBtn.disabled = false;
                clearSelection();
                loadTracks(); // Refresh tracks after upload
            }, 2000);
        } else {
            uploadText.textContent = 'Upload failed';
            uploadBtn.disabled = false;
        }
    } catch (error) {
        console.error('Upload error:', error);
        uploadText.textContent = 'Upload error';
        uploadBtn.disabled = false;
    }
}

// Handle file selection display
document.getElementById('audioUpload').addEventListener('change', function(e) {
    const fileLabel = document.getElementById('file-label-text');
    const fileLabelContainer = document.querySelector('.file-label');
    
    if (e.target.files.length > 0) {
        const fileName = e.target.files[0].name;
        const fileSize = (e.target.files[0].size / (1024 * 1024)).toFixed(1);
        fileLabel.innerHTML = `📄 ${fileName} <small>(${fileSize} MB)</small>`;
        fileLabelContainer.classList.add('file-selected');
    } else {
        fileLabel.textContent = '📁 Select audio file or drag & drop';
        fileLabelContainer.classList.remove('file-selected');
    }
});

// Clear file selection
function clearSelection() {
    document.getElementById('audioUpload').value = '';
    document.getElementById('file-label-text').textContent = '📁 Select audio file or drag & drop';
    document.querySelector('.file-label').classList.remove('file-selected');
}

// Audio player events
const audioPlayer = document.getElementById('audioPlayer');
audioPlayer.addEventListener('ended', () => {
    if (currentTrackId) {
        const trackItem = document.querySelector(`.track-item[onclick*="${currentTrackId}"]`);
        if (trackItem) {
            trackItem.classList.remove('active');
            const statusSpan = trackItem.querySelector('.track-status');
            if (statusSpan) {
                statusSpan.className = 'track-status status-ready';
                statusSpan.textContent = '✓ Ready';
            }
        }
        document.getElementById('playbackStatus').textContent = 'Finished';
    }
});

// Refresh tracks
function refreshTracks() {
    const btn = event.target;
    btn.textContent = '↻';
    setTimeout(() => {
        loadTracks();
        btn.textContent = 'Refresh';
    }, 500);
}

// Load HLS.js library dynamically
function loadHlsJs() {
    return new Promise((resolve, reject) => {
        if (window.Hls) {
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/hls.js@latest';
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Initialize the player
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadHlsJs();
        loadTracks();
    } catch (error) {
        console.error('Failed to load HLS.js:', error);
        alert('Error: Required streaming support could not be loaded. Please try refreshing.');
    }
});