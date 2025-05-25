let currentAudio = null;

async function loadTracks() {
    try {
        const response = await fetch('/api/tracks');
        const tracks = await response.json();
        
        const tracksList = document.getElementById('tracksList');
        tracksList.innerHTML = '';
        
        tracks.forEach(track => {
            const trackDiv = document.createElement('div');
            trackDiv.className = 'track-item';
            trackDiv.innerHTML = `
                <h3>${track.title}</h3>
                <p>by ${track.artist}</p>
                <button onclick="playTrack('${track.id}', '${track.title}')">▶️ Play</button>
            `;
            tracksList.appendChild(trackDiv);
        });
    } catch (error) {
        console.error('Error loading tracks:', error);
    }
}

function playTrack(trackId, title) {
    const audio = document.getElementById('audioPlayer');
    const streamUrl = `/stream/${trackId}/playlist.m3u8`;
    
    audio.src = streamUrl;
    audio.play();
    
    document.getElementById('currentTrack').textContent = `Now Playing: ${title}`;
    console.log(`🎵 Playing: ${title} (${streamUrl})`);
}

async function uploadAudio() {
    const fileInput = document.getElementById('audioUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an audio file');
        return;
    }
    
    const formData = new FormData();
    formData.append('audio', file);
    formData.append('trackId', `track_${Date.now()}`);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            alert('Upload successful! Processing audio...');
            setTimeout(loadTracks, 5000); // Refresh tracks after processing
        } else {
            alert('Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload error');
    }
}

// Load tracks on page load
document.addEventListener('DOMContentLoaded', loadTracks);