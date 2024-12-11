// Get the base path for audio files, considering the Jekyll variables
export function getAudioBasePath() {
  // Check if we're running in the Jekyll environment
  if (window.TRANSCRIPT_PATH && window.BASE_URL) {
    // Extract the directory path from TRANSCRIPT_PATH
    const pathParts = window.TRANSCRIPT_PATH.split('/');
    pathParts.pop(); // Remove the filename
    return `${window.BASE_URL}/assets/audio/${pathParts.join('/')}`;
  }
  
  // Default path for local development
  return '/audio';
}

// Get the full path for a specific audio file or JSON
export function getAudioPath(filename) {
  return `${getAudioBasePath()}/${filename}`;
}
