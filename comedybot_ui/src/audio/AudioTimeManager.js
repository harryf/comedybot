class AudioTimeManager {
  constructor(debug = false) {
    this.currentTime = 0;
    this.totalDuration = 0;
    this.segmentDuration = 10; // seconds
    this.isPlaying = false;
    this.lastUpdateTime = null;
    this.updateInterval = null;
    this.onTimeUpdate = null;
    this.debug = debug;
  }

  _log(...args) {
    if (this.debug) {
      console.log('[AudioTimeManager]', ...args);
    }
  }

  initialize(totalDuration) {
    this.totalDuration = totalDuration;
    this._log('Initialized with duration:', totalDuration);
  }

  // Start tracking time
  start() {
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.lastUpdateTime = Date.now();
      this._startTimeUpdate();
      this._log('Started time tracking at:', this.currentTime);
    }
  }

  // Pause time tracking
  pause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this._stopTimeUpdate();
      this._log('Paused at:', this.currentTime);
    }
  }

  // Stop and reset time
  stop() {
    this.isPlaying = false;
    this._stopTimeUpdate();
    this.setTime(0);
    this._log('Stopped and reset time');
  }

  // Set the current time directly
  setTime(time) {
    this.currentTime = Math.max(0, Math.min(time, this.totalDuration));
    this.lastUpdateTime = Date.now();
    if (this.onTimeUpdate) {
      this.onTimeUpdate(this.currentTime);
    }
    this._log('Time set to:', this.currentTime);
  }

  // Get current segment index
  getCurrentSegmentIndex() {
    return Math.floor(this.currentTime / this.segmentDuration);
  }

  // Get time within current segment
  getCurrentSegmentTime() {
    return this.currentTime % this.segmentDuration;
  }

  // Set callback for time updates
  setTimeUpdateCallback(callback) {
    this.onTimeUpdate = callback;
  }

  // Internal method to start time update interval
  _startTimeUpdate() {
    this._stopTimeUpdate();
    this.updateInterval = setInterval(() => {
      if (this.isPlaying) {
        const now = Date.now();
        const delta = (now - this.lastUpdateTime) / 1000; // Convert to seconds
        this.lastUpdateTime = now;

        // Update time
        const newTime = Math.min(this.currentTime + delta, this.totalDuration);
        if (newTime !== this.currentTime) {
          this.currentTime = newTime;
          if (this.onTimeUpdate) {
            this.onTimeUpdate(this.currentTime);
          }
        }

        // Stop if we've reached the end
        if (this.currentTime >= this.totalDuration) {
          this.pause();
        }
      }
    }, 100); // Update more frequently than display for smooth timing
  }

  // Internal method to stop time update interval
  _stopTimeUpdate() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}

export default AudioTimeManager;
