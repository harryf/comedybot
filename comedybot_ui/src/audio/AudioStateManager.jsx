class AudioStateManager {
  constructor() {
    this.observers = new Set();
    this.state = {
      currentTime: 0,
      duration: 0,
      isPlaying: false,
      isStopped: true,
      isLoaded: false
    };
  }

  initialize() {
    this.updateState({
      isLoaded: true,
      isStopped: true,
      isPlaying: false,
      currentTime: 0
    });
  }

  subscribe(observer) {
    this.observers.add(observer);
    observer(this.state);
    return () => this.observers.delete(observer);
  }

  notify() {
    this.observers.forEach(observer => observer({...this.state}));
  }

  updateState(newState) {
    const prevState = {...this.state};
    this.state = { ...this.state, ...newState };
    
    // Only notify if state actually changed
    if (JSON.stringify(prevState) !== JSON.stringify(this.state)) {
      this.notify();
    }
  }

  play() {
    if (!this.state.isLoaded) return;
    this.updateState({ 
      isPlaying: true, 
      isStopped: false 
    });
  }

  pause() {
    this.updateState({ 
      isPlaying: false 
    });
  }

  stop() {
    this.updateState({ 
      isPlaying: false, 
      isStopped: true,
      currentTime: 0 
    });
  }

  seek(time) {
    if (!this.state.isLoaded) return;
    this.updateState({ 
      currentTime: time,
      isStopped: false 
    });
  }
}

export default AudioStateManager; 