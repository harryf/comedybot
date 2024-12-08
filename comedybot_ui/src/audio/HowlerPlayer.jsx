import { Howl } from 'howler';
import useStore from '../store/useStore.jsx';

class HowlerPlayer {
  constructor() {
    this.howl = null;
    this.store = useStore.getState();
    this.isReady = false;
    
    // Subscribe to store updates
    useStore.subscribe(
      (state) => state.audioState,
      (audioState) => {
        console.log('Audio state updated:', audioState);
      }
    );
  }

  initialize() {
    console.log('Initializing HowlerPlayer');
    
    if (this.howl) {
      this.howl.unload();
    }

    return new Promise((resolve) => {
      const { sounds } = this.store;
      if (!sounds?.main) {
        console.error('No audio file path specified in sounds');
        return;
      }

      this.howl = new Howl({
        src: [sounds.main],
        html5: true,
        preload: true,
        onload: () => {
          console.log('Audio loaded');
          this.isReady = true;
          this.store.setAudioState({
            duration: this.howl.duration(),
            isLoaded: true
          });
          resolve();
        },
        onloaderror: (id, error) => {
          console.error('Error loading audio:', error);
        },
        onplay: () => {
          console.log('Audio playing');
          this.store.setAudioState({ isPlaying: true });
          this._startTimeUpdate();
        },
        onpause: () => {
          console.log('Audio paused');
          this.store.setAudioState({ isPlaying: false });
          this._stopTimeUpdate();
        },
        onstop: () => {
          console.log('Audio stopped');
          this.store.setAudioState({
            isPlaying: false,
            currentTime: 0
          });
          this._stopTimeUpdate();
        },
        onend: () => {
          console.log('Audio ended');
          this.store.setAudioState({
            isPlaying: false,
            currentTime: this.howl.duration()
          });
          this._stopTimeUpdate();
        },
        onseek: () => {
          console.log('Audio seeked to:', this.howl.seek());
          this.store.setAudioState({
            currentTime: this.howl.seek()
          });
        }
      });

      // Store the player instance in the store
      this.store.setPlayer(this);
    });
  }

  play() {
    console.log('Play requested');
    if (this.howl) {
      this.howl.play();
    }
  }

  pause() {
    console.log('Pause requested');
    if (this.howl) {
      this.howl.pause();
    }
  }

  stop() {
    console.log('Stop requested');
    if (this.howl) {
      this.howl.stop();
    }
  }

  seek(time) {
    console.log('Seek requested:', time);
    if (this.howl) {
      this.howl.seek(time);
    }
  }

  _startTimeUpdate() {
    this._stopTimeUpdate();
    this.timeUpdateInterval = setInterval(() => {
      if (this.howl && this.howl.playing()) {
        this.store.setAudioState({
          currentTime: this.howl.seek()
        });
      }
    }, 100);
  }

  _stopTimeUpdate() {
    if (this.timeUpdateInterval) {
      clearInterval(this.timeUpdateInterval);
      this.timeUpdateInterval = null;
    }
  }
}

export default HowlerPlayer;