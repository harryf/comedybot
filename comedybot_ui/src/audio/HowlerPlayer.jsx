import { Howl } from 'howler';
import useStore from '../store/useStore.js';
import AudioSegmentManager from './AudioSegmentManager';
import AudioTimeManager from './AudioTimeManager';
import debounce from 'lodash.debounce';
import Logger from '../utils/logger';

class HowlerPlayer {
  constructor(debug = false) {
    this.currentHowl = null;
    this.store = useStore.getState();
    this.isReady = false;
    this.segmentManager = new AudioSegmentManager(debug);
    this.timeManager = new AudioTimeManager(debug);
    this.currentSegmentIndex = -1;
    this.isTransitioning = false;
    this.transitionPromise = null;
    this.debug = debug;
    this.preloadStarted = false;

    // Set up time update callback
    this.timeManager.setTimeUpdateCallback((time) => {
      this.store.setAudioState({ currentTime: time });
      this._checkAndPreloadNext(time);
    });

    useStore.subscribe(
      (state) => state.audioState,
      (audioState) => {
        if (this.debug) {
          Logger.debug('[HowlerPlayer] Audio state updated:', audioState);
        }
      }
    );
  }

  _log(...args) {
    if (this.debug) {
      Logger.debug('[HowlerPlayer]', ...args);
    }
  }

  _error(...args) {
    Logger.error('[HowlerPlayer]', ...args);
  }

  async initialize() {
    this._log('Initializing HowlerPlayer');
    
    if (this.currentHowl) {
      this.currentHowl.unload();
    }

    return new Promise(async (resolve, reject) => {
      try {
        const { metadata } = this.store;
        if (!metadata?.segments) {
          throw new Error('No segments specified in metadata');
        }

        const totalDuration = await this.segmentManager.initialize(metadata);
        this.timeManager.initialize(totalDuration);
        
        const segment = await this.segmentManager.loadSegmentsAroundTime(0);
        
        if (segment?.howl) {
          this._setupHowlEvents(segment.howl, segment.segmentIndex);
          this.currentHowl = segment.howl;
          this.currentSegmentIndex = segment.segmentIndex;
          
          this.isReady = true;
          this.store.setAudioState({
            duration: totalDuration,
            isLoaded: true,
            currentTime: 0
          });
          resolve();
        } else {
          throw new Error('Failed to load initial segment');
        }
      } catch (error) {
        this._error('Initialization failed:', error);
        reject(error);
      }
    });
  }

  _setupHowlEvents(howl, segmentIndex) {
    // Clean up existing event listeners
    howl.off('play');
    howl.off('pause');
    howl.off('end');
    howl.off('loaderror');

    howl.on('play', () => {
      this._log('Segment playing:', segmentIndex);
      this.store.setAudioState({ isPlaying: true });
      this.timeManager.start();
    });

    howl.on('pause', () => {
      this._log('Segment paused:', segmentIndex);
      this.store.setAudioState({ isPlaying: false });
      this.timeManager.pause();
    });

    howl.on('end', async () => {
      if (this.isTransitioning) {
        this._log('Ignoring end event during transition');
        return;
      }
      
      const nextSegmentIndex = this.currentSegmentIndex + 1;
      if (nextSegmentIndex < this.segmentManager.segments.length) {
        this._log('Auto-transitioning to next segment:', nextSegmentIndex);
        await this._transitionToSegment(nextSegmentIndex, 0, true);
      } else {
        this._log('Audio ended');
        this.store.setAudioState({
          isPlaying: false,
          currentTime: this.store.audioState.duration
        });
        this.timeManager.stop();
      }
    });

    howl.on('loaderror', (id, error) => {
      this._error('Segment load error:', segmentIndex, error);
    });
  }

  _checkAndPreloadNext(currentTime) {
    if (this.preloadStarted || !this.currentHowl?.playing()) {
      return;
    }

    if (this.segmentManager.shouldPreloadNext(currentTime, this.currentSegmentIndex)) {
      this.preloadStarted = true;
      this._log('Starting preload of next segment');
      this.segmentManager.preloadNextSegment(this.currentSegmentIndex)
        .catch(error => this._error('Preload failed:', error))
        .finally(() => {
          this.preloadStarted = false;
        });
    }
  }

  async _transitionToSegment(targetSegmentIndex, offset = 0, autoplay = false) {
    if (this.isTransitioning) {
      this._log('Transition already in progress, waiting...');
      try {
        await this.transitionPromise;
      } catch (error) {
        this._error('Previous transition failed:', error);
      }
    }

    this.isTransitioning = true;
    this.transitionPromise = (async () => {
      try {
        const wasPlaying = autoplay || this.currentHowl?.playing();
        const targetTime = (targetSegmentIndex * this.segmentManager.segmentDuration) + offset;
        
        // Check if the next segment is already preloaded
        let nextHowl = null;
        if (targetSegmentIndex === this.currentSegmentIndex + 1) {
          nextHowl = this.segmentManager.getNextPreloadedSegment(this.currentSegmentIndex);
        }

        if (!nextHowl) {
          // If not preloaded, load it normally
          const segment = await this.segmentManager.loadSegmentsAroundTime(targetTime);
          if (!segment?.howl) {
            throw new Error(`Failed to load segment ${targetSegmentIndex}`);
          }
          nextHowl = segment.howl;
        }

        // Stop current howl after we have the next one ready
        if (this.currentHowl) {
          this.currentHowl.stop();
        }

        this._setupHowlEvents(nextHowl, targetSegmentIndex);
        this.currentHowl = nextHowl;
        this.currentSegmentIndex = targetSegmentIndex;
        this.preloadStarted = false;

        this.currentHowl.seek(offset);
        this.timeManager.setTime(targetTime);
        
        if (wasPlaying) {
          this.currentHowl.play();
        }

        // Start preloading the next segment immediately
        this._checkAndPreloadNext(targetTime);
      } catch (error) {
        this._error('Transition failed:', { targetSegmentIndex, offset }, error);
        throw error;
      } finally {
        this.isTransitioning = false;
      }
    })();

    return this.transitionPromise;
  }

  async play() {
    this._log('Play requested');
    try {
      if (!this.isReady) {
        this._error('Player not ready');
        return;
      }

      if (!this.currentHowl) {
        this._error('No current howl instance');
        return;
      }

      // Unlock audio on iOS
      const unlockiOS = () => {
        const audioContext = Howler.ctx;
        if (audioContext && audioContext.state === 'suspended') {
          audioContext.resume();
        }
        document.body.removeEventListener('touchstart', unlockiOS);
        document.body.removeEventListener('touchend', unlockiOS);
      };
      
      document.body.addEventListener('touchstart', unlockiOS);
      document.body.addEventListener('touchend', unlockiOS);

      // Play the current segment
      await this.currentHowl.play();
    } catch (error) {
      this._error('Play failed:', error);
    }
  }

  pause() {
    this._log('Pause requested');
    if (this.currentHowl) {
      this.currentHowl.pause();
    }
  }

  async stop() {
    this._log('Stop requested');
    if (this.isTransitioning) {
      try {
        await this.transitionPromise;
      } catch (error) {
        this._error('Error waiting for transition during stop:', error);
      }
    }

    this.isTransitioning = true;
    try {
      if (this.currentHowl) {
        this.currentHowl.stop();
      }
      this.segmentManager.unloadAll();
      this.timeManager.stop();
      await this._transitionToSegment(0, 0, false);
    } catch (error) {
      this._error('Error during stop:', error);
    } finally {
      this.isTransitioning = false;
    }
  }

  // Debounced seek method to handle rapid seeking
  seek = debounce(async (targetTime) => {
    this._log('Seek requested:', targetTime);
    try {
      const targetSegmentIndex = Math.floor(targetTime / this.segmentManager.segmentDuration);
      const offset = targetTime % this.segmentManager.segmentDuration;
      await this._transitionToSegment(targetSegmentIndex, offset);
    } catch (error) {
      this._error('Seek failed:', { targetTime }, error);
    }
  }, 200);
}

export default HowlerPlayer;