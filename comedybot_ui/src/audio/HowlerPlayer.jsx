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
    this.segmentManager = new AudioSegmentManager(true); // Force debug on
    this.timeManager = new AudioTimeManager(true); // Force debug on
    this.currentSegmentIndex = -1;
    this.isTransitioning = false;
    this.transitionPromise = null;
    this.preloadStarted = false;
    this.lastTransitionTime = 0;
    this.debug = true; // Force debug on

    // Set up time update callback with enhanced logging
    this.timeManager.setTimeUpdateCallback((time) => {
      const currentSegmentEndTime = (this.currentSegmentIndex + 1) * this.segmentManager.segmentDuration;
      const timeUntilEnd = currentSegmentEndTime - time;
      
      this._log('Time update:', { 
        time,
        currentSegmentIndex: this.currentSegmentIndex,
        timeUntilEnd,
        currentSegmentEndTime,
        isTransitioning: this.isTransitioning,
        howlState: this.currentHowl?.state(),
        howlSeek: this.currentHowl?.seek(),
        preloadStarted: this.preloadStarted,
        preloadThreshold: this.segmentManager.PRELOAD_THRESHOLD,
        shouldPreload: timeUntilEnd <= this.segmentManager.PRELOAD_THRESHOLD && !this.preloadStarted
      });
      
      this.store.setAudioState({ currentTime: time });
      
      // Update segment manager's current time
      this.segmentManager.updateCurrentTime(time);

      // Always check preload first
      this._checkAndPreloadNext(time);
      
      // Then check if we need to transition
      if (time >= currentSegmentEndTime && !this.isTransitioning) {
        const timeSinceLastTransition = time - this.lastTransitionTime;
        this._log('Time-based segment transition triggered:', {
          time,
          currentSegmentEndTime,
          nextSegment: this.currentSegmentIndex + 1,
          timeSinceLastTransition,
          howlState: this.currentHowl?.state(),
          howlSeek: this.currentHowl?.seek()
        });
        
        this._transitionToSegment(this.currentSegmentIndex + 1, 0, true)
          .catch(error => this._error('Time-based transition failed:', error));
      }
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
      const now = new Date();
      const timestamp = `${now.getMinutes()}:${now.getSeconds()}.${now.getMilliseconds()}`;
      Logger.debug(`[HowlerPlayer ${timestamp}]`, ...args);
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
      this._log('Segment playing:', { 
        segmentIndex,
        seek: howl.seek(),
        state: howl.state(),
        duration: howl.duration(),
        isTransitioning: this.isTransitioning,
        timeManagerTime: this.timeManager.currentTime,
        timeSinceLastTransition: this.timeManager.currentTime - this.lastTransitionTime
      });
      if (!this.isTransitioning) {
        this.store.setAudioState({ isPlaying: true });
        this.timeManager.start();
      }
    });

    howl.on('pause', () => {
      this._log('Segment paused:', {
        segmentIndex,
        seek: howl.seek(),
        state: howl.state(),
        isTransitioning: this.isTransitioning,
        timeManagerTime: this.timeManager.currentTime
      });
      this.store.setAudioState({ isPlaying: false });
      this.timeManager.pause();
    });

    howl.on('end', () => {
      this._log('Segment ended (ignoring end event, using time-based transitions):', {
        segmentIndex,
        seek: howl.seek(),
        state: howl.state(),
        isTransitioning: this.isTransitioning,
        timeManagerTime: this.timeManager.currentTime,
        timeSinceLastTransition: this.timeManager.currentTime - this.lastTransitionTime
      });
    });

    howl.on('loaderror', (id, error) => {
      this._error('Segment load error:', {
        segmentIndex,
        error,
        state: howl.state(),
        timeManagerTime: this.timeManager.currentTime
      });
    });
  }

  async _transitionToSegment(targetSegmentIndex, offset = 0, autoplay = false) {
    this._log('Starting transition:', {
      targetSegmentIndex,
      offset,
      autoplay,
      currentSegmentIndex: this.currentSegmentIndex,
      isTransitioning: this.isTransitioning,
      currentTime: this.timeManager.currentTime,
      howlState: this.currentHowl?.state(),
      howlSeek: this.currentHowl?.seek(),
      timeSinceLastTransition: this.timeManager.currentTime - this.lastTransitionTime
    });

    if (this.isTransitioning) {
      this._log('Transition already in progress, waiting...', {
        currentTransition: {
          from: this.currentSegmentIndex,
          to: targetSegmentIndex,
          timeSinceStart: Date.now() - this.lastTransitionTime
        }
      });
      try {
        await this.transitionPromise;
      } catch (error) {
        this._error('Previous transition failed:', error);
      }
    }

    // Start loading the next segment before stopping the current one
    let nextHowl = null;
    try {
      // Try to get the preloaded segment first
      if (targetSegmentIndex === this.currentSegmentIndex + 1) {
        this._log('Attempting to get preloaded segment');
        nextHowl = this.segmentManager.getNextPreloadedSegment(this.currentSegmentIndex);
        this._log('Preloaded segment result:', {
          found: !!nextHowl,
          state: nextHowl?.state(),
          duration: nextHowl?.duration(),
          targetIndex: targetSegmentIndex,
          preloadQueue: [...this.segmentManager.preloadQueue]
        });
      }

      // If no preloaded segment, load manually
      if (!nextHowl) {
        this._log('No preloaded segment, loading manually');
        const segment = await this.segmentManager.loadSegmentsAroundTime(targetSegmentIndex * this.segmentManager.segmentDuration);
        nextHowl = segment?.howl;
        this._log('Manual load result:', {
          success: !!nextHowl,
          state: nextHowl?.state(),
          duration: nextHowl?.duration(),
          targetIndex: targetSegmentIndex,
          preloadQueue: [...this.segmentManager.preloadQueue]
        });
      }

      if (!nextHowl || nextHowl.state() !== 'loaded') {
        throw new Error(`Failed to load segment ${targetSegmentIndex} (state: ${nextHowl?.state()})`);
      }
    } catch (error) {
      this._error('Failed to load next segment:', {
        error,
        targetIndex: targetSegmentIndex,
        currentState: this.currentHowl?.state(),
        timeManagerState: this.timeManager.currentTime,
        preloadQueue: [...this.segmentManager.preloadQueue]
      });
      this.timeManager.pause();
      throw error;
    }

    // Now that we have the next segment ready, start the transition
    this.isTransitioning = true;
    const wasPlaying = autoplay || this.currentHowl?.playing();
    const transitionStartTime = Date.now();
    
    if (wasPlaying) {
      this._log('Pausing time manager during transition');
      this.timeManager.pause();
    }

    this.transitionPromise = (async () => {
      try {
        const targetTime = (targetSegmentIndex * this.segmentManager.segmentDuration) + offset;
        this._log('Transition details:', {
          targetTime,
          targetSegmentIndex,
          offset,
          wasPlaying,
          nextHowlState: nextHowl.state(),
          nextHowlDuration: nextHowl.duration(),
          transitionStartTime,
          elapsedSinceStart: Date.now() - transitionStartTime,
          preloadQueue: [...this.segmentManager.preloadQueue]
        });

        // Stop current howl only after we have the next one ready
        if (this.currentHowl) {
          this._log('Stopping current howl:', {
            currentSegmentIndex: this.currentSegmentIndex,
            currentSeek: this.currentHowl.seek(),
            currentState: this.currentHowl.state(),
            elapsedSinceStart: Date.now() - transitionStartTime,
            preloadQueue: [...this.segmentManager.preloadQueue]
          });
          this.currentHowl.stop();
        }

        // Remove the segment we're transitioning to from the preload queue
        const preloadQueueIndex = this.segmentManager.preloadQueue.indexOf(targetSegmentIndex);
        if (preloadQueueIndex !== -1) {
          this._log('Removing transitioned segment from preload queue:', {
            targetSegmentIndex,
            preloadQueueBefore: [...this.segmentManager.preloadQueue]
          });
          this.segmentManager.preloadQueue.splice(preloadQueueIndex, 1);
          this._log('Updated preload queue:', {
            preloadQueueAfter: [...this.segmentManager.preloadQueue]
          });
        }

        this._setupHowlEvents(nextHowl, targetSegmentIndex);
        this.currentHowl = nextHowl;
        this.currentSegmentIndex = targetSegmentIndex;
        this.preloadStarted = false;
        this.lastTransitionTime = this.timeManager.currentTime;

        // Set the correct time and start playing if needed
        this._log('Setting up new segment:', {
          targetTime,
          offset,
          howlState: nextHowl.state(),
          howlDuration: nextHowl.duration(),
          elapsedSinceStart: Date.now() - transitionStartTime,
          preloadQueue: [...this.segmentManager.preloadQueue]
        });

        this.timeManager.setTime(targetTime);
        this.currentHowl.seek(offset);

        if (wasPlaying) {
          this._log('Resuming playback', {
            targetTime,
            currentSeek: this.currentHowl.seek(),
            elapsedSinceStart: Date.now() - transitionStartTime,
            preloadQueue: [...this.segmentManager.preloadQueue]
          });
          this.isTransitioning = false;  // Clear transitioning before play
          await new Promise(resolve => {
            nextHowl.once('play', () => {
              this._log('Play event received', {
                currentSeek: nextHowl.seek(),
                elapsedSinceStart: Date.now() - transitionStartTime,
                preloadQueue: [...this.segmentManager.preloadQueue]
              });
              resolve();
            });
            nextHowl.play();
          });
          this._log('Playback resumed successfully', {
            finalSeek: nextHowl.seek(),
            elapsedSinceStart: Date.now() - transitionStartTime,
            preloadQueue: [...this.segmentManager.preloadQueue]
          });
        } else {
          this._log('Not resuming playback (wasPlaying: false)');
          this.isTransitioning = false;
        }
        
      } catch (error) {
        this._error('Transition failed:', error);
        this.isTransitioning = false;
        throw error;
      }
    })();

    return this.transitionPromise;
  }

  async play() {
    this._log('Play requested');
    if (this.currentHowl) {
      this.currentHowl.play();
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

  _checkAndPreloadNext(currentTime) {
    const timeUntilEnd = (this.currentSegmentIndex + 1) * this.segmentManager.segmentDuration - currentTime;
    const shouldPreload = timeUntilEnd <= this.segmentManager.PRELOAD_THRESHOLD && !this.preloadStarted;
    
    this._log('Preload check:', {
      currentTime,
      timeUntilEnd,
      threshold: this.segmentManager.PRELOAD_THRESHOLD,
      preloadStarted: this.preloadStarted,
      shouldPreload,
      currentSegmentIndex: this.currentSegmentIndex,
      segmentDuration: this.segmentManager.segmentDuration,
      nextSegmentIndex: this.currentSegmentIndex + 1,
      segmentEndTime: (this.currentSegmentIndex + 1) * this.segmentManager.segmentDuration,
      howlState: this.currentHowl?.state(),
      howlSeek: this.currentHowl?.seek()
    });
    
    if (this.preloadStarted) {
      this._log('Skipping preload - already started:', {
        currentTime,
        timeUntilEnd,
        currentSegmentIndex: this.currentSegmentIndex,
        preloadStarted: this.preloadStarted,
        howlState: this.currentHowl?.state()
      });
      return;
    }
    
    if (shouldPreload) {
      this.preloadStarted = true;
      const nextSegmentIndex = this.currentSegmentIndex + 1;
      if (nextSegmentIndex < this.segmentManager.segments.length) {
        this._log('Starting preload of next segment:', {
          nextSegmentIndex,
          timeUntilEnd,
          currentTime,
          segmentEndTime: (this.currentSegmentIndex + 1) * this.segmentManager.segmentDuration,
          howlState: this.currentHowl?.state()
        });
        this.segmentManager._loadSingleSegment(nextSegmentIndex)
          .then(() => {
            this._log('Preload successful:', {
              nextSegmentIndex,
              currentTime: this.timeManager.currentTime,
              timeUntilEnd: (this.currentSegmentIndex + 1) * this.segmentManager.segmentDuration - this.timeManager.currentTime,
              howlState: this.currentHowl?.state()
            });
          })
          .catch(error => {
            this._error('Preload failed:', {
              error,
              nextSegmentIndex,
              currentTime: this.timeManager.currentTime,
              howlState: this.currentHowl?.state()
            });
            this.preloadStarted = false; // Reset so we can try again
          });
      } else {
        this._log('No next segment to preload - reached end:', {
          currentSegmentIndex: this.currentSegmentIndex,
          totalSegments: this.segmentManager.segments.length,
          howlState: this.currentHowl?.state()
        });
      }
    }
  }
}

export default HowlerPlayer;