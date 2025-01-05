import { Howl } from 'howler';
import { getAssetPath } from '../utils/paths';
import Logger from '../utils/logger';

class AudioSegmentManager {
  constructor(debug = false) {
    this.segments = [];
    this.howlCache = new Map(); // Cache for Howl instances
    this.loadingSegments = new Set(); // Track segments currently being loaded
    this.preloadQueue = []; // Queue for preloaded segments
    this.metadata = null;
    this.segmentDuration = 10; // seconds
    this.isInitialized = false;
    this.totalDuration = 0;
    this.debug = true; // Force debug on
    this.PRELOAD_THRESHOLD = 3; // Increased from 1 to 3 seconds
    this.isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    this.HTML5_POOL_SIZE = 15;
    this.currentTime = 0;

    // Configure Howler globally
    Howler.html5PoolSize = this.HTML5_POOL_SIZE;  
    if (this.isSafari) {
      Howler.autoSuspend = false;  // Prevent auto-suspension on Safari
    }
  }

  _log(...args) {
    if (this.debug) {
      const now = new Date();
      const timestamp = `${now.getMinutes()}:${now.getSeconds()}.${now.getMilliseconds()}`;
      Logger.debug(`[AudioSegmentManager ${timestamp}]`, ...args);
    }
  }

  _error(...args) {
    const now = new Date();
    const timestamp = `${now.getMinutes()}:${now.getSeconds()}.${now.getMilliseconds()}`;
    Logger.error(`[AudioSegmentManager ${timestamp}]`, ...args);
  }

  async initialize(metadata) {
    this._log('Initializing with metadata:', metadata);
    
    this.metadata = metadata;
    this.segments = metadata.segments;

    if (metadata.length_of_set) {
      if (typeof metadata.length_of_set === 'number') {
        // If it's already a number (in seconds), use it directly
        this.totalDuration = metadata.length_of_set;
        this._log('Duration from length_of_set number:', this.totalDuration);
      } else if (typeof metadata.length_of_set === 'string') {
        // Try to parse the old "XXmYYs" format
        const durationMatch = metadata.length_of_set.match(/(\d+)m(\d+)s/);
        if (durationMatch) {
          const [_, minutes, seconds] = durationMatch;
          this.totalDuration = (parseInt(minutes) * 60) + parseInt(seconds);
          this._log('Duration parsed from length_of_set string:', this.totalDuration);
        }
      }
    }

    if (!this.totalDuration) {
      this.totalDuration = this.segments.length * this.segmentDuration;
      this._log('Duration calculated from segments:', this.totalDuration);
    }

    this.isInitialized = true;
    return this.totalDuration;
  }

  updateCurrentTime(time) {
    this.currentTime = time;
  }

  _createHowlForSegment(segmentIndex) {
    const path = getAssetPath(`segment_${segmentIndex.toString().padStart(3, '0')}.m4a`);
    this._log('Creating howl:', { segmentIndex, path });
    
    return new Howl({
      src: [path],
      html5: true,
      preload: true,
      pool: this.HTML5_POOL_SIZE,
      onload: () => {
        this._log('Segment loaded:', { segmentIndex, path });
        if (this.isSafari) {
          const howl = this.howlCache.get(segmentIndex);
          if (howl && Math.abs(howl.duration() - this.segmentDuration) > 0.1) {
            this._log('Correcting duration for Safari:', {
              original: howl.duration(),
              corrected: this.segmentDuration
            });
            howl._duration = this.segmentDuration;
          }
        }
      },
      onplay: () => {
        this._log('Segment started playing:', { segmentIndex, path });
        if (this.isSafari) {
          const howl = this.howlCache.get(segmentIndex);
          const expectedTime = (segmentIndex * this.segmentDuration) % this.segmentDuration;
          if (howl && Math.abs(howl.seek() - expectedTime) > 0.1) {
            this._log('Correcting playback position:', {
              actual: howl.seek(),
              expected: expectedTime
            });
            howl.seek(expectedTime);
          }
        }
      },
      onend: () => {
        this._log('Segment ended:', { segmentIndex, path });
      },
      onloaderror: (id, error) => this._error('Howl load error:', { segmentIndex, error })
    });
  }

  _getSegmentForTime(timeInSeconds) {
    const segmentIndex = Math.floor(timeInSeconds / this.segmentDuration);
    const offset = timeInSeconds % this.segmentDuration;
    return { segmentIndex, offset };
  }

  async _loadSingleSegment(targetIndex) {
    this._log('Loading segment:', {
      targetIndex,
      cached: this.howlCache.has(targetIndex),
      loading: this.loadingSegments.has(targetIndex),
      preloadQueue: [...this.preloadQueue],
      cacheSize: this.howlCache.size
    });

    if (this.howlCache.has(targetIndex)) {
      const cachedHowl = this.howlCache.get(targetIndex);
      const state = cachedHowl.state();
      this._log('Found cached howl:', { targetIndex, state });
      
      if (state === 'loaded') {
        return cachedHowl;
      }
      // If not loaded, remove from cache and reload
      this._log('Cached howl not loaded, reloading');
      this.howlCache.delete(targetIndex);
    }

    if (this.loadingSegments.has(targetIndex)) {
      this._log('Segment already loading:', targetIndex);
      return null;
    }

    const howl = this._createHowlForSegment(targetIndex);
    if (!howl) {
      this._error('Failed to create howl for segment:', targetIndex);
      return null;
    }

    this.loadingSegments.add(targetIndex);
    this.howlCache.set(targetIndex, howl);

    return new Promise((resolve, reject) => {
      let resolved = false;
      
      const onLoad = () => {
        if (resolved) return;
        resolved = true;
        
        this._log('Segment loaded successfully:', {
          targetIndex,
          state: howl.state(),
          duration: howl.duration(),
          cacheSize: this.howlCache.size
        });
        
        this.loadingSegments.delete(targetIndex);

        // Only add to preload queue if it's ahead of current segment
        const currentSegmentIndex = Math.floor(this.currentTime / this.segmentDuration);
        if (targetIndex > currentSegmentIndex && !this.preloadQueue.includes(targetIndex)) {
          // Insert in sorted order
          const insertIndex = this.preloadQueue.findIndex(idx => idx > targetIndex);
          if (insertIndex === -1) {
            this.preloadQueue.push(targetIndex);
          } else {
            this.preloadQueue.splice(insertIndex, 0, targetIndex);
          }
          
          this._log('Added to preload queue:', {
            targetIndex,
            queue: [...this.preloadQueue],
            cacheSize: this.howlCache.size
          });
        }
        
        // For Safari, ensure the duration is correct before resolving
        if (this.isSafari && Math.abs(howl.duration() - this.segmentDuration) > 0.1) {
          this._log('Setting exact duration for Safari:', {
            original: howl.duration(),
            corrected: this.segmentDuration
          });
          howl._duration = this.segmentDuration;
        }

        howl.off('load', onLoad);
        howl.off('loaderror', onError);
        resolve(howl);
      };

      const onError = (id, error) => {
        if (resolved) return;
        resolved = true;
        
        this._error('Failed to load segment:', {
          error,
          state: howl.state(),
          cacheSize: this.howlCache.size
        });
        
        this.loadingSegments.delete(targetIndex);
        this.howlCache.delete(targetIndex);
        
        howl.off('load', onLoad);
        howl.off('loaderror', onError);
        reject(error);
      };

      // Check if already loaded
      const state = howl.state();
      this._log('Initial howl state:', { targetIndex, state });
      
      if (state === 'loaded') {
        onLoad();
      } else {
        howl.on('load', onLoad);
        howl.on('loaderror', onError);
      }
    });
  }

  async loadSegmentsAroundTime(timeInSeconds) {
    if (!this.isInitialized) {
      this._error('AudioSegmentManager not initialized');
      return null;
    }

    const { segmentIndex } = this._getSegmentForTime(timeInSeconds);
    const segmentsToLoad = new Set();
    const segmentsToKeep = new Set();

    // Calculate segments to load and keep
    for (let i = -1; i <= 3; i++) {
      const targetIndex = segmentIndex + i;
      if (targetIndex >= 0 && targetIndex < this.segments.length) {
        segmentsToLoad.add(targetIndex);
        segmentsToKeep.add(targetIndex);
      }
    }

    // Unload segments we don't need, but preserve preloaded segments
    for (const [index, howl] of this.howlCache.entries()) {
      if (!segmentsToKeep.has(index) && !this.preloadQueue.includes(index)) {
        this._log('Unloading non-preloaded segment:', {
          index,
          preloadQueue: [...this.preloadQueue],
          isPreloaded: this.preloadQueue.includes(index)
        });
        howl.unload();
        this.howlCache.delete(index);
      }
    }

    // Load new segments
    const loadPromises = [];
    for (const targetIndex of segmentsToLoad) {
      try {
        const loadPromise = this._loadSingleSegment(targetIndex);
        if (loadPromise) {
          loadPromises.push(loadPromise);
        }
      } catch (error) {
        this._error('Error initiating segment load:', error);
      }
    }

    try {
      await Promise.all(loadPromises);
    } catch (error) {
      this._error('Error loading segments:', error);
    }

    // Return the requested segment
    const howl = this.howlCache.get(segmentIndex);
    if (!howl) {
      this._error('Failed to load requested segment:', segmentIndex);
      return null;
    }

    return {
      howl,
      segmentIndex,
      offset: timeInSeconds % this.segmentDuration
    };
  }

  async preloadNextSegment(currentSegmentIndex) {
    const nextIndex = currentSegmentIndex + 1;
    if (nextIndex >= this.segments.length) {
      this._log('No next segment to preload');
      return null;
    }

    try {
      const howl = await this._loadSingleSegment(nextIndex);
      this._log('Preloaded next segment:', nextIndex);
      return howl;
    } catch (error) {
      this._error('Failed to preload next segment:', error);
      return null;
    }
  }

  getNextPreloadedSegment(currentSegmentIndex) {
    const nextIndex = currentSegmentIndex + 1;
    if (nextIndex >= this.segments.length) return null;

    const howl = this.howlCache.get(nextIndex);
    const state = howl?.state();
    const inPreloadQueue = this.preloadQueue.includes(nextIndex);
    
    this._log('Getting preloaded segment:', {
      nextIndex,
      found: !!howl,
      state,
      inPreloadQueue,
      preloadQueue: [...this.preloadQueue],
      cacheSize: this.howlCache.size
    });
    
    if (howl?.state() === 'loaded') {
      // Remove from preload queue since we're about to use it
      const queueIndex = this.preloadQueue.indexOf(nextIndex);
      if (queueIndex !== -1) {
        this.preloadQueue.splice(queueIndex, 1);
        this._log('Removed segment from preload queue:', {
          nextIndex,
          updatedQueue: [...this.preloadQueue]
        });
      }
      return howl;
    }
    return null;
  }

  shouldPreloadNext(currentTime, currentSegmentIndex) {
    const timeUntilEnd = (currentSegmentIndex + 1) * this.segmentDuration - currentTime;
    const shouldPreload = timeUntilEnd <= this.PRELOAD_THRESHOLD;

    this._log('shouldPreloadNext check:', {
      currentTime,
      currentSegmentIndex,
      timeUntilEnd,
      threshold: this.PRELOAD_THRESHOLD,
      shouldPreload,
      preloadQueue: [...this.preloadQueue],
      nextSegmentIndex: currentSegmentIndex + 1
    });

    // Only preload if we're within threshold and the next segment exists
    return shouldPreload && currentSegmentIndex + 1 < this.segments.length;
  }

  unloadAll() {
    this._log('Unloading all segments');
    for (const [index, howl] of this.howlCache.entries()) {
      howl.unload();
    }
    this.howlCache.clear();
    this.loadingSegments.clear();
    this.preloadQueue = [];
  }

  getSegmentAtIndex(segmentIndex) {
    return this.howlCache.get(segmentIndex);
  }
}

export default AudioSegmentManager;
