import { Howl } from 'howler';
import { getAssetPath } from '../utils/paths';

class AudioSegmentManager {
  constructor(debug = false) {
    this.segments = [];
    this.howlCache = new Map(); // Cache for Howl instances
    this.loadingSegments = new Set(); // Track segments currently being loaded
    this.metadata = null;
    this.segmentDuration = 10; // seconds
    this.isInitialized = false;
    this.totalDuration = 0;
    this.debug = debug;
  }

  _log(...args) {
    if (this.debug) {
      console.log('[AudioSegmentManager]', ...args);
    }
  }

  _error(...args) {
    console.error('[AudioSegmentManager]', ...args);
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

  _createHowlForSegment(segmentIndex) {
    if (!this.segments[segmentIndex]) {
      this._error('Invalid segment index:', segmentIndex);
      return null;
    }

    const segmentPath = getAssetPath(this.segments[segmentIndex]);
    this._log('Creating Howl instance for segment:', { segmentIndex, path: segmentPath });
    
    return new Howl({
      src: [segmentPath],
      html5: true,
      preload: true,
      format: ['opus'],
      onload: () => this._log('Segment loaded:', { segmentIndex, path: segmentPath }),
      onloaderror: (id, error) => this._error('Segment load error:', { segmentIndex, path: segmentPath, error })
    });
  }

  _getSegmentForTime(timeInSeconds) {
    const segmentIndex = Math.floor(timeInSeconds / this.segmentDuration);
    const offset = timeInSeconds % this.segmentDuration;
    return { segmentIndex, offset };
  }

  async _loadSingleSegment(targetIndex) {
    if (this.howlCache.has(targetIndex)) {
      return this.howlCache.get(targetIndex);
    }

    if (this.loadingSegments.has(targetIndex)) {
      this._log('Segment already loading:', targetIndex);
      return null;
    }

    const howl = this._createHowlForSegment(targetIndex);
    if (!howl) return null;

    this.loadingSegments.add(targetIndex);
    this.howlCache.set(targetIndex, howl);

    return new Promise((resolve, reject) => {
      howl.once('load', () => {
        this._log('Segment loaded successfully:', targetIndex);
        this.loadingSegments.delete(targetIndex);
        resolve(howl);
      });

      howl.once('loaderror', (id, error) => {
        this._error('Failed to load segment:', targetIndex, error);
        this.howlCache.delete(targetIndex);
        this.loadingSegments.delete(targetIndex);
        reject(new Error(`Failed to load segment ${targetIndex}: ${error}`));
      });
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

    // Unload segments we don't need
    for (const [index, howl] of this.howlCache.entries()) {
      if (!segmentsToKeep.has(index)) {
        this._log('Unloading segment:', index);
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
      return Promise.resolve();
    }

    try {
      const howl = await this._loadSingleSegment(nextIndex);
      this._log('Preloaded next segment:', nextIndex);
      return howl;
    } catch (error) {
      this._error('Failed to preload next segment:', error);
      return Promise.reject(error);
    }
  }

  getSegmentAtIndex(segmentIndex) {
    return this.howlCache.get(segmentIndex);
  }

  unloadAll() {
    this._log('Unloading all segments');
    for (const [index, howl] of this.howlCache.entries()) {
      howl.unload();
    }
    this.howlCache.clear();
    this.loadingSegments.clear();
  }
}

export default AudioSegmentManager;
