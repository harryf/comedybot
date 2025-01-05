import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { getAssetPath } from '../utils/paths';
import Logger from '../utils/logger';

const initialState = {
  // Audio state
  audioState: {
    currentTime: 0,
    duration: 0,
    isPlaying: false,
    isStopped: true,
    isLoaded: false,
  },
  
  // Data state
  transcript: [],
  metadata: null,
  sounds: null,
  dataLoaded: false,
  player: null,
  transcriptContainerRef: null,
};

const useStore = create(
  devtools(
    (set, get) => ({
      ...initialState,

      // Actions for audio
      setAudioState: (newState) => set((state) => {
        const updatedState = { audioState: { ...state.audioState, ...newState }};
        Logger.debug('Audio State Update:', {
          previous: state.audioState,
          changes: newState,
          new: updatedState.audioState
        });
        return updatedState;
      }, false, 'setAudioState'),
      
      // Actions for data loading
      setTranscript: (transcript) => set({ transcript }, false, 'setTranscript'),
      setMetadata: (metadata) => set({ metadata }, false, 'setMetadata'),
      setSounds: (sounds) => set({ sounds }, false, 'setSounds'),
      setDataLoaded: (loaded) => set({ dataLoaded: loaded }, false, 'setDataLoaded'),
      
      // Audio control actions
      setPlayer: (player) => set({ player }, false, 'setPlayer'),

      // UI Refs setters
      setTranscriptContainerRef: (ref) => set({ transcriptContainerRef: ref }, false, 'setTranscriptContainerRef'),
      
      seek: (time) => {
        const { player } = get();
        if (player) {
          Logger.debug('Seeking to:', time);
          player.seek(time);
        }
      },
      
      togglePlayPause: () => {
        const { player, audioState } = get();
        if (player) {
          if (audioState.isPlaying) {
            Logger.debug('Pausing audio');
            player.pause();
            set({ audioState: { ...audioState, isPlaying: false } }, false, 'togglePlayPause');
          } else {
            Logger.debug('Playing audio');
            player.play();
            set({ audioState: { ...audioState, isPlaying: true } }, false, 'togglePlayPause');
          }
        }
      },

      play: () => {
        const { player, audioState } = get();
        if (player) {
          Logger.debug('Playing audio');
          player.play();
          set({ audioState: { ...audioState, isPlaying: true } }, false, 'play');
        }
      },
      
      stop: () => {
        const { player, audioState } = get();
        if (player) {
          Logger.debug('Stopping audio');
          player.stop();
          set({ audioState: { ...audioState, isPlaying: false, isStopped: true } }, false, 'stop');
        }
      },
      
      // Data loading action
      loadData: async () => {
        Logger.info('Starting data load');
        try {
          const [metadataResponse, transcriptResponse, soundsResponse] = await Promise.all([
            fetch(getAssetPath('metadata.json')),
            fetch(getAssetPath('transcript_clean.json')),
            fetch(getAssetPath('sounds_clean.json'))
          ]);

          // Check if any responses failed
          if (!metadataResponse.ok) {
            throw new Error(`Failed to load metadata: ${metadataResponse.status} ${metadataResponse.statusText}`);
          }
          if (!transcriptResponse.ok) {
            throw new Error(`Failed to load transcript: ${transcriptResponse.status} ${transcriptResponse.statusText}`);
          }
          if (!soundsResponse.ok) {
            throw new Error(`Failed to load sounds: ${soundsResponse.status} ${soundsResponse.statusText}`);
          }

          const metadata = await metadataResponse.json();
          Logger.debug('Metadata contents:', {
            metadata,
            lengthOfSet: metadata.length_of_set,
            typeOfLengthOfSet: typeof metadata.length_of_set
          });
          const transcript = await transcriptResponse.json();
          const sounds = await soundsResponse.json();

          Logger.debug('Loaded data:', {
            metadata,
            transcriptLength: transcript.length,
            soundsLength: sounds.length
          });

          set({
            metadata,
            transcript,
            sounds,
            dataLoaded: true
          }, false, 'loadData');
        } catch (error) {
          Logger.error('Error loading data:', error);
          Logger.error('Current window.PLAYER_CONFIG:', window.PLAYER_CONFIG);
          throw error;
        }
      }
    })
  )
);

export default useStore;