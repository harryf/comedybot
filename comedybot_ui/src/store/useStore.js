import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { getAssetPath } from '../utils/paths';

const logger = (config) => (set, get, api) => config((args) => {
  console.log('Previous state:', get());
  set(args);
  console.log('Next state:', get());
}, get, api);

const useStore = create(
  logger(
    devtools((set, get) => ({
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
      
      // Audio player instance
      player: null,

      // UI Refs
      transcriptContainerRef: null,
      
      // Actions for audio
      setAudioState: (newState) => set((state) => {
        const updatedState = { audioState: { ...state.audioState, ...newState }};
        console.log('Audio State Update:', {
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
          console.log('Seeking to:', time);
          player.seek(time);
        }
      },
      
      togglePlayPause: () => {
        const { player, audioState } = get();
        if (player) {
          if (audioState.isPlaying) {
            console.log('Pausing audio');
            player.pause();
            set({ audioState: { ...audioState, isPlaying: false } }, false, 'togglePlayPause');
          } else {
            console.log('Playing audio');
            player.play();
            set({ audioState: { ...audioState, isPlaying: true } }, false, 'togglePlayPause');
          }
        }
      },
      
      stop: () => {
        const { player, audioState } = get();
        if (player) {
          console.log('Stopping audio');
          player.stop();
          set({ audioState: { ...audioState, isPlaying: false, isStopped: true } }, false, 'stop');
        }
      },
      
      // Data loading action
      loadData: async () => {
        console.log('Starting data load');
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
          console.log('Metadata contents:', {
            metadata,
            lengthOfSet: metadata.length_of_set,
            typeOfLengthOfSet: typeof metadata.length_of_set
          });
          const transcript = await transcriptResponse.json();
          const sounds = await soundsResponse.json();

          console.log('Loaded data:', {
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
          console.error('Error loading data:', error);
          console.error('Current window.PLAYER_CONFIG:', window.PLAYER_CONFIG);
          throw error;
        }
      }
    }))
  )
);

export default useStore;