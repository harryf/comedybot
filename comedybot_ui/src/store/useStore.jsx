import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

const logger = (config) => (set, get, api) => config((args) => {
  console.log('Previous state:', get());
  set(args);
  console.log('Next state:', get());
}, get, api);

const useStore = create(
  logger(
    devtools(
      (set, get) => ({
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
        
        stop: () => {
          const { player, setAudioState, transcriptContainerRef } = get();
          if (player) {
            console.log('Stopping audio');
            player.stop();
            setAudioState({ 
              isPlaying: false, 
              currentTime: 0,
              isStopped: true 
            });
            
            // Scroll transcript to top
            if (transcriptContainerRef?.current) {
              transcriptContainerRef.current.scrollTo({
                top: 0,
                behavior: 'smooth'
              });
            }
          }
        },

        seek: (time) => {
          const { player, setAudioState } = get();
          if (player) {
            console.log('Seeking to:', time);
            player.seek(time);
            player.play();
            setAudioState({ isPlaying: true, currentTime: time });
          }
        },
        
        togglePlayPause: () => {
          const { player, audioState, setAudioState } = get();
          if (player) {
            if (audioState.isPlaying) {
              console.log('Pausing audio');
              player.pause();
              setAudioState({ isPlaying: false });
            } else {
              console.log('Playing audio');
              player.play();
              setAudioState({ isPlaying: true });
            }
          }
        },
        
        // Data loading action
        loadData: async () => {
          console.log('Starting data load');
          try {
            const [metadataResponse, transcriptResponse, soundsResponse] = await Promise.all([
              fetch('/metadata.json'),
              fetch('/transcript_clean.json'),
              fetch('/sounds_clean.json')
            ]);

            const metadata = await metadataResponse.json();
            const transcript = await transcriptResponse.json();
            const sounds = await soundsResponse.json();

            set({
              metadata,
              transcript,
              sounds,
              dataLoaded: true
            }, false, 'loadData');
            
            console.log('Data loaded successfully', {
              hasMetadata: !!metadata,
              transcriptLength: transcript.length,
              hasSounds: !!sounds
            });
          } catch (error) {
            console.error('Error loading data:', error);
            throw error;
          }
        },
        
        // UI Refs setters
        setTranscriptContainerRef: (ref) => set({ transcriptContainerRef: ref }, false, 'setTranscriptContainerRef'),
      })
    )
  )
);

export default useStore;