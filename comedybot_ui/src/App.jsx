  import React, { useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import TranscriptViewer from './components/TranscriptViewer';
import AudioControls from './components/AudioControls';
import useStore from './store/useStore.js';
import HowlerPlayer from './audio/HowlerPlayer';
import { useHashParams } from './hooks/useHashParams';
import Logger from './utils/logger';

function App() {
  const { 
    metadata, 
    transcript, 
    sounds, 
    dataLoaded,
    loadData,
    player,
    setAudioState,
    play,
    seek,
    togglePlayPause
  } = useStore();

  const { time: hashTime, autoplay } = useHashParams();

  // Load data on mount
  useEffect(() => {
    loadData();
  }, [loadData]);

  // Initialize audio player
  useEffect(() => {
    async function initPlayer() {
      if (dataLoaded && !player) {
        Logger.info('Initializing audio player');
        const howlerPlayer = new HowlerPlayer();
        await howlerPlayer.initialize();
        useStore.getState().setPlayer(howlerPlayer);
      }
    }
    initPlayer();
  }, [dataLoaded, player]);

  // Handle URL hash parameters - after player is ready
  useEffect(() => {
    if (hashTime !== undefined && player?.isReady) {
      Logger.debug('Setting initial position and play state:', { hashTime, autoplay });
      
      // First seek to position
      player.seek(hashTime);
      
      // Update state with new position
      setAudioState({ 
        currentTime: hashTime,
        isPlaying: autoplay,
        isStopped: false
      });

      // Start playing if autoplay is true
      if (autoplay) {
        Logger.debug('Starting playback');
        player.play();
      }
    }
  }, [hashTime, autoplay, player?.isReady, setAudioState]);

  // Debug data loading state
  Logger.debug('App render:', {
    dataLoaded,
    hasMetadata: !!metadata,
    hasTranscript: !!transcript.length,
    hasSounds: !!sounds?.reactions
  });

  if (!dataLoaded || !metadata || !transcript.length || !sounds?.reactions) {
    return (
      <div className="h-screen flex items-center justify-center bg-white">
        <div className="text-gray-600">Loading data...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      <Toaster />
      <Header metadata={metadata} />
      <div className="flex-1 flex flex-col mt-4">
        <TranscriptViewer
          transcript={transcript}
        />
        <div data-debug="audio-controls-wrapper">
          <AudioControls
            metadata={metadata}
            sounds={sounds}
          />
        </div>
      </div>
    </div>
  );
}

export default App;