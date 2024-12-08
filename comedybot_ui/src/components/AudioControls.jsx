import React, { useMemo } from 'react';
import useStore from '../store/useStore.jsx';
import { PlayIcon, PauseIcon, StopIcon } from '@heroicons/react/24/solid';

const SCORE_TO_COLOR = {
  0: '#FFFFFF',
  1: '#FFEDD5',
  2: '#FED7AA',
  3: '#FDBA74',
  4: '#FB923C',
  5: '#F97316'
};

const DECAY_DURATION = 3; // seconds of decay on each side
const SEGMENTS = 200; // number of segments to divide the timeline into

const EMOJIS = {
  0: '🙂',
  1: '😀',
  2: '😄',
  3: '😆',
  4: '😂',
  5: '🤣'
};

const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const AudioControls = () => {
  const { 
    audioState, 
    togglePlayPause,
    seek,
    sounds,
    metadata,
    stop
  } = useStore();
  
  const { currentTime, duration, isPlaying } = audioState;

  // Calculate heatmap data with decay
  const heatmapData = useMemo(() => {
    if (!sounds?.reactions || !duration) return [];
    
    // Initialize segments array
    const segments = Array(SEGMENTS).fill(0);
    const segmentDuration = duration / SEGMENTS;
    
    // Process each reaction and apply decay
    sounds.reactions.forEach(reaction => {
      if (!reaction.reaction_score) return;
      
      const centerSegment = Math.floor(reaction.start / segmentDuration);
      const decaySegments = Math.floor(DECAY_DURATION / segmentDuration);
      
      // Apply score with decay to surrounding segments
      for (let i = -decaySegments; i <= decaySegments; i++) {
        const segment = centerSegment + i;
        if (segment >= 0 && segment < SEGMENTS) {
          const distance = Math.abs(i * segmentDuration);
          const decayFactor = Math.max(0, 1 - distance / DECAY_DURATION);
          const score = reaction.reaction_score * decayFactor;
          segments[segment] = Math.max(segments[segment], score);
        }
      }
    });
    
    return segments;
  }, [sounds, duration]);

  // Convert score to color with opacity
  const getColorForScore = (score) => {
    const baseScore = Math.min(5, Math.ceil(score));
    const opacity = score / baseScore;
    return `${SCORE_TO_COLOR[baseScore]}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`;
  };

  // Get current score based on time
  const getCurrentScore = useMemo(() => {
    if (!sounds?.reactions) return 0;
    
    const currentReaction = sounds.reactions.find((reaction, index) => {
      const nextReaction = sounds.reactions[index + 1];
      return currentTime >= reaction.start && 
             (!nextReaction || currentTime < nextReaction.start);
    });
    
    return currentReaction?.reaction_score || 0;
  }, [sounds, currentTime]);

  const handleSeek = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const newTime = percentage * duration;
    seek(newTime);
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-blue-600 text-white py-6 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Controls and Timeline */}
        <div className="flex items-center gap-4">
          {/* Play/Pause button */}
          <button
            onClick={togglePlayPause}
            className="p-1.5 rounded-full bg-white hover:bg-gray-100 transition-colors flex-shrink-0"
            data-testid="play-button"
          >
            {isPlaying ? (
              <PauseIcon className="w-6 h-6 text-gray-700" />
            ) : (
              <PlayIcon className="w-6 h-6 text-gray-700" />
            )}
          </button>

          <div className="flex-grow">
            {/* Timeline wrapper - ensures vertical centering */}
            <div className="relative">
              {/* Timeline */}
              <div 
                className="relative h-8 bg-gray-200 rounded-lg cursor-pointer overflow-hidden"
                onClick={handleSeek}
              >
                {/* Heatmap visualization */}
                {heatmapData.map((score, index) => {
                  if (score === 0) return null;
                  const position = (index / SEGMENTS) * 100;
                  const width = (1 / SEGMENTS) * 100;
                  return (
                    <div
                      key={index}
                      className="absolute top-0 bottom-0"
                      style={{
                        left: `${position}%`,
                        width: `${width}%`,
                        backgroundColor: getColorForScore(score),
                      }}
                    />
                  );
                })}

                {/* Progress bar */}
                <div 
                  className="absolute top-0 left-0 bottom-0 bg-white/20"
                  style={{ width: `${(currentTime / duration) * 100}%` }}
                />
                
                {/* Progress indicator */}
                <div 
                  className="absolute top-1/2 text-xl"
                  style={{ 
                    left: `${(currentTime / duration) * 100}%`,
                    transform: 'translate(-50%, -50%)',
                    transition: 'left 100ms linear',
                    willChange: 'left'
                  }}
                >
                  <div
                    key={getCurrentScore}
                    style={{
                      transition: 'opacity 150ms ease-out',
                      opacity: 1
                    }}
                  >
                    {EMOJIS[getCurrentScore]}
                  </div>
                </div>
              </div>

              {/* Time indicator - absolutely positioned below the timeline */}
              <div className="absolute -bottom-5 left-0 right-0 text-xs text-center opacity-90">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>
          </div>

          {/* Stop button */}
          <button
            onClick={stop}
            className="p-1.5 rounded-full bg-white hover:bg-gray-100 transition-colors flex-shrink-0"
            data-testid="stop-button"
          >
            <StopIcon className="w-6 h-6 text-gray-700" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default AudioControls;