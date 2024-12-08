import React, { useRef, useEffect } from 'react';
import classNames from 'classnames';
import useStore from '../store/useStore.jsx';
import useAutoScroll from '../hooks/useAutoScroll';

const formatTimestamp = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const TranscriptViewer = ({ transcript }) => {
  const containerRef = useRef(null);
  const currentLineRef = useRef(null);
  const { audioState, seek, setTranscriptContainerRef } = useStore();
  const { currentTime, isPlaying } = audioState;

  // Store ref in global state
  useEffect(() => {
    setTranscriptContainerRef(containerRef);
  }, [setTranscriptContainerRef]);

  // Find the current line index based on currentTime
  const currentLineIndex = transcript.findIndex((line, index) => {
    const nextLine = transcript[index + 1];
    return (
      // Exact match within a line's range
      (currentTime >= line.start && (nextLine ? currentTime < nextLine.start : true)) ||
      // Or this is the first line after the current time
      (index > 0 && transcript[index - 1].start <= currentTime && line.start > currentTime)
    );
  });

  // If no line found and we have a valid time, use the next line
  const effectiveLineIndex = currentLineIndex === -1 && currentTime > 0
    ? transcript.findIndex(line => line.start > currentTime)
    : currentLineIndex;

  // Use auto-scroll hook with the effective line index
  useAutoScroll(containerRef, effectiveLineIndex, isPlaying, transcript);

  const handleLineClick = (start) => {
    seek(start);
  };

  return (
    <div 
      ref={containerRef}
      className="transcript-container overflow-y-auto h-full"
      style={{ height: 'calc(100vh - 64px)' }}
    >
      <div className="max-w-3xl mx-auto px-4 py-4">
        {transcript.map((line, index) => {
          const isCurrentLine = index === effectiveLineIndex;
          return (
            <div 
              key={index}
              className="flex items-center gap-2 mb-4"
            >
              <div className="w-12 text-right text-sm text-gray-500 flex-shrink-0">
                {formatTimestamp(line.start)}
              </div>

              {/* Transcript Line */}
              <div
                key={line.start}
                className="flex-1 relative"
                onClick={() => handleLineClick(line.start)}
                ref={isCurrentLine ? currentLineRef : null}
              >
                {/* Background + Accent Container */}
                <div className={`
                  absolute inset-0 rounded-lg
                  ${isCurrentLine ? 'bg-blue-100' : 'bg-gray-50'}
                `}>
                  {isCurrentLine && (
                    <div className="absolute left-0 top-0 bottom-0 w-2 bg-gradient-to-r from-blue-600 to-blue-400 rounded-l-lg" />
                  )}
                </div>

                {/* Content */}
                <div className="relative p-4">
                  <div className="flex justify-between items-start gap-2">
                    <p className={classNames(
                      "text-gray-900",
                      { "pl-2": isCurrentLine }
                    )}>
                      {line.text}
                    </p>
                    
                    {/* Edit Button */}
                    <button 
                      className="flex-shrink-0 text-gray-400 hover:text-gray-600"
                      onClick={(e) => {
                        e.stopPropagation();
                        // Edit functionality will be added later
                      }}
                    >
                      <svg 
                        width="16" 
                        height="16" 
                        viewBox="0 0 16 16" 
                        fill="none" 
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path 
                          d="M11.5 2.5L13.5 4.5M12.5 1.5L8 6L7 9L10 8L14.5 3.5C14.5 3.5 14.5 2.5 13.5 1.5C12.5 0.5 11.5 1.5 11.5 1.5Z" 
                          stroke="currentColor" 
                          strokeLinecap="round" 
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TranscriptViewer;