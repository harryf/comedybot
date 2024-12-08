import React, { useRef, useEffect } from 'react';
import classNames from 'classnames';
import { ShareIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
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

  const handleShare = async (e, start) => {
    e.stopPropagation();
    const url = new URL(window.location.href);
    url.hash = `t=${Math.floor(start)}`;
    try {
      await navigator.clipboard.writeText(url.toString());
      toast.success('Link copied to clipboard!', {
        duration: 2000,
        position: 'bottom-center',
        style: {
          background: '#4B5563',
          color: '#fff',
          borderRadius: '8px',
          padding: '12px 24px',
        },
        iconTheme: {
          primary: '#10B981',
          secondary: '#fff',
        },
      });
    } catch (err) {
      console.error('Failed to copy URL:', err);
      toast.error('Failed to copy link', {
        duration: 2000,
        position: 'bottom-center',
      });
    }
  };

  return (
    <div 
      ref={containerRef}
      className="transcript-container overflow-y-auto h-full"
      style={{ height: 'calc(100vh - 64px)' }}
    >
      <div className="max-w-3xl mx-auto pl-0 pr-4 py-4">
        {transcript.map((line, index) => {
          const isCurrentLine = index === effectiveLineIndex;
          return (
            <div 
              key={index}
              className="flex items-center gap-4 mb-4"
            >
              <div className="w-20 -ml-8 text-right text-xs text-gray-500 flex-shrink-0 opacity-90">
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
                  <div className="flex justify-between items-center gap-2">
                    <p className={classNames(
                      "text-lg font-semibold text-gray-900",
                      { "pl-2": isCurrentLine }
                    )}>
                      {line.text}
                    </p>
                    
                    {/* Share Button */}
                    <button 
                      className="flex-shrink-0 text-gray-400 hover:text-gray-600 p-1"
                      onClick={(e) => handleShare(e, line.start)}
                      title="Share this moment"
                    >
                      <ShareIcon className="w-5 h-5" />
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