import { useState, useEffect, useRef } from 'react';

const useTranscriptScroll = (transcript, currentTime) => {
  const containerRef = useRef(null);
  const [currentLineIndex, setCurrentLineIndex] = useState(0);
  const [userScrolling, setUserScrolling] = useState(false);

  // Find current line based on time
  useEffect(() => {
    if (!transcript.length || currentTime === 0) return;

    const newIndex = transcript.findIndex((line, index) => {
      const nextLine = transcript[index + 1];
      return currentTime >= line.start && (!nextLine || currentTime < nextLine.start);
    });

    if (newIndex !== -1) {
      setCurrentLineIndex(newIndex);
    }
  }, [currentTime, transcript]);

  // Handle automatic scrolling
  useEffect(() => {
    if (!containerRef.current || userScrolling || currentTime === 0) return;

    const container = containerRef.current;
    const currentLine = container.children[currentLineIndex];
    
    if (currentLine) {
      const containerHeight = container.clientHeight;
      const lineTop = currentLine.offsetTop;
      const lineHeight = currentLine.clientHeight;

      container.scrollTo({
        top: lineTop - (containerHeight * 0.4) + (lineHeight / 2),
        behavior: 'smooth'
      });
    }
  }, [currentLineIndex, userScrolling, currentTime]);

  // Detect user scrolling
  useEffect(() => {
    if (!containerRef.current) return;

    let scrollTimeout;
    const container = containerRef.current;

    const handleScroll = () => {
      setUserScrolling(true);
      clearTimeout(scrollTimeout);
      
      scrollTimeout = setTimeout(() => {
        setUserScrolling(false);
      }, 1000);
    };

    container.addEventListener('scroll', handleScroll);
    return () => {
      container.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollTimeout);
    };
  }, []);

  const reset = () => {
    if (containerRef.current) {
      // Force immediate scroll reset
      containerRef.current.style.scrollBehavior = 'auto';
      containerRef.current.scrollTop = 0;
      containerRef.current.style.scrollBehavior = '';
    }
    setCurrentLineIndex(0);
    setUserScrolling(false);
  };

  const scrollToLine = (index) => {
    if (index === 0) {
      reset();
    } else {
      setCurrentLineIndex(index);
      setUserScrolling(false);
    }
  };

  return {
    currentLineIndex,
    scrollToLine,
    reset,
    containerRef,
    isUserScrolling: userScrolling
  };
};

export default useTranscriptScroll; 