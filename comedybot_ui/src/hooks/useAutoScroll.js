import { useEffect, useRef } from 'react';
import Logger from '../utils/logger';

const useAutoScroll = (containerRef, activeIndex, isPlaying, items) => {
  Logger.debug('useAutoScroll called:', {
    containerRefExists: !!containerRef?.current,
    activeIndex,
    isPlaying,
    itemsLength: items?.length
  });

  const userScrolledRef = useRef(false);
  const lastScrollTime = useRef(0);
  const scrollTimeoutRef = useRef(null);

  // Calculate the target scroll position for the active item
  const calculateTargetScroll = () => {
    if (!containerRef.current || activeIndex === null || !items?.length) {
      Logger.debug('Early return conditions:', {
        hasContainer: !!containerRef.current,
        activeIndex,
        itemsLength: items?.length
      });
      return null;
    }

    const container = containerRef.current;
    const containerHeight = container.clientHeight;
    
    // Get all transcript line elements
    const transcriptLines = container.querySelectorAll('.rounded-lg');
    Logger.debug('Found transcript lines:', {
      count: transcriptLines.length,
      activeIndex
    });

    const activeItem = transcriptLines[activeIndex];
    
    if (!activeItem) {
      Logger.debug('No active item found:', {
        activeIndex,
        numLines: transcriptLines.length
      });
      return null;
    }

    // Get the active item's position relative to the container
    const activeItemRect = activeItem.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const activeItemRelativeTop = activeItemRect.top - containerRect.top;

    // Calculate one-third of the container height
    const oneThirdHeight = containerHeight / 3;

    // Calculate how far we need to scroll to get the active item one-third from the top
    const targetScroll = container.scrollTop + (activeItemRelativeTop - oneThirdHeight);

    // Handle edge cases
    const maxScroll = container.scrollHeight - containerHeight;
    if (activeIndex <= 2) {
      Logger.debug('Near start - scrolling to top');
      return 0;
    } else if (activeIndex >= items.length - 3) {
      Logger.debug('Near end - scrolling to bottom');
      return maxScroll;
    }

    Logger.debug('Calculated target scroll:', {
      targetScroll,
      maxScroll,
      containerHeight,
      activeItemRelativeTop,
      oneThirdHeight
    });

    return Math.max(0, Math.min(targetScroll, maxScroll));
  };

  // Scroll to active item
  useEffect(() => {
    if (containerRef.current && activeIndex !== null && items?.length) {
      const targetScroll = calculateTargetScroll();
      if (targetScroll !== null) {
        containerRef.current.scrollTo({
          top: targetScroll,
          behavior: 'smooth'
        });
      }
    }
  }, [activeIndex, items?.length]);

  // Handle user scroll
  useEffect(() => {
    const container = containerRef.current;
    Logger.debug('Setting up scroll listener:', {
      hasContainer: !!container,
      userScrolled: userScrolledRef.current
    });

    if (!container) return;

    const handleScroll = () => {
      const now = Date.now();
      if (now - lastScrollTime.current < 50) return; // Debounce
      
      lastScrollTime.current = now;
      userScrolledRef.current = true;
      Logger.debug('User scrolled:', {
        time: now,
        userScrolled: userScrolledRef.current
      });

      // Reset the user scroll flag after a delay
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      
      scrollTimeoutRef.current = setTimeout(() => {
        userScrolledRef.current = false;
        Logger.debug('Reset user scroll flag');
      }, 1000); // Reset after 1 second of no scrolling
    };

    container.addEventListener('scroll', handleScroll);
    return () => {
      container.removeEventListener('scroll', handleScroll);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Handle auto-scrolling
  useEffect(() => {
    Logger.debug('Auto-scroll effect running:', {
      hasContainer: !!containerRef.current,
      activeIndex,
      isPlaying,
      userScrolled: userScrolledRef.current
    });

    if (!containerRef.current || activeIndex === null || !isPlaying) {
      Logger.debug('Auto-scroll conditions not met:', {
        hasContainer: !!containerRef.current,
        activeIndex,
        isPlaying
      });
      return;
    }

    const targetScroll = calculateTargetScroll();
    Logger.debug('Calculated target scroll:', {
      targetScroll,
      userScrolled: userScrolledRef.current,
      isPlaying
    });

    if (targetScroll === null) return;

    // Only scroll if we're playing or the user hasn't scrolled
    if (isPlaying || !userScrolledRef.current) {
      containerRef.current.scrollTo({
        top: targetScroll,
        behavior: 'smooth'
      });
      Logger.debug('Scrolling to:', targetScroll);
    }
  }, [activeIndex, isPlaying, items]);

  return {
    userScrolled: userScrolledRef.current
  };
};

export default useAutoScroll;
