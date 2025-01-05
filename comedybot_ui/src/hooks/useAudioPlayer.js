import { useState, useEffect, useCallback, useRef } from 'react';
import { Howl } from 'howler';

const useAudioPlayer = (audioUrl) => {
  const [sound, setSound] = useState(null);
  const [loading, setLoading] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioReady, setAudioReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationFrameRef = useRef(null);
  const soundRef = useRef(null);

  // Initialize Howler sound object only when needed
  const initializeAudio = useCallback(() => {
    if (soundRef.current) return soundRef.current;

    const newSound = new Howl({
      src: [audioUrl],
      html5: true,
      preload: true,
      onload: () => {
        setDuration(newSound.duration());
        setAudioReady(true);
        setLoading(false);
      },
      onloaderror: (id, error) => {
        console.error('Error loading audio:', error);
        setLoading(false);
      },
      onplay: () => {
        setIsPlaying(true);
        // Start the time update loop
        const updateTimeLoop = () => {
          if (soundRef.current && soundRef.current.playing()) {
            setCurrentTime(soundRef.current.seek());
            animationFrameRef.current = requestAnimationFrame(updateTimeLoop);
          }
        };
        updateTimeLoop();
      },
      onpause: () => {
        setIsPlaying(false);
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      },
      onstop: () => {
        setIsPlaying(false);
        setCurrentTime(0);
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      },
      onend: () => {
        setIsPlaying(false);
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      },
    });

    soundRef.current = newSound;
    setSound(newSound);
    return newSound;
  }, [audioUrl]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (soundRef.current) {
        soundRef.current.unload();
      }
    };
  }, []);

  const play = useCallback(() => {
    setLoading(true);
    const currentSound = initializeAudio();
    if (currentSound && !currentSound.playing()) {
      currentSound.play();
    }
  }, [initializeAudio]);

  const pause = useCallback(() => {
    if (soundRef.current && soundRef.current.playing()) {
      soundRef.current.pause();
      setIsPlaying(false);
    }
  }, []);

  const stop = useCallback(() => {
    if (soundRef.current) {
      soundRef.current.stop();
      soundRef.current.seek(0);
      setCurrentTime(0);
      setIsPlaying(false);
    }
  }, []);

  const seek = useCallback((time) => {
    const currentSound = soundRef.current || initializeAudio();
    if (currentSound) {
      currentSound.seek(time);
      setCurrentTime(time);
    }
  }, [initializeAudio]);

  return {
    play,
    pause,
    stop,
    seek,
    duration,
    currentTime,
    loading,
    isPlaying,
  };
};

export default useAudioPlayer; 