import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import App from '../App';
import useStore from '../store/useStore';

// Mock data
const mockData = {
  metadata: {
    title: "Test Show",
    performer: "Test Performer",
    venue: "Test Venue",
    laughs_per_minute: 4.5,
    duration: 771,
  },
  transcript: [
    { start: 0, end: 5, text: "Test line 1" },
    { start: 5, end: 10, text: "Test line 2" },
  ],
  reactions: [
    { time: 2, score: 3 },
    { time: 7, score: 4 },
  ]
};

// Mock Howler
const mockPlayer = {
  play: jest.fn().mockImplementation(() => {
    const { setAudioState } = useStore.getState();
    setAudioState({ isPlaying: true });
  }),
  pause: jest.fn().mockImplementation(() => {
    const { setAudioState } = useStore.getState();
    setAudioState({ isPlaying: false });
  }),
  stop: jest.fn(),
  seek: jest.fn().mockImplementation((time) => {
    const { setAudioState } = useStore.getState();
    setAudioState({ currentTime: time });
  }),
  playing: jest.fn(),
  duration: jest.fn().mockReturnValue(771), // 12:51 in seconds
  unload: jest.fn(),
};

jest.mock('howler', () => ({
  Howl: jest.fn().mockImplementation(() => mockPlayer)
}));

// Mock fetch calls for data
global.fetch = jest.fn((url) => {
  let responseData;
  
  switch(url) {
    case '/metadata.json':
      responseData = mockData.metadata;
      break;
    case '/transcript_clean.json':
      responseData = mockData.transcript;
      break;
    case '/sounds_clean.json':
      responseData = { reactions: mockData.reactions };
      break;
    default:
      throw new Error(`Unhandled fetch url: ${url}`);
  }
  
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve(responseData)
  });
});

// Mock scrollTo method for tests
window.scrollTo = jest.fn();
Element.prototype.scrollTo = jest.fn();

// Create a wrapper to reset store between tests
const createWrapper = () => {
  const store = useStore.getState();
  
  beforeEach(() => {
    // Reset store to initial state
    useStore.setState({
      audioState: {
        isPlaying: false,
        currentTime: 0,
        duration: 0,
        isLoading: false,
      },
      dataLoaded: false,
      player: null,
      metadata: {
        title: 'Test Show',
        performer: 'Test Performer',
        date: new Date('2023-01-01'),
        venue: 'Test Venue',
      },
      transcript: [
        { text: 'Test line 1', timestamp: 0 },
        { text: 'Test line 2', timestamp: 5 },
      ],
      reactions: [
        { emoji: '😆', timestamp: 2 },
        { emoji: '😂', timestamp: 7 },
      ],
    });

    // Mock loadData function
    useStore.setState({
      loadData: jest.fn().mockImplementation(async () => {
        useStore.getState().setAudioState({ 
          dataLoaded: true,
          player: mockPlayer,
        });
        return Promise.resolve();
      }),
    });
  });

  return store;
};

describe('ComedyBot UI', () => {
  // Initialize store wrapper
  const store = createWrapper();

  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    render(<App />);
    expect(screen.getByText('Loading data...')).toBeInTheDocument();
  });

  test('loads and displays data', async () => {
    render(<App />);
    
    // Wait for data to be loaded in the store
    await waitFor(() => {
      const state = useStore.getState();
      console.log('Current store state:', state);
      return state.dataLoaded && state.metadata && state.transcript.length > 0;
    }, { timeout: 5000 });

    // Debug log the final state
    console.log('Final store state:', useStore.getState());

    // Now check if loading is gone
    expect(screen.queryByText('Loading data...')).not.toBeInTheDocument();
    expect(screen.getByText('Test Show')).toBeInTheDocument();
    expect(screen.getByText('Test Performer')).toBeInTheDocument();
  });

  test('initializes audio player after data load', async () => {
    render(<App />);
    
    await waitFor(() => {
      const state = useStore.getState();
      console.log('Audio player state:', state.player);
      return state.player !== null;
    }, { timeout: 5000 });
  });

  test('toggles play/pause', async () => {
    render(<App />);
    
    // Wait for data to be loaded
    await waitFor(() => {
      const state = useStore.getState();
      return state.dataLoaded;
    });

    // Find play button by data-testid
    const playButton = screen.getByTestId('play-button');
    
    // Initial state should be paused
    expect(useStore.getState().audioState.isPlaying).toBe(false);
    
    // Click play
    await act(async () => {
      fireEvent.click(playButton);
    });
    
    // Should now be playing
    expect(mockPlayer.play).toHaveBeenCalled();
    expect(useStore.getState().audioState.isPlaying).toBe(true);
  });

  test('clicking transcript line seeks and starts playing', async () => {
    render(<App />);
    
    // Wait for data to be loaded and player to be initialized
    await waitFor(() => {
      const state = useStore.getState();
      return state.dataLoaded && state.player !== null;
    }, { timeout: 5000 });

    // Click the second line which starts at 5 seconds
    const secondLine = screen.getByText('Test line 2');
    
    await act(async () => {
      fireEvent.click(secondLine);
    });

    // Verify seeking to correct position
    expect(mockPlayer.seek).toHaveBeenCalledWith(5);
    
    // Verify playback starts
    expect(mockPlayer.play).toHaveBeenCalled();
    await waitFor(() => {
      console.log('After play:', useStore.getState().audioState);
      expect(useStore.getState().audioState.isPlaying).toBe(true);
    });
    
    // Verify current time is updated
    await waitFor(() => {
      console.log('Current time:', useStore.getState().audioState.currentTime);
      expect(useStore.getState().audioState.currentTime).toBe(5);
    });
  });

  test('header remains fixed at top of screen', async () => {
    render(<App />);
    
    await waitFor(() => {
      const state = useStore.getState();
      return state.dataLoaded;
    });

    // Use a more specific selector to find the header
    const header = document.querySelector('[data-testid="header"]');
    expect(header).not.toBeNull(); // Verify header is rendered
    
    // Check for 'fixed' class directly instead of computed style
    expect(header.classList.contains('fixed')).toBe(true);
  });

  test('transcript container has correct spacing below header', async () => {
    render(<App />);
    
    await waitFor(() => {
      const state = useStore.getState();
      return state.dataLoaded;
    });

    // Find the content container div that comes after the header
    const header = document.querySelector('[data-testid="header"]');
    const contentContainer = header.nextElementSibling;
    expect(contentContainer).not.toBeNull();
    
    // Get all classes on the container
    const classList = Array.from(contentContainer.classList);
    console.log('Container classes:', classList);
    
    // Check for mt-4 class
    expect(classList).toContain('mt-4');
  });

  test('transcript lines have consistent spacing', async () => {
    render(<App />);
    
    await waitFor(() => {
      const state = useStore.getState();
      return state.dataLoaded;
    });

    // Find all transcript lines
    const transcriptLines = document.querySelectorAll('.flex.items-center.gap-2.mb-4');
    expect(transcriptLines.length).toBeGreaterThan(0);

    // Check that all lines have the correct classes
    transcriptLines.forEach((line) => {
      const classList = Array.from(line.classList);
      expect(classList).toContain('flex');
      expect(classList).toContain('items-center');
      expect(classList).toContain('gap-2');
      expect(classList).toContain('mb-4');
    });
  });
});