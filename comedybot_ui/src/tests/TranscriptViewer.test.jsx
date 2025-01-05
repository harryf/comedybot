import React from 'react';
import { render, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import TranscriptViewer from '../components/TranscriptViewer';
import useStore from '../store/useStore';

// Mock the store
jest.mock('../store/useStore');

// Mock date functions
const mockDateNow = jest.spyOn(Date, 'now');
mockDateNow.mockImplementation(() => 0);

// Mock scroll behavior
const mockScrollTo = jest.fn();

const mockTranscript = [
  { text: 'Line 1', start: 0, end: 2 },
  { text: 'Line 2', start: 2, end: 4 },
  { text: 'Line 3', start: 4, end: 6 },
  { text: 'Line 4', start: 6, end: 8 },
  { text: 'Line 5', start: 8, end: 10 }
];

describe('TranscriptViewer', () => {
  let mockStore;
  let mockContainer;
  let containerRef;
  let currentLineRef;

  beforeEach(() => {
    jest.useFakeTimers();
    mockDateNow.mockImplementation(() => 0);
    mockScrollTo.mockClear();

    // Create mock element factory
    const createMockElement = () => {
      const classes = new Set(['flex-1', 'p-4', 'rounded-lg', 'cursor-pointer', 'transition-colors', 'relative', 'hover:bg-gray-100']);
      return {
        classList: { contains: (className) => className === 'transcript-container' },
        clientHeight: 600,
        scrollHeight: 1000,
        scrollTop: 0,
        getBoundingClientRect: () => ({
          top: 0,
          bottom: 600,
          height: 600
        }),
        querySelectorAll: () => 
          Array.from({ length: 5 }, (_, i) => {
            const lineClasses = new Set(['flex-1', 'p-4', 'rounded-lg', 'cursor-pointer', 'transition-colors', 'relative', 'hover:bg-gray-100']);
            if (i === 1) {
              lineClasses.add('bg-blue-100');
            } else {
              lineClasses.add('bg-gray-50');
            }
            return {
              getBoundingClientRect: () => ({
                top: i * 50,
                bottom: (i + 1) * 50,
                height: 50
              }),
              closest: (selector) => {
                if (selector === '.rounded-lg') {
                  return {
                    className: Array.from(lineClasses).join(' '),
                    classList: {
                      contains: (className) => lineClasses.has(className),
                      add: (className) => lineClasses.add(className),
                      remove: (className) => lineClasses.delete(className)
                    }
                  };
                }
                return null;
              }
            };
          }),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        scrollTo: mockScrollTo
      };
    };

    // Create mock container element
    mockContainer = createMockElement();

    // Create refs
    containerRef = { current: mockContainer };
    currentLineRef = { current: mockContainer.querySelectorAll()[0] };

    // Mock useRef
    jest.spyOn(React, 'useRef')
      .mockImplementationOnce(() => containerRef)
      .mockImplementationOnce(() => currentLineRef)
      .mockImplementation((val) => ({ current: val }));

    // Set up store mock
    mockStore = {
      audioState: {
        currentTime: 0,
        isPlaying: false
      },
      seek: jest.fn()
    };
    useStore.mockImplementation(() => mockStore);
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it('highlights the current line based on time', () => {
    mockStore = {
      audioState: {
        currentTime: 3,  // This is within Line 2's range (2-4)
        isPlaying: false
      },
      seek: jest.fn()
    };
    useStore.mockImplementation(() => mockStore);

    const { getByText } = render(
      <TranscriptViewer transcript={mockTranscript} />
    );

    // Let effects run
    act(() => {
      jest.runAllTimers();
    });

    const line2Element = getByText('Line 2').closest('.rounded-lg');
    expect(line2Element.classList.contains('bg-blue-100')).toBe(true);
    expect(line2Element.classList.contains('bg-gray-50')).toBe(false);
  });

  it('seeks to correct time when clicking a line', () => {
    const { getByText } = render(
      <TranscriptViewer transcript={mockTranscript} />
    );

    fireEvent.click(getByText('Line 3'));
    expect(mockStore.seek).toHaveBeenCalledWith(4); // Start time of Line 3
  });

  describe('Auto-scrolling behavior', () => {
    it('auto-scrolls to keep current line visible during playback', () => {
      // Set up initial store state with playback
      mockStore = {
        audioState: {
          currentTime: 6, // Line 4 (index 3)
          isPlaying: true
        },
        seek: jest.fn()
      };
      useStore.mockImplementation(() => mockStore);

      // Create a mock container with scroll methods
      const mockScrollContainer = {
        ...mockContainer,
        scrollTo: mockScrollTo,
        getBoundingClientRect: () => ({
          top: 0,
          bottom: 300,
          height: 300
        }),
        scrollTop: 0,
        clientHeight: 300,
        scrollHeight: 1000
      };

      // Mock the current line element
      const mockCurrentLine = {
        getBoundingClientRect: () => ({
          top: 350,  // Position below viewport
          bottom: 400,
          height: 50
        })
      };

      // Set up refs
      containerRef = { current: mockScrollContainer };
      currentLineRef = { current: mockCurrentLine };

      // Mock useRef specifically for this test
      jest.spyOn(React, 'useRef')
        .mockImplementationOnce(() => containerRef)
        .mockImplementationOnce(() => currentLineRef)
        .mockImplementation((val) => ({ current: val }));

      // Render with mocked refs
      render(
        <TranscriptViewer transcript={mockTranscript} />
      );

      // Let effects run
      act(() => {
        jest.runAllTimers();
      });

      // Verify scroll was called to bring line into view
      expect(mockScrollTo).toHaveBeenCalledWith({
        top: expect.any(Number),
        behavior: 'smooth'
      });
    });

    it('auto-scrolls when playing starts', () => {
      mockStore = {
        audioState: {
          currentTime: 6, // Line 4
          isPlaying: false
        },
        seek: jest.fn()
      };
      useStore.mockImplementation(() => mockStore);

      const { rerender } = render(
        <TranscriptViewer transcript={mockTranscript} />
      );

      // Update store to start playing
      mockStore.audioState.isPlaying = true;
      rerender(<TranscriptViewer transcript={mockTranscript} />);

      act(() => {
        jest.runAllTimers();
      });

      expect(mockScrollTo).toHaveBeenCalled();
    });

    it('resumes auto-scroll after playing starts', () => {
      mockStore = {
        audioState: {
          currentTime: 6,
          isPlaying: true
        },
        seek: jest.fn()
      };
      useStore.mockImplementation(() => mockStore);

      render(
        <TranscriptViewer transcript={mockTranscript} />
      );

      // Get the scroll handler from addEventListener mock
      expect(mockContainer.addEventListener).toHaveBeenCalledWith('scroll', expect.any(Function));
      const scrollHandler = mockContainer.addEventListener.mock.calls[0][1];

      // Simulate user scroll by calling the scroll handler
      scrollHandler();

      // Let effects run
      act(() => {
        jest.runAllTimers();
      });

      // Should auto-scroll when playing starts
      expect(mockScrollTo).toHaveBeenCalled();
    });

    it('scrolls to top for lines near the start', () => {
      mockStore = {
        audioState: {
          currentTime: 1,
          isPlaying: true
        },
        seek: jest.fn()
      };
      useStore.mockImplementation(() => mockStore);

      render(<TranscriptViewer transcript={mockTranscript} />);

      // Let effects run
      act(() => {
        jest.runAllTimers();
      });

      expect(mockScrollTo).toHaveBeenCalledWith({
        top: 0,
        behavior: 'smooth'
      });
    });
  });
});
