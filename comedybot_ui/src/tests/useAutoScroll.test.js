import React from 'react';
import { render, act, fireEvent } from '@testing-library/react';
import useAutoScroll from '../hooks/useAutoScroll';

// Mock scrollTo since it's not implemented in JSDOM
const mockScrollTo = jest.fn();
Element.prototype.scrollTo = mockScrollTo;

// Mock getBoundingClientRect
const mockGetBoundingClientRect = jest.fn();
Element.prototype.getBoundingClientRect = mockGetBoundingClientRect;

// Test component that uses the hook
function TestComponent({ activeIndex, isPlaying, items }) {
  const containerRef = React.useRef(null);
  useAutoScroll(containerRef, activeIndex, isPlaying, items);
  
  return (
    <div ref={containerRef} className="transcript-container" style={{ height: '600px' }}>
      {items.map((_, index) => (
        <div key={index} className="rounded-lg" data-testid={`line-${index}`}>
          Line {index + 1}
        </div>
      ))}
    </div>
  );
}

describe('useAutoScroll', () => {
  const mockItems = Array(10).fill(null);

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Mock getBoundingClientRect for container and lines
    mockGetBoundingClientRect.mockImplementation(function() {
      if (this.classList?.contains('transcript-container')) {
        return {
          top: 0,
          height: 600
        };
      }
      // For transcript lines
      return {
        top: 100 * (this.dataset.testid?.split('-')[1] || 1),
        height: 50
      };
    });
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('scrolls to one-third position when active index changes', () => {
    const { rerender } = render(
      <TestComponent activeIndex={0} isPlaying={true} items={mockItems} />
    );

    rerender(
      <TestComponent activeIndex={3} isPlaying={true} items={mockItems} />
    );

    expect(mockScrollTo).toHaveBeenCalled();
    const scrollCall = mockScrollTo.mock.calls[0][0];
    expect(scrollCall).toHaveProperty('behavior', 'smooth');
  });

  it('does not scroll when paused and user has scrolled', () => {
    const { container } = render(
      <TestComponent activeIndex={2} isPlaying={false} items={mockItems} />
    );

    // Reset mock calls from initial render
    mockScrollTo.mockClear();

    // Simulate user scroll
    fireEvent.scroll(container.firstChild);

    // Change active index while paused
    render(
      <TestComponent activeIndex={3} isPlaying={false} items={mockItems} />
    );

    // Should not auto-scroll while paused
    expect(mockScrollTo).not.toHaveBeenCalled();
  });

  it('resumes auto-scroll after user scroll timeout', () => {
    const { container, rerender } = render(
      <TestComponent activeIndex={2} isPlaying={true} items={mockItems} />
    );

    // Simulate user scroll
    fireEvent.scroll(container.firstChild);

    // Reset mock calls
    mockScrollTo.mockClear();

    // Fast-forward past the scroll timeout
    act(() => {
      jest.advanceTimersByTime(1100);
    });

    // Change active index
    rerender(
      <TestComponent activeIndex={3} isPlaying={true} items={mockItems} />
    );

    // Should auto-scroll after timeout
    expect(mockScrollTo).toHaveBeenCalled();
  });

  it('scrolls to top for items near the start', () => {
    render(
      <TestComponent activeIndex={1} isPlaying={true} items={mockItems} />
    );

    expect(mockScrollTo).toHaveBeenCalledWith({
      top: 0,
      behavior: 'smooth'
    });
  });

  it('scrolls to bottom for items near the end', () => {
    const { container } = render(
      <TestComponent activeIndex={mockItems.length - 2} isPlaying={true} items={mockItems} />
    );

    expect(mockScrollTo).toHaveBeenCalledWith({
      top: expect.any(Number),
      behavior: 'smooth'
    });
  });

  it('cleans up scroll listener on unmount', () => {
    const removeEventListenerSpy = jest.spyOn(Element.prototype, 'removeEventListener');
    
    const { unmount } = render(
      <TestComponent activeIndex={2} isPlaying={true} items={mockItems} />
    );

    unmount();

    expect(removeEventListenerSpy).toHaveBeenCalledWith('scroll', expect.any(Function));
  });
});
