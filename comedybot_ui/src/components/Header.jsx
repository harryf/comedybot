import React, { useState } from 'react';
import { Menu } from '@headlessui/react';
import classNames from 'classnames';

const formatDate = (dateString) => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });
};

const Header = ({ metadata }) => {
  const [showInfo, setShowInfo] = useState(false);

  if (!metadata) return null;

  return (
    <header className="fixed top-0 left-0 right-0 bg-blue-600 text-white p-4 z-10" data-testid="header">
      <div className="max-w-3xl mx-auto flex justify-between items-center">
        {/* Left side - Menu */}
        <Menu as="div" className="relative">
          <Menu.Button className="p-2 hover:bg-white/10 rounded-lg">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </Menu.Button>
        </Menu>

        {/* Center - Show Info */}
        <div className="flex flex-col items-center flex-1 mx-4">
          <h1 className="text-lg font-semibold">
            {metadata.name_of_show}
          </h1>
          <div className="text-sm flex items-center gap-2">
            <span>{formatDate(metadata.date_of_show)}</span>
            <span className="text-white/60">•</span>
            <span>{metadata.name_of_venue}</span>
            <span className="text-white/60">•</span>
            <span>{Math.round(metadata.laughs_per_minute)} LPM</span>
          </div>
        </div>

        {/* Right side - Info Button */}
        <button 
          onClick={() => setShowInfo(!showInfo)}
          className="p-2 hover:bg-white/10 rounded-lg"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 16V12M12 8H12.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>

        {/* Info Panel */}
        {showInfo && (
          <div className="absolute top-16 right-4 bg-white text-gray-900 p-4 rounded-lg shadow-lg">
            <p className="mb-2">
              <span className="font-semibold">Date:</span> {formatDate(metadata.date_of_show)}
            </p>
            <p className="mb-2">
              <span className="font-semibold">Duration:</span> {Math.round(metadata.duration / 60)} mins
            </p>
            <p>
              <span className="font-semibold">Laughs:</span> {metadata.total_laughs}
            </p>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header; 