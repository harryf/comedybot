import { useEffect, useState } from 'react';

export function useHashParams() {
  const [params, setParams] = useState(() => parseHash(window.location.hash));

  useEffect(() => {
    const handleHashChange = () => {
      setParams(parseHash(window.location.hash));
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  return params;
}

function parseHash(hash) {
  if (!hash || hash === '#') return {};
  
  // Remove the leading # and split by &
  const params = {};
  const pairs = hash.substring(1).split('&');
  
  for (const pair of pairs) {
    const [key, value] = pair.split('=');
    if (key === 't') {
      const seconds = parseFloat(value);
      if (!isNaN(seconds) && seconds >= 0) {
        params.time = seconds;
      }
    } else if (key === 'play') {
      params.autoplay = value === 'true';
    }
  }
  
  return params;
}
