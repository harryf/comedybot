// Get the full path for any file in the asset directory
export function getAssetPath(filename) {
  // Check if we're running in the Jekyll environment
  if (window.PLAYER_CONFIG?.assetPath) {
    const path = `${window.PLAYER_CONFIG.assetPath}/${filename}`;
    console.log('Asset path (Jekyll):', {
      config: window.PLAYER_CONFIG,
      filename,
      resultPath: path
    });
    return path;
  }
  
  // Default path for local development
  const path = `/audio/${filename}`;
  console.log('Asset path (dev):', {
    filename,
    resultPath: path
  });
  return path;
}
