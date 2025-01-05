import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field
from typing import List
import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class AudioFileFinderTool(BaseTool):
    """
    Scans a directory for audio files (m4a, wav, mp3) and returns their full paths.
    This tool is designed to work in conjunction with AudioPreparationTool.
    """
    directory_path: str = Field(
        ..., 
        description="Path to the directory to scan for audio files"
    )

    def run(self):
        """
        Returns a list of audio files found in the directory.
        """
        # Supported audio formats
        audio_extensions = ('.m4a', '.wav', '.mp3')
        logger.debug(f"Looking for files ending in {audio_extensions} in {self.directory_path}")
        
        # List to store found audio files
        audio_files = []

        try:
            # Walk through the directory
            for root, _, files in os.walk(self.directory_path):
                for file in files:
                    if file.lower().endswith(audio_extensions):
                        full_path = os.path.join(root, file)
                        logger.info(f"Found: {full_path}")
                        audio_files.append(full_path)

            if not audio_files:
                return "No audio files found in the specified directory."

            # Return the list of files as a formatted string
            return audio_files

        except Exception as e:
            return f"Error scanning directory: {str(e)}"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find audio files (m4a, wav, mp3) in a specified directory')
    parser.add_argument('--directory', '-d', required=True, 
                       help='Path to the directory to scan for audio files')
    
    args = parser.parse_args()
    
    tool = AudioFileFinderTool(directory_path=args.directory)
    print(tool.run())