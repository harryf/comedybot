import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field
import re
from datetime import datetime
import shutil

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class AudioPreparationTool(BaseTool):
    """
    Prepares audio files by checking for existing subdirectories, cleaning filenames, and creating new subdirectories.
    """
    audio_file_path: str = Field(..., description="Path to the input audio file (m4a, wav, or mp3 format).")

    output_directory_path: str = Field(
        ..., 
        description="Output directory to copy audio file to and create JSONs"
    )

    def run(self):
        """
        Prepares the audio file by cleaning the filename and creating a new subdirectory if it doesn't exist.
        Copies the file instead of moving it.
        
        Returns a tuple containing the path to the new subdirectory and the path to the copied file. 
        Returns False if the subdirectory already exists.
        """
        # Extract the filename and extension
        _, filename = os.path.split(self.audio_file_path)
        file_base, file_extension = os.path.splitext(filename)

        # Clean the filename
        cleaned_filename = self.clean_filename(file_base) + file_extension

        # Get the modification time of the file
        modification_time = os.path.getmtime(self.audio_file_path)
        date_prefix = datetime.fromtimestamp(modification_time).strftime('%Y%m%d_')

        # Use the cleaned filename for the subdirectory, prepending the date
        cleaned_subdirectory_path = os.path.join(self.output_directory_path, date_prefix + self.clean_filename(file_base))

        # Check if a subdirectory exists
        if os.path.exists(cleaned_subdirectory_path):
            logger.warning(f"Subdirectory already exists for {self.audio_file_path}, skipping processing.") 
            return False

        # Create the subdirectory
        os.makedirs(cleaned_subdirectory_path, exist_ok=True)

        # Copy the file to the new directory
        cleaned_file_path = os.path.join(cleaned_subdirectory_path, cleaned_filename)

        # Debug logging
        logger.debug(f"Attempting to copy {self.audio_file_path} to {cleaned_file_path}")

        # Copy the file
        shutil.copy2(self.audio_file_path, cleaned_file_path)

        logger.info(f"Copied {self.audio_file_path} to {cleaned_file_path} and created subdirectory {cleaned_subdirectory_path}")
        return (cleaned_subdirectory_path, cleaned_file_path)

    def rollback(self, cleaned_subdirectory_path):
        """
        Removes the created subdirectory and all its contents if processing fails.
        
        Args:
            cleaned_subdirectory_path: Path to the subdirectory to remove
        """
        logger.info(f"Rolling back {cleaned_subdirectory_path}")
        if os.path.exists(cleaned_subdirectory_path):
            try:
                shutil.rmtree(cleaned_subdirectory_path)
                logger.info(f"Successfully removed directory and contents: {cleaned_subdirectory_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to remove directory {cleaned_subdirectory_path}: {str(e)}")
                return False
        return False

    def clean_filename(self, filename):
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        # Transliterate non-ASCII characters
        filename = filename.encode('ascii', 'ignore').decode('ascii')
        # Remove non-alphanumeric characters
        filename = re.sub(r'[^A-Za-z0-9_]', '', filename)
        return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare audio files for transcription by cleaning and converting them')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to the input audio file')
    parser.add_argument('--output', '-o', required=True,
                       help='Directory where processed files will be saved')
    parser.add_argument('--test-rollback', '-r', action='store_true',
                       help='Test the rollback functionality after processing')
    
    args = parser.parse_args()
    
    tool = AudioPreparationTool(
        audio_file_path=args.input,
        output_directory_path=args.output
    )
    result = tool.run()
    print(result)
    
    # Test rollback if requested
    if args.test_rollback and result:
        cleaned_subdirectory_path, _ = result
        print(f"Testing rollback for {cleaned_subdirectory_path}")
        tool.rollback(cleaned_subdirectory_path)