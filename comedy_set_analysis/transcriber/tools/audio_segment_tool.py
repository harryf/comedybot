import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field
import json
import subprocess
import glob
import logging
import argparse

logger = logging.getLogger(__name__)

class AudioSegmentTool(BaseTool):
    """
    Splits an audio file into 10-second segments using ffmpeg and updates metadata accordingly.
    """
    input_directory: str = Field(
        ..., 
        description="Path to directory containing metadata.json and audio file"
    )

    def run(self):
        """
        Process the audio file in the input directory, splitting it into segments.
        Updates metadata.json with the list of generated segment files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read metadata.json
            metadata_path = os.path.join(self.input_directory, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Get audio file path
            audio_file = metadata.get('audio_file')
            if not audio_file:
                logger.error("No audio_file specified in metadata.json")
                return False

            input_audio_path = os.path.join(self.input_directory, audio_file)
            if not os.path.exists(input_audio_path):
                logger.error(f"Audio file not found: {input_audio_path}")
                return False

            logger.info(f"Processing audio file: {input_audio_path}")

            # Construct ffmpeg command
            ffmpeg_command = [
                'ffmpeg',
                '-i', input_audio_path,
                '-f', 'segment',
                '-segment_time', '10',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-vbr', 'constrained',
                '-b:a', '128k',
                '-reset_timestamps', '1',
                '-map', '0',
                os.path.join(self.input_directory, 'segment_%03d.m4a')
            ]

            logger.info(f"FFmpeg command: {ffmpeg_command}")

            # Run ffmpeg command
            result = subprocess.run(ffmpeg_command, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            # Get list of generated segments in order
            segment_pattern = os.path.join(self.input_directory, 'segment_*.m4a')
            segments = sorted(glob.glob(segment_pattern))
            
            if not segments:
                logger.error("No segments were generated")
                return False

            logger.info(f"Generated {len(segments)} segments")

            # Update segments in metadata with relative paths
            metadata['segments'] = [os.path.basename(s) for s in segments]

            # Write updated metadata back to file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.info("Successfully updated metadata.json with segment information")
            return True

        except Exception as e:
            logger.error(f"Error in audio_segment_tool: {str(e)}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split audio files into 10-second segments using ffmpeg')
    parser.add_argument('--input-dir', '-i', required=True, 
                      help='Path to directory containing metadata.json and audio file')
    
    args = parser.parse_args()
    
    # Ensure input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        exit(1)
        
    tool = AudioSegmentTool(
        input_directory=args.input_dir
    )
    
    if tool.run():
        logger.info("Audio segmentation completed successfully")
    else:
        logger.error("Audio segmentation failed")
        exit(1)
