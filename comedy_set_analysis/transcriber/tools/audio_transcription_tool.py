import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field
import whisper as whisper
import json
import argparse

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class AudioTranscriptionTool(BaseTool):
    """
    Transcribes audio files using OpenAI's Whisper model.
    """
    audio_file_path: str = Field(..., description="Path to the input audio file (m4a format).")
    output_directory: str = Field(..., description="Directory where the transcription output will be saved.")

    def run(self):
        """
        Transcribes the audio file and saves the result to a JSON file.

        Returns the path to the transcription JSON file.
        """
        model_name = "large-v3"
        model = whisper.load_model(model_name)
        logger.info(f"Transcribing {self.audio_file_path} with Whisper model {model_name}")
        result = model.transcribe(self.audio_file_path)
        
        output_file_path = os.path.join(self.output_directory, "transcript_raw.json")
        with open(output_file_path, "w") as output_file:
            json.dump(result, output_file, indent=4)
        
        logger.info(f"Transcribed {self.audio_file_path} and saved to {output_file_path}")
        return output_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio files using OpenAI Whisper')
    parser.add_argument('--input', '-i', required=True, help='Path to the input audio file (m4a format)')
    parser.add_argument('--output', '-o', required=True, help='Directory where the transcription output will be saved')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    tool = AudioTranscriptionTool(
        audio_file_path=args.input,
        output_directory=args.output
    )
    print(f"Transcription saved to: {tool.run()}")