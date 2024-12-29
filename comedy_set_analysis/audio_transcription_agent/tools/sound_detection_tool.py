from base_tool import BaseTool
from pydantic import Field
import os
import whisper_at as whisper
import json

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class SoundDetectionTool(BaseTool):
    """
    Detects sounds in audio files using OpenAI's Whisper model and the whisper_at fork.
    """
    audio_file_path: str = Field(..., description="Path to the input audio file (m4a format).")
    output_directory: str = Field(..., description="Directory where the sound detection output will be saved.")

    def run(self):
        """
        Detects sounds in audio files using OpenAI's Whisper model and the whisper_at fork.

        Returns the path to the sound detection JSON file.
        """    
        sound_class_list = [
                10, # Whoop
                16, # Laughter
                18, # Giggle
                19, # Snicker
                21, # Chuckle, chortle
                25, # Wail, moan
                38, # Groan
                63, # Clapping
                66, # Cheering
                67, # Applause
            ]
        
        audio_tagging_time_resolution = 1.2

        model = whisper.load_model("large-v1")
        result = model.transcribe(self.audio_file_path, at_time_res=audio_tagging_time_resolution, fp16=False)
        audio_tag_result = whisper.parse_at_label(result, language='en', top_k=5, p_threshold=-1, include_class_list=sound_class_list)

        output_file_path = os.path.join(self.output_directory, "sounds_raw.json")
        with open(output_file_path, "w") as output_file:
            json.dump(audio_tag_result, output_file, indent=4)
        
        logger.info(f"Sounds saved to {output_file_path}")
        return output_file_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect sounds and laughter in an audio file')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to the input audio file')
    parser.add_argument('--output', '-o', required=True,
                       help='Directory where the sound detection results will be saved')
    
    args = parser.parse_args()
    
    tool = SoundDetectionTool(
        audio_file_path=args.input,
        output_directory=args.output
    )
    print(tool.run())