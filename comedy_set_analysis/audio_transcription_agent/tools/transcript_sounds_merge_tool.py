from agency_swarm.tools import BaseTool
from pydantic import Field
import json
import os
import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class TranscriptSoundsMergeTool(BaseTool):
    """
    A tool to merge cleaned transcript and sound JSON files into a single file.

    CURRENTLY NOT USED
    """
    transcript_file: str = Field(
        ..., description="Path to the cleaned transcript JSON file."
    )
    sounds_file: str = Field(
        ..., description="Path to the cleaned sounds JSON file."
    )
    output_file: str = Field(
        ..., description="Path to the output JSON file where the merged data will be saved."
    )

    def run(self):
        """
        Merge the cleaned transcript and sounds files and save the result to the output file.
        """
        # Load the processed transcript and sounds data
        with open(self.transcript_file, 'r') as file:
            transcript_data = json.load(file)

        with open(self.sounds_file, 'r') as file:
            sounds_data = json.load(file)

        # Combine the two lists
        combined_data = transcript_data + sounds_data

        # Sort the combined list by the start time
        combined_data.sort(key=lambda x: x['start'])

        # Output the combined data to a JSON file
        with open(self.output_file, 'w') as file:
            json.dump(combined_data, file, indent=4)

if __name__ == "__main__":
    tool = TranscriptSoundsMergeTool(
        transcript_file='/Users/harry/Code/comedybot/data/example_audio/transcript_clean.json',
        sounds_file='/Users/harry/Code/comedybot/data/example_audio/sounds_clean.json',
        output_file='/Users/harry/Code/comedybot/data/example_audio/transcript_sounds.json'
    )
    tool.run() 