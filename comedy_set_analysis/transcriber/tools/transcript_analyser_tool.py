import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field
import json
import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class TranscriptAnalyserTool(BaseTool):
    """
    A tool to process raw transcript JSON files from Whisper and output a cleaned version.
    """
    input_file: str = Field(
        ..., description="Path to the input JSON file containing the raw transcript."
    )
    output_directory: str = Field(
        ..., description="Path to the directory where the cleaned transcript will be saved."
    )

    def run(self):
        """
        Process the input transcript file and save the cleaned transcript to the output file.

        Returns True if the processing was successful, False otherwise.
        """
        with open(self.input_file, 'r') as file:
            data = json.load(file)

        # Extract and merge relevant information from each segment
        processed_data = []
        i = 0
        while i < len(data["segments"]):
            current_segment = data["segments"][i]
            start = current_segment["start"]
            end = current_segment["end"]
            seek = current_segment["seek"]
            text = current_segment["text"].lstrip()

            # Skip lines that are entirely sound effects enclosed in square brackets
            if text.strip().startswith('[') and text.strip().endswith(']'):
                i += 1
                continue

            # Check for consecutive segments with the same text and matching end/start times
            while (i + 1 < len(data["segments"]) and
                   data["segments"][i + 1]["text"].lstrip() == text and
                   data["segments"][i + 1]["start"] == end):
                # Merge segments
                end = data["segments"][i + 1]["end"]
                i += 1

            processed_data.append({
                "type": "text",
                "start": start,
                "end": end,
                "seek": seek,
                "text": text
            })

            i += 1

        output_file_path = os.path.join(self.output_directory, "transcript_clean.json")

        # Output the processed data to a JSON file
        with open(output_file_path, 'w') as file:
            json.dump(processed_data, file, indent=4)

        logger.info(f"Processed {self.input_file} and saved to {output_file_path}")
        return output_file_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and process raw transcript data')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to the input JSON file containing raw transcript data')
    parser.add_argument('--output', '-o', required=True,
                       help='Directory where the processed transcript will be saved')
    
    args = parser.parse_args()
    
    tool = TranscriptAnalyserTool(
        input_file=args.input,
        output_directory=args.output
    )
    print(tool.run())