#!/usr/bin/env python3
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

class TranscriptAnalyserTool2(BaseTool):
    """
    A tool to process raw transcript JSON files from whisper-cpp and output a cleaned version.
    This is a replacement for TranscriptAnalyserTool that works with the JSON format from whisper-cpp.
    """
    input_file: str = Field(
        ..., description="Path to the input JSON file containing the raw transcript."
    )
    output_directory: str = Field(
        ..., description="Path to the directory where the cleaned transcript will be saved."
    )

    def parse_timestamp(self, ts):
        """Convert timestamp format 'HH:MM:SS,mmm' to seconds as float"""
        h, m, s = ts.replace(',', '.').split(':')
        return round(float(h) * 3600 + float(m) * 60 + float(s), 2)

    def run(self):
        """
        Process the input transcript file and save the cleaned transcript to the output file.

        Returns the path to the cleaned transcript file.
        """
        logger.info(f"Processing transcript file: {self.input_file}")
        
        # Load the original JSON
        with open(self.input_file, 'r') as file:
            data = json.load(file)

        # Process the data
        output = []
        for entry in data.get("transcription", []):
            start = self.parse_timestamp(entry["timestamps"]["from"])
            end = self.parse_timestamp(entry["timestamps"]["to"])
            text = entry["text"].strip()

            # Skip lines that are entirely sound effects enclosed in brackets
            if (text.startswith('[') and text.endswith(']')) or (text.startswith('(') and text.endswith(')')): 
                continue

            output.append({
                "type": "text",
                "start": start,
                "end": end,
                "text": text
            })

        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Define output file path
        output_file_path = os.path.join(self.output_directory, "transcript_clean.json")
        
        # Save the result
        with open(output_file_path, 'w') as file:
            json.dump(output, file, indent=4)
        
        logger.info(f"Processed {self.input_file} and saved to {output_file_path}")
        return output_file_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and process raw transcript data from whisper-cpp')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to the input JSON file containing raw transcript data')
    parser.add_argument('--output', '-o', required=True,
                       help='Directory where the processed transcript will be saved')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    tool = TranscriptAnalyserTool2(
        input_file=args.input,
        output_directory=args.output
    )
    print(f"Processed transcript saved to: {tool.run()}")
