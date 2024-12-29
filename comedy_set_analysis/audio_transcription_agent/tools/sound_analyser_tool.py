from base_tool import BaseTool
from pydantic import Field
import json
import os
import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class SoundAnalyserTool(BaseTool):
    """
    A tool to process raw sound JSON files and output a cleaned version with reaction scores.
    """
    input_file: str = Field(
        ..., description="Path to the input JSON file containing the raw sound data."
    )
    output_directory: str = Field(
        ..., description="Path to the directory where the cleaned sound data will be saved."
    )

    def run(self):
        """
        Process the input sound file and save the cleaned sound data to the output file.

        Returns the path to the cleaned sound data JSON file.
        """
        with open(self.input_file, 'r') as file:
            data = json.load(file)

        processed_data = []
        total_score = 0
        cumulative_score = 0

        for item in data:
            start = round(item['time']['start'], 1)
            end = round(item['time']['end'], 1)
            audio_tags = {tag[0] for tag in item['audio tags']}  # Use a set to avoid duplicates

            # Calculate reaction score
            reaction_score = self.calculate_reaction_score(audio_tags)
            total_score += reaction_score
            cumulative_score += reaction_score

            processed_data.append({
                "start": start,
                "end": end,
                "audio tags": list(audio_tags),  # Convert set back to list
                "reaction_score": reaction_score,
                "cumulative_score": cumulative_score
            })

        # Calculate laughs_per_minute
        last_end_time = processed_data[-1]['end'] if processed_data else 0
        laughs_per_minute = round(total_score / (last_end_time / 60)) if last_end_time > 0 else 0

        # Create the root dictionary with summary
        output_data = {
            "summary": {
                "total_score": total_score,
                "laughs_per_minute": laughs_per_minute
            },
            "reactions": processed_data
        }

        # Step 3: Output the final JSON
        output_file_path = os.path.join(self.output_directory, "sounds_clean.json")
        with open(output_file_path, 'w') as file:
            json.dump(output_data, file, indent=4)
        
        logger.info(f"Processed {self.input_file} and saved to {output_file_path}")
        return output_file_path

    def calculate_reaction_score(self, audio_tags):
        """
        Calculate the reaction score based on the audio tags.
        """
        # Define sets for different categories
        snicker_giggle_chuckle = {"Snicker", "Giggle", "Chuckle, chortle"}
        clapping_cheering_applause = {"Clapping", "Cheering", "Applause"}
        laughter = {"Laughter"}

        # Convert audio_tags to a set for easier operations
        tags_set = set(audio_tags)

        # Calculate score based on rules
        if tags_set & snicker_giggle_chuckle:
            if len(tags_set & snicker_giggle_chuckle) == 1 and tags_set.isdisjoint(clapping_cheering_applause | laughter):
                return 1
            elif len(tags_set & snicker_giggle_chuckle) >= 2 and tags_set.isdisjoint(clapping_cheering_applause | laughter):
                return 2
            elif tags_set & clapping_cheering_applause:
                return 3
        if tags_set & laughter:
            if tags_set.isdisjoint(clapping_cheering_applause):
                return 3
            elif tags_set & {"Clapping"}:
                return 4
            elif tags_set & {"Cheering", "Applause"}:
                return 5
        return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and clean raw sound detection data')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to the input JSON file containing raw sound detection data')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to directory where the cleaned sound analysis will be saved')
    
    args = parser.parse_args()
    
    tool = SoundAnalyserTool(
        input_file=args.input,
        output_directory=args.output
    )
    tool.run()