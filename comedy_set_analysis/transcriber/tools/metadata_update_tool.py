import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field
import json
import logging

logger = logging.getLogger(__name__)

class MetadataUpdateTool(BaseTool):
    """
    Updates the metadata.json file with laughs per minute and length of set values from analysis files.
    """
    directory_path: str = Field(
        ...,
        description="Path to the directory containing metadata.json, sounds_clean.json, and transcript_clean.json"
    )

    def run(self):
        """
        Reads values from sounds_clean.json and transcript_clean.json and updates metadata.json
        Returns True if successful, False otherwise.
        """
        try:
            # Read laughs per minute from sounds_clean.json
            sounds_path = os.path.join(self.directory_path, 'sounds_clean.json')
            with open(sounds_path, 'r') as f:
                sounds_data = json.load(f)
                laughs_per_minute = sounds_data.get('summary', {}).get('laughs_per_minute')

            # Read length of set from transcript_clean.json
            transcript_path = os.path.join(self.directory_path, 'transcript_clean.json')
            with open(transcript_path, 'r') as f:
                transcript_data = json.load(f)
                if transcript_data and len(transcript_data) > 0:
                    length_of_set = transcript_data[-1].get('end')
                else:
                    length_of_set = None

            # Update metadata.json
            metadata_path = os.path.join(self.directory_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Update values
            metadata['laughs_per_minute'] = laughs_per_minute
            metadata['length_of_set'] = length_of_set

            # Write updated metadata back to file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Successfully updated metadata with laughs_per_minute: {laughs_per_minute} and length_of_set: {length_of_set}")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            return False

if __name__ == "__main__":
    # Test the tool
    import argparse
    
    parser = argparse.ArgumentParser(description="Update metadata.json with analysis results")
    parser.add_argument('-d', '--directory_path', type=str, required=True, 
                       help='Directory containing metadata.json and analysis files')
    
    args = parser.parse_args()
    
    tool = MetadataUpdateTool(directory_path=args.directory_path)
    result = tool.run()
    print(f"Metadata update {'successful' if result else 'failed'}") 