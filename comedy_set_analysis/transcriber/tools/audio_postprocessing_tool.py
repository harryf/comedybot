import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field
import json
from datetime import datetime
import logging
import argparse
import shutil

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class AudioPostprocessingTool(BaseTool):
    """
    A tool to rename the subdirectory based on the date_of_show field in the metadata.json file.
    This is the final step in the processing pipeline.
    """
    directory_path: str = Field(
        ..., description="Path to the directory containing the processed files and metadata.json"
    )
    
    def run(self):
        """
        Renames the subdirectory based on the date_of_show field in the metadata.json file.
        
        Returns the path to the renamed directory or False if renaming failed.
        """
        logger.info(f"Post-processing directory: {self.directory_path}")
        
        # Check if the directory exists
        if not os.path.exists(self.directory_path):
            logger.error(f"Directory does not exist: {self.directory_path}")
            return False
            
        # Check if metadata.json exists
        metadata_path = os.path.join(self.directory_path, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
            
        try:
            # Load the metadata
            with open(metadata_path, 'r') as file:
                metadata = json.load(file)
                
            # Check if date_of_show exists
            if "date_of_show" not in metadata:
                logger.error(f"date_of_show field not found in metadata: {metadata_path}")
                return False
                
            # Parse the date_of_show field (format: "DD MMM YYYY, HH:MM")
            date_of_show = metadata["date_of_show"]
            try:
                # Parse the date string
                date_obj = datetime.strptime(date_of_show, "%d %b %Y, %H:%M")
                
                # Format the date for the new directory name
                date_prefix = date_obj.strftime('%Y%m%d_')
                
                # Get the parent directory and current directory name
                parent_dir = os.path.dirname(self.directory_path)
                current_dir_name = os.path.basename(self.directory_path)
                
                # Check if the directory already has the correct date prefix
                if current_dir_name.startswith(date_prefix):
                    logger.info(f"Directory already has the correct date prefix: {current_dir_name}")
                    return self.directory_path
                
                # Extract the base name (remove the current date prefix if it exists)
                if '_' in current_dir_name and len(current_dir_name.split('_')[0]) == 8:
                    base_name = current_dir_name[9:]  # Remove YYYYMMDD_
                else:
                    base_name = current_dir_name
                
                # Create the new directory name
                new_dir_name = date_prefix + base_name
                new_dir_path = os.path.join(parent_dir, new_dir_name)
                
                # Check if the new directory already exists
                if os.path.exists(new_dir_path):
                    logger.error(f"Target directory already exists: {new_dir_path}")
                    return False
                
                # Rename the directory
                shutil.move(self.directory_path, new_dir_path)
                logger.info(f"Successfully renamed directory from {self.directory_path} to {new_dir_path}")
                
                return new_dir_path
                
            except ValueError as e:
                logger.error(f"Failed to parse date_of_show: {date_of_show}. Error: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing metadata: {str(e)}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename directory based on date_of_show in metadata.json")
    parser.add_argument("--directory", "-d", required=True, help="Path to the directory containing metadata.json")
    
    args = parser.parse_args()
    
    tool = AudioPostprocessingTool(directory_path=args.directory)
    result = tool.run()
    
    if result:
        print(f"Directory renamed to: {result}")
    else:
        print("Failed to rename directory")