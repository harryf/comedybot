import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import json
from typing import Dict, Optional, Any
import re

import logging
logger = logging.getLogger(__name__)

from bit_utils import select_bit, flatten_bit

class BitEntity:
    def __init__(self, directory_path: str):
        """
        Initialize a BitEntity instance.
        
        Args:
            directory_path (str): Path to the directory containing the bit data
            
        Raises:
            ValueError: If bit_data is provided but doesn't contain required fields or has invalid types
        """
        self.directory_path = directory_path
            
        self.bit_data = {
            "bit_id": None,
            "show_info": {},
            "bit_info": {},
            "transcript": {
                "text": "",
                "lines": []
            },
            "audience_reactions": []
        }
    
    def load_from_database(self, bit_id: str):
        bit_data_path = os.path.join(self.directory_path, f"{bit_id}.json")
        if not os.path.exists(bit_data_path):
            raise FileNotFoundError(f"Bit data file not found: {bit_data_path}")
        self.bit_data = json.load(open(bit_data_path, 'r'))
    
    def load_from_set(self, bit_data: Dict[str, Any]):
        self._validate_bit_data(bit_data)
        self.bit_data["bit_info"] = bit_data
        self._load_show_info()
        self.bit_data["bit_id"] = self._generate_bit_id()
        self._load_transcript()
        self._load_audience_reactions()


    def _validate_bit_data(self, bit_data: Dict[str, Any]) -> None:
        """
        Validate the bit_data structure contains all required fields with correct types.
        
        Args:
            bit_data: Dictionary containing bit information
            
        Raises:
            ValueError: If validation fails
        """
        required_fields = {
            "title": str,
            "start": (int, float),
            "end": (int, float),
            "joke_types": list,
            "themes": list,
            "lpm": int
        }
        
        for field, expected_type in required_fields.items():
            if field not in bit_data:
                raise ValueError(f"Missing required field: {field}")
            
            if not isinstance(bit_data[field], expected_type):
                raise ValueError(f"Field {field} must be of type {expected_type.__name__}, got {type(bit_data[field]).__name__}")
                
        # Additional validation for start/end times
        if bit_data["start"] >= bit_data["end"]:
            raise ValueError("Start time must be less than end time")

        
    def _load_show_info(self) -> None:
        """
        Load show information from metadata.json in the specified directory.
        Updates the show_info section of the bit data structure.
        """
        metadata_path = os.path.join(self.directory_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {self.directory_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        path = Path(self.directory_path)
        show_identifier = path.name if path.name else path.parent.name
            
        # Map metadata fields to show_info structure
        self.bit_data["show_info"] = {
            "show_identifier": show_identifier,
            "comedian": metadata.get("comedian"),
            "name_of_show": metadata.get("name_of_show"),
            "date_of_show": metadata.get("date_of_show"),
            "name_of_venue": metadata.get("name_of_venue"),
            "link_to_venue_on_google_maps": metadata.get("link_to_venue_on_google_maps"),
            "length_of_set": metadata.get("length_of_set"),
            "lpm": metadata.get("laughs_per_minute"),
            "notes": metadata.get("notes")
        }
    
    def _generate_bit_id(self) -> str:
        """
        Generate a bit ID using show identifier and sanitized title.
        Only allows ASCII alphanumeric characters and underscores in the title portion.
        """
        title = self.bit_data["bit_info"]["title"].lower()
        # Replace spaces with underscores and remove all non-alphanumeric characters
        sanitized_title = re.sub(r'[^a-z0-9_]', '', title.replace(' ', '_'))
        bit_id = f"{self.bit_data['show_info']['show_identifier']}_{sanitized_title}"
        return bit_id
    
    def _load_transcript(self):
        transcript_path = os.path.join(self.directory_path, "transcript_clean.json")
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"transcript_clean.json not found in {self.directory_path}")
        transcript_data = json.load(open(transcript_path, 'r'))
        transcript_lines = select_bit(transcript_data, self.bit_data["bit_info"]["start"], self.bit_data["bit_info"]["end"])
        transcript_lines = [{k:v for k,v in line.items() if k in ("start", "end", "text")} for line in transcript_lines]
        segments = []
        for line in transcript_lines:
            segments.append(line["text"].strip())
        transcript_text = " ".join(segments).strip()
        
        self.bit_data["transcript"]["text"] = transcript_text
        self.bit_data["transcript"]["lines"] = transcript_lines
    
    def _load_audience_reactions(self):
        reactions_path = os.path.join(self.directory_path, "sounds_clean.json")
        if not os.path.exists(reactions_path):
            raise FileNotFoundError(f"sounds_clean.json not found in {self.directory_path}")
        data = json.load(open(reactions_path, 'r'))
        reaction_data = data["reactions"]  # Get the reactions array from the JSON
        bit_start = self.bit_data["bit_info"]["start"]
        bit_end = self.bit_data["bit_info"]["end"]
        max_end = bit_end + 3  # Include reactions up to 3 seconds after bit ends
        
        # Find reactions that start at or after bit start, and before max_end
        # Find index of first reaction that starts at or after bit start
        start_idx = next((i for i, reaction in enumerate(reaction_data) if reaction['start'] >= bit_start), None)
        if start_idx is None:
            bit_reactions = []
        else:
            # Find index of first reaction that starts after max_end
            end_idx = next((i for i, reaction in enumerate(reaction_data[start_idx:]) if reaction['start'] > max_end), None)
            if end_idx is None:
                bit_reactions = reaction_data[start_idx:]
            else:
                bit_reactions = reaction_data[start_idx:start_idx+end_idx]
        
        # Convert reactions to tuples of (start, end, reaction_score)
        self.bit_data["audience_reactions"] = [
            (reaction["start"], reaction["end"], reaction["reaction_score"])
            for reaction in bit_reactions
        ]
        
    
    def print(self):
        print(json.dumps(self.bit_data, indent=2))

    def write_to_database(self, directory_path: Optional[str] = None) -> None:
        """
        Write the bit data to a JSON file in the specified directory.
        
        Args:
            directory_path (Optional[str]): Directory to write the JSON file to.
                                          If not provided, uses self.directory_path
        """
        if not self.bit_data["bit_id"]:
            raise ValueError("bit_id is not set. Call load_from_set first.")
            
        target_dir = directory_path if directory_path else self.directory_path
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        file_path = os.path.join(target_dir, f"{self.bit_data['bit_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(self.bit_data, f, indent=2)

if __name__ == "__main__":
    dir_name = "/Users/harry/Code/comedybot/docs/assets/audio/20241102_The_Comedy_Clubhouse"
    b = BitEntity(dir_name)
    b.load_from_set({"title": "Joke", "start": 19, "end": 43, "joke_types": ["punchline"], "themes": ["comedy"], "lpm": 5})
    b.print()
    # b.write_to_database()  # Writes to the same directory