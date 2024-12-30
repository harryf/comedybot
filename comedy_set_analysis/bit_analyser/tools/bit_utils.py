#!/usr/bin/env python3

"""
Utility functions for working with comedy bits and their analysis.

This module provides functions for:
1. Selecting and extracting bits from transcript data
2. Reading and writing bit analysis data
3. Processing and validating bit properties
4. Converting bit data between different formats

The main data structures handled by these utilities are:
- Transcript data: List of dictionaries containing text and timing information
- Bits data: List of dictionaries containing bit titles, timing, and analysis properties
"""

import json
import argparse
from typing import List, Dict, Any

def select_bit(transcript_data: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    """
    Select items from transcript data that fall within the given start and end times.
    
    Args:
        transcript_data: List of transcript items with start and end times
        start: Start time to select from
        end: End time to select to
        
    Returns:
        List of transcript items that fall within the time range
    """
    return [
        item for item in transcript_data 
        if (item["start"] >= start and item["end"] <= end)
    ]

def flatten_bit(bit_data: List[Dict[str, Any]]) -> str:
    """
    Convert a list of transcript items into a single string with each text on a new line.
    Useful for preparing bit text for analysis by AI assistants.
    
    Args:
        bit_data: List of transcript items with text property
        
    Returns:
        String containing all text items separated by newlines
    """
    return "\n".join(item["text"] for item in bit_data)

def read_bits_file(bits_file_path: str) -> List[Dict[str, Any]]:
    """
    Read a bits.json file and return its items as a list.
    The file should contain a JSON object with an 'items' array of bit data.
    
    Args:
        bits_file_path: Path to the bits.json file
        
    Returns:
        List of bit items from the file
    """
    with open(bits_file_path, 'r') as f:
        data = json.load(f)
        return data.get('items', [])

def write_bits_file(bits_file_path: str, items: List[Dict[str, Any]]) -> None:
    """
    Write items to a bits.json file, preserving the file structure.
    Any additional properties in the items will be preserved.
    
    Args:
        bits_file_path: Path to the bits.json file
        items: List of bit items to write. Each item should at minimum have:
               - title: String name of the bit
               - start: Float start time in seconds
               - end: Float end time in seconds
               Additional properties (like joke_types, themes) will be preserved
    """
    data = {
        "type": "object",
        "properties": {},
        "items": items
    }
    
    with open(bits_file_path, 'w') as f:
        json.dump(data, f, indent=2)

def should_process_bit(bit: Dict[str, Any], property_name: str, force_regenerate: bool = False) -> bool:
    """
    Determine if a bit needs to be processed based on existing property and regenerate flag.
    Used to check if a bit needs analysis or can be skipped.
    
    Args:
        bit: The bit data to check
        property_name: Name of the property to check for (e.g., 'themes', 'joke_types')
        force_regenerate: If True, always return True to force reprocessing
        
    Returns:
        True if the bit should be processed, False otherwise
    """
    if force_regenerate:
        return True
        
    if property_name not in bit:
        return True
        
    if not isinstance(bit[property_name], list) or not bit[property_name]:
        return True
        
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select and format bit segments.')
    parser.add_argument('-t', '--transcript-file', type=str, required=True,
                      help='Path to transcript JSON file')
    parser.add_argument('-s', '--start', type=float, required=True,
                      help='Start time in seconds')
    parser.add_argument('-e', '--end', type=float, required=True,
                      help='End time in seconds')
    parser.add_argument('-f', '--flatten', action='store_true',
                      help='Output as flattened text')
    
    args = parser.parse_args()
    
    # Read transcript data
    with open(args.transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    # Select bit
    bit_data = select_bit(transcript_data, args.start, args.end)
    
    if args.flatten:
        # Output flattened text
        print(flatten_bit(bit_data))
    else:
        # Output JSON
        print(json.dumps(bit_data, indent=2))
