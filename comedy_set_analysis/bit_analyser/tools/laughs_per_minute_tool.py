#!/usr/bin/env python3

"""
Tool for calculating laughs per minute (LPM) for comedy bits.

This tool:
1. Reads bits from a bits.json file
2. Reads audience reactions from a reactions.json file
3. For each bit:
   - Finds relevant reactions (from bit start to 10s after bit end)
   - Calculates total reaction score
   - Computes laughs per minute
4. Updates bits.json with LPM scores incrementally
"""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pydantic import Field
from base_tool import BaseTool
from bit_utils import read_bits_file, write_bits_file, should_process_bit

logger = logging.getLogger(__name__)

class LaughsPerMinuteTool(BaseTool):
    bits_file_path: str = Field(description="Path to the bits JSON file")
    reactions_file_path: str = Field(description="Path to the audience reactions JSON file")
    regenerate: bool = Field(default=False, description="Force regeneration of all LPM calculations")
    
    def __init__(self, bits_file_path: str, reactions_file_path: str, regenerate: bool = False):
        super().__init__(
            bits_file_path=bits_file_path,
            reactions_file_path=reactions_file_path,
            regenerate=regenerate
        )
        
    def read_reactions_file(self) -> List[Dict[str, Any]]:
        """Read and return the reactions list from the reactions JSON file."""
        with open(self.reactions_file_path, 'r') as f:
            data = json.load(f)
            return data.get('reactions', [])
            
    def find_bit_reactions(self, bit: Dict[str, Any], reactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find all reactions that occur during a bit and up to 10 seconds after it ends.
        
        Args:
            bit: Dictionary containing bit data with 'start' and 'end' times
            reactions: List of reaction dictionaries with 'start' and 'end' times
            
        Returns:
            List of reaction dictionaries that apply to this bit
        """
        bit_start = bit['start']
        bit_end = bit['end']
        max_end = bit_end + 5  # Include reactions up to 5 seconds after bit ends
        
        # Find reactions that start at or after bit start, and before max_end
        bit_reactions = [
            reaction for reaction in reactions
            if reaction['start'] >= bit_start and reaction['start'] < max_end
        ]
        
        return bit_reactions
        
    def calculate_laughs_per_minute(self, bit: Dict[str, Any], reactions: List[Dict[str, Any]]) -> int:
        """
        Calculate laughs per minute for a bit based on its reactions.
        
        Args:
            bit: Dictionary containing bit data
            reactions: List of reaction dictionaries that apply to this bit
            
        Returns:
            Integer representing laughs per minute
        """
        # Sum up all reaction scores
        total_score = sum(reaction['reaction_score'] for reaction in reactions)
        
        # Calculate bit length in seconds
        bit_length_seconds = bit['end'] - bit['start']
        
        # Calculate and round LPM
        if bit_length_seconds > 0:
            # Convert to per minute rate: (score * 60) / seconds
            lpm = round((total_score * 60) / bit_length_seconds)
        else:
            lpm = 0
            
        return lpm
        
    def run(self):
        """
        Process all bits to calculate their laughs per minute scores.
        Updates bits.json incrementally as each bit is processed.
        """
        # Read input files
        bits = read_bits_file(self.bits_file_path)
        reactions = self.read_reactions_file()
        
        # Track if we processed any bits
        processed_any = False
        
        # Process each bit
        for bit in bits:
            # Skip if bit already has LPM and we're not regenerating
            if not should_process_bit(bit, 'lpm', self.regenerate):
                logger.info(f"Skipping bit '{bit['title']}' - already has LPM")
                continue
                
            logger.info(f"Processing bit: {bit['title']}")
            processed_any = True
            
            # Find relevant reactions for this bit
            bit_reactions = self.find_bit_reactions(bit, reactions)
            
            # Calculate LPM
            lpm = self.calculate_laughs_per_minute(bit, bit_reactions)
            logger.info(f"Calculated LPM: {lpm}")
            
            # Update bit with LPM
            bit['lpm'] = lpm
            
            # Write updated bits back to file
            write_bits_file(self.bits_file_path, bits)
            logger.info(f"Saved progress after processing bit: {bit['title']}")
            
        if not processed_any:
            logger.info("No bits needed processing")
        else:
            logger.info("Successfully processed all required bits")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate laughs per minute for comedy bits.')
    parser.add_argument('-b', '--bits-file', type=str, required=True,
                      help='Path to bits JSON file')
    parser.add_argument('-a', '--reactions-file', type=str, required=True,
                      help='Path to audience reactions JSON file')
    parser.add_argument('-r', '--regenerate', action='store_true',
                      help='Force regeneration of all LPM calculations')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                      format='%(message)s')
    
    tool = LaughsPerMinuteTool(
        bits_file_path=args.bits_file,
        reactions_file_path=args.reactions_file,
        regenerate=args.regenerate
    )
    tool.run()
