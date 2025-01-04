#!/usr/bin/env python3

"""
Tool to extract flattened text from bits in a directory.
Creates a JSON file containing bit titles and their full text content.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import logging
from typing import List, Dict, Any
from pydantic import Field
from base_tool import BaseTool
from bit_utils import (
    read_bits_file, select_bit, flatten_bit
)

logger = logging.getLogger(__name__)

class BitTextExtractor(BaseTool):
    directory_path: str = Field(description="Path to directory containing bits.json and transcript.json")
    output_file: str = Field(description="Path to output JSON file")
    bits_file: str = Field(description="Path to bits.json file")
    transcript_file: str = Field(description="Path to transcript file")
    
    def __init__(self, directory_path: str, output_file: str = None):
        # Find bits.json and transcript.json in the directory
        bits_file = os.path.join(directory_path, 'bits.json')
        transcript_file = os.path.join(directory_path, 'transcript_clean.json')
        
        if not os.path.exists(bits_file):
            raise FileNotFoundError(f"Could not find bits.json in {directory_path}")
        if not os.path.exists(transcript_file):
            raise FileNotFoundError(f"Could not find transcript_clean.json in {directory_path}")
            
        # If no output file specified, create one in the same directory
        if output_file is None:
            output_file = os.path.join(directory_path, 'bit_texts.json')
            
        super().__init__(
            directory_path=directory_path,
            output_file=output_file,
            bits_file=bits_file,
            transcript_file=transcript_file
        )
        
        self.bits_file = bits_file
        self.transcript_file = transcript_file
        
    def read_transcript_file(self) -> List[Dict[str, Any]]:
        """Read and return the transcript data."""
        with open(self.transcript_file, 'r') as f:
            return json.load(f)
            
    def extract_texts(self) -> List[Dict[str, str]]:
        """Extract title and flattened text for each bit."""
        # Read input files
        bits = read_bits_file(self.bits_file)
        transcript_data = self.read_transcript_file()
        
        bit_texts = []
        for bit in bits:
            logger.info(f"Processing bit: {bit['title']}")
            
            # Get bit text from transcript
            bit_items = select_bit(transcript_data, bit['start'], bit['end'])
            text = flatten_bit(bit_items)
            
            bit_texts.append({
                "title": bit['title'],
                "text": text
            })
            
        return bit_texts
        
    def run(self) -> None:
        """Extract texts and save to output file."""
        bit_texts = self.extract_texts()
        
        # Save to output file
        with open(self.output_file, 'w') as f:
            json.dump(bit_texts, f, indent=2)
            
        logger.info(f"Extracted {len(bit_texts)} bit texts")
        logger.info(f"Saved to: {self.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract flattened text from bits in a directory.'
    )
    parser.add_argument('-d', '--directory', type=str, required=True,
                      help='Directory containing bits.json and transcript.json')
    parser.add_argument('-o', '--output', type=str,
                      help='Output JSON file path (default: bit_texts.json in same directory)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                      format='%(message)s')
    
    tool = BitTextExtractor(
        directory_path=args.directory,
        output_file=args.output
    )
    tool.run()
