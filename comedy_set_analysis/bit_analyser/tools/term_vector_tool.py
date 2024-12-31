#!/usr/bin/env python3

"""
Tool for calculating term vectors for comedy bits using spaCy.

This tool:
1. Reads bits from a bits.json file
2. Loads corresponding transcript text for each bit
3. Uses spaCy to calculate term vectors
4. Generates a hash for each vector
5. Stores hashes in bits.json and vectors in bit_vectors.json
"""

import os
import json
import argparse
import logging
import numpy as np
import hashlib
import base64
from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field
import spacy
from base_tool import BaseTool
from bit_utils import (
    read_bits_file, write_bits_file, should_process_bit,
    select_bit, flatten_bit
)

logger = logging.getLogger(__name__)

class TermVectorTool(BaseTool):
    bits_file_path: str = Field(description="Path to the bits JSON file")
    transcript_file_path: str = Field(description="Path to the transcript JSON file")
    regenerate: bool = Field(default=False, description="Force regeneration of all term vectors")
    nlp: Any = Field(default=None, description="spaCy language model")
    vectors_file_path: str = Field(description="Path to the vectors JSON file")
    
    def __init__(self, bits_file_path: str, transcript_file_path: str, regenerate: bool = False):
        # Calculate vectors file path
        vectors_file_path = os.path.join(
            os.path.dirname(bits_file_path),
            'bit_vectors.json'
        )
        
        # First initialize pydantic model
        super().__init__(
            bits_file_path=bits_file_path,
            transcript_file_path=transcript_file_path,
            regenerate=regenerate,
            nlp=None,  # Initialize as None first
            vectors_file_path=vectors_file_path
        )
        
        # Then load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model en_core_web_md")
        except OSError:
            logger.error("Failed to load spaCy model. Please install it using:")
            logger.error("python -m spacy download en_core_web_md")
            raise
            
    def read_transcript_file(self) -> List[Dict[str, Any]]:
        """Read and return the transcript data."""
        with open(self.transcript_file_path, 'r') as f:
            return json.load(f)
            
    def read_vectors_file(self) -> Dict[str, List[float]]:
        """Read the vectors file if it exists, otherwise return empty dict."""
        if os.path.exists(self.vectors_file_path):
            with open(self.vectors_file_path, 'r') as f:
                return json.load(f)
        return {}
            
    def write_vectors_file(self, vectors: Dict[str, List[float]]) -> None:
        """Write vectors dictionary to the vectors file."""
        with open(self.vectors_file_path, 'w') as f:
            json.dump(vectors, f, indent=2)
            
    def vector_to_hash(self, vector: np.ndarray) -> str:
        """
        Generate a hash string from a vector.
        
        Args:
            vector: numpy array of vector values
            
        Returns:
            Base64 encoded hash string
        """
        # Convert vector to bytes
        vector_bytes = vector.tobytes()
        
        # Hash using SHA-256
        hash_obj = hashlib.sha256(vector_bytes)
        hash_digest = hash_obj.digest()
        
        # Encode as base64 and clean up
        return base64.urlsafe_b64encode(hash_digest).decode("utf-8").rstrip("=")
        
    def calculate_term_vector(self, text: str) -> Tuple[List[float], str]:
        """
        Calculate term vector for a text using spaCy and generate its hash.
        
        Returns:
            Tuple of (vector as list, hash string)
        """
        doc = self.nlp(text)
        vector = doc.vector
        vector_hash = self.vector_to_hash(vector)
        
        # Convert vector to list for JSON serialization
        return vector.tolist(), vector_hash
        
    def run(self):
        """
        Process all bits to calculate their term vectors.
        Updates bits.json with hashes and bit_vectors.json with vectors.
        """
        # Read input files
        bits = read_bits_file(self.bits_file_path)
        transcript_data = self.read_transcript_file()
        vectors = self.read_vectors_file()
        
        # Track if we processed any bits
        processed_any = False
        
        # Process each bit
        for bit in bits:
            # Skip if bit already has term vector hash and we're not regenerating
            if not should_process_bit(bit, 'term_vector_hash', self.regenerate):
                logger.info(f"Skipping bit '{bit['title']}' - already has term vector")
                continue
                
            logger.info(f"Processing bit: {bit['title']}")
            processed_any = True
            
            # Get bit text from transcript
            bit_items = select_bit(transcript_data, bit['start'], bit['end'])
            bit_text = flatten_bit(bit_items)
            
            # Calculate term vector and hash
            term_vector, vector_hash = self.calculate_term_vector(bit_text)
            logger.info(f"Calculated term vector (hash: {vector_hash})")
            
            # Update bit with hash
            bit['term_vector_hash'] = vector_hash
            
            # Store vector in vectors dictionary
            vectors[vector_hash] = term_vector
            
            # Write updated bits and vectors back to files
            write_bits_file(self.bits_file_path, bits)
            self.write_vectors_file(vectors)
            logger.info(f"Saved progress after processing bit: {bit['title']}")
            
        if not processed_any:
            logger.info("No bits needed processing")
        else:
            logger.info("Successfully processed all required bits")
            logger.info(f"Term vectors saved to: {self.vectors_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate term vectors for comedy bits.')
    parser.add_argument('-b', '--bits-file', type=str, required=True,
                      help='Path to bits JSON file')
    parser.add_argument('-t', '--transcript-file', type=str, required=True,
                      help='Path to transcript JSON file')
    parser.add_argument('-r', '--regenerate', action='store_true',
                      help='Force regeneration of all term vectors')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                      format='%(message)s')
    
    tool = TermVectorTool(
        bits_file_path=args.bits_file,
        transcript_file_path=args.transcript_file,
        regenerate=args.regenerate
    )
    tool.run()
