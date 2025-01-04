#!/usr/bin/env python3

"""
Advanced bit comparison tool that uses multi-level vector comparison.
Compares bits based on:
1. Full bit similarity
2. Matching sentences
3. Shared n-grams
4. Similar punchlines
"""

import os
import sys

# Add tools directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import argparse
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import colorama
from colorama import Fore, Style
from bit_vector_db import BitVectorDB
from term_vector_tool import TermVectorTool
from bit_vectors import BitVectors
from bit_entity import BitEntity

logger = logging.getLogger(__name__)

# Thresholds for match classification
HARD_MATCH_THRESHOLD = 0.7
SOFT_MATCH_THRESHOLD = 0.0
MIN_REPORT_THRESHOLD = 0.0

class TranscriptEntry(BaseModel):
    """Model for a single transcript entry."""
    type: str
    start: float
    end: float
    text: str
    seek: Optional[int] = None

class BitComparisonTool(BaseModel):
    """Tool for comparing comedy bits."""
    
    # Input paths
    directory: str = Field(description="Directory containing input files")
    bits_file: str = Field(description="Path to bits.json")
    metadata_file: str = Field(description="Path to metadata.json")
    transcript_file: str = Field(description="Path to transcript_clean.json")
    
    # Tool configuration
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    
    # Component instances
    vector_tool: Optional[TermVectorTool] = Field(default=None, description="Vector tool instance")
    bit_database: Optional[BitVectorDB] = Field(default=None, description="Bit database instance")
    transcript: Optional[List[Dict[str, Any]]] = Field(default=None, description="Transcript data")
    
    # Match tracking
    exact_matches: List[Dict[str, Any]] = Field(default_factory=list, description="List of exact matches")
    soft_matches: List[Dict[str, Any]] = Field(default_factory=list, description="List of soft matches")
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def initialize(cls, directory: str, similarity_threshold: float = 0.7) -> 'BitComparisonTool':
        """Initialize the comparison tool with proper file validation."""
        try:
            # Define required files and their descriptions
            required_files = {
                'bits.json': "Contains bit definitions and metadata",
                'metadata.json': "Contains show metadata and settings",
                'transcript_clean.json': "Contains processed transcript text"
            }
            
            # Get input files with validation
            input_files = {}
            missing_files = []
            permission_errors = []
            
            for filename, description in required_files.items():
                file_path = os.path.join(directory, filename)
                try:
                    if not os.path.exists(file_path):
                        missing_files.append((filename, description))
                    elif not os.access(file_path, os.R_OK):
                        permission_errors.append(filename)
                    else:
                        input_files[filename] = file_path
                except OSError as e:
                    logger.error(f"Error accessing {filename}: {e}")
                    raise
            
            # Handle missing files
            if missing_files:
                error_msg = "Required files not found:\n" + "\n".join(
                    f"- {name}: {desc}" for name, desc in missing_files
                )
                raise FileNotFoundError(error_msg)
            
            # Handle permission errors
            if permission_errors:
                error_msg = "Permission denied for files:\n" + "\n".join(
                    f"- {name}" for name in permission_errors
                )
                raise PermissionError(error_msg)
            
            # Load transcript with proper error handling
            try:
                with open(input_files['transcript_clean.json'], 'r') as f:
                    transcript = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in transcript file: {e}")
                raise
            except OSError as e:
                logger.error(f"Error reading transcript file: {e}")
                raise
            
            # Initialize vector tool with validated files
            try:
                vector_tool = TermVectorTool(
                    bits_file=input_files['bits.json'],
                    metadata_file=input_files['metadata.json'],
                    transcript_file=input_files['transcript_clean.json'],
                    regenerate=False
                )
            except (ModuleNotFoundError, OSError) as e:
                logger.error(f"Error initializing vector tool: {e}")
                raise

            # Initialize bit database with proper error handling
            try:
                bit_database = BitVectorDB(
                    dimension=384,
                    similarity_threshold=similarity_threshold
                )
            except (faiss.FaissException, RuntimeError) as e:
                logger.error(f"Error initializing bit database: {e}")
                raise
            
            # Create instance with validated components
            return cls(
                directory=directory,
                bits_file=input_files['bits.json'],
                metadata_file=input_files['metadata.json'],
                transcript_file=input_files['transcript_clean.json'],
                similarity_threshold=similarity_threshold,
                vector_tool=vector_tool,
                bit_database=bit_database,
                transcript=transcript
            )
            
        except Exception as e:
            logger.error(f"Error initializing comparison tool: {e}")
            raise
        
    def process_directory(self, directory_path: str) -> None:
        """Process all bits in a directory."""
        logger.info(f"Processing directory: {directory_path}")
        
        # Get the bits file path
        bits_file = os.path.join(directory_path, "bits.json")
        if not os.path.exists(bits_file):
            logger.error(f"No bits.json found in {directory_path}")
            return
        
        logger.info(f"Processing bits from: {bits_file}")
        with open(bits_file, 'r') as f:
            data = json.load(f)
            
        # Extract bits from schema
        if isinstance(data, dict):
            if 'items' in data:
                bits = data['items']
            else:
                logger.error("No 'items' field found in bits.json")
                return
        else:
            bits = data
            
        for bit in bits:
            if isinstance(bit, dict) and 'title' in bit:
                bit_entity = BitEntity(directory_path)
                bit_entity.load_from_set(bit)
                self._process_bit(bit_entity)
            
        logger.info("Finished processing bits")
        # Print summary table at the end
        self._print_summary_table()
    
    def _process_bit(self, bit_entity: BitEntity) -> None:
        """Process a single bit."""
        try:
            bit_id = bit_entity.bit_data['bit_id']
            bit_title = bit_entity.bit_data['bit_info']['title']

            logger.info(f"\nProcessing bit: {bit_title} ({bit_id})")

            if self.bit_database.has_bit(bit_id):
                logger.warning(f"Bit {bit_id} already exists in database")
                return
            
            # Process bit text into vectors
            vectors = self.vector_tool.process_bit(bit_entity.bit_data['transcript']['text'])
            if vectors is None:
                logger.warning(f"Failed to generate vectors for bit {bit_id}")
                return
            
            # Compare with existing bits
            try:
                matches = self.bit_database.find_matching_bits(vectors)
                if matches:
                    best_match = matches[0]
                    match_data = {
                        'original': bit_title,
                        'matched': best_match.title,
                        'score': best_match.overall_score
                    }
                    
                    if best_match.overall_score > HARD_MATCH_THRESHOLD:
                        logger.info(f"\nFound exact match for: '{bit_title}' ({bit_id}):")
                        logger.info(f"- Matched with: '{best_match.title}' ({best_match.bit_id})")
                        logger.info(f"- Score: {best_match.overall_score:.3f}")
                        self.exact_matches.append(match_data)
                    else:
                        # Track all other matches as soft matches
                        logger.info(f"\nFound potential match for '{bit_title}' ({bit_id}):")
                        logger.info(f"- Matched with: '{best_match.title}' ({best_match.bit_id})")
                        logger.info(f"- Score: {best_match.overall_score:.3f}")
                        self.soft_matches.append(match_data)
            except Exception as e:
                logger.error(f"Error finding matching bits for {bit_id}: {e}")
            
            # Add bit to database if no strong match found
            try:
                if (matches and len(matches) > 0 and matches[0].overall_score > HARD_MATCH_THRESHOLD):
                    self.bit_database.add_to_database(bit_entity, vectors, matches[0])
                else:
                    self.bit_database.add_to_database(bit_entity, vectors)
                logger.info(f"Added bit {bit_id} to database")
            except Exception as e:
                logger.error(f"Error adding bit {bit_id} to database: {e}")
                return
            
        except Exception as e:
            logger.error(f"Error processing bit {bit_id}: {e}")
            return

    def _print_summary_table(self) -> None:
        """Print a summary table of all matches."""
        if not (self.exact_matches or self.soft_matches):
            return

        print("\n" + "="*80)
        print(f"{Fore.CYAN}COMPARISON SUMMARY{Style.RESET_ALL}")
        print("="*80)
        
        if self.exact_matches:
            print(f"\n{Fore.GREEN}EXACT MATCHES (score > {HARD_MATCH_THRESHOLD}):{Style.RESET_ALL}")
            print("-"*80)
            print(f"{'Original Bit':<35} | {'Matched Bit':<35} | {'Score':<8}")
            print("-"*80)
            for match in sorted(self.exact_matches, key=lambda x: x['score'], reverse=True):
                print(f"{match['original']:<35} | {match['matched']:<35} | {match['score']:.3f}")
        
        if self.soft_matches:
            print(f"\n{Fore.YELLOW}POTENTIAL MATCHES (score > {MIN_REPORT_THRESHOLD}):{Style.RESET_ALL}")
            print("-"*80)
            print(f"{'Original Bit':<35} | {'Matched Bit':<35} | {'Score':<8}")
            print("-"*80)
            # Filter and sort soft matches
            filtered_matches = [m for m in self.soft_matches if m['score'] > MIN_REPORT_THRESHOLD]
            for match in sorted(filtered_matches, key=lambda x: x['score'], reverse=True):
                score_color = Fore.GREEN if match['score'] > 0.3 else Fore.YELLOW if match['score'] > 0.1 else Fore.RED
                print(f"{match['original']:<35} | {match['matched']:<35} | {score_color}{match['score']:.3f}{Style.RESET_ALL}")
        
        print("="*80 + "\n")
            
    def run(self) -> None:
        """Run the bit comparison tool."""
        self.process_directory(self.directory)
        logger.info("Finished processing bits")


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Comedy Bit Comparison Tool")
    parser.add_argument("-d", "--directory", required=True, help="Directory containing input files")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, help="Similarity threshold")
    
    args = parser.parse_args()
    
    # Initialize and run tool
    tool = BitComparisonTool.initialize(
        directory=args.directory,
        similarity_threshold=args.threshold
    )
    tool.run()
