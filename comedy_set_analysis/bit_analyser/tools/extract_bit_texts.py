#!/usr/bin/env python3

"""
Utility to extract flattened text from bits in a directory.
Creates a JSON file containing bit titles and their full text content.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BitTextExtractor:
    
    def __init__(self):
        self._transcript_cache = {}

    def extract_bit_text(self, bit_data: Dict[str, Any], transcript: List[Dict[str, Any]]) -> str:
        """Extract bit text from transcript with efficient time-based lookup."""
        try:
            # Get bit info from items array if needed
            if 'items' in bit_data:
                if not bit_data['items']:
                    logger.warning("Empty items array in bit data")
                    return ""
                bit_info = bit_data['items'][0]  # Get first item
            else:
                bit_info = bit_data  # Use as is
            
            # Extract timing info from bit info
            try:
                start_time = float(bit_info.get('start', 0))
                end_time = float(bit_info.get('end', 0))
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid time values in bit data: {e}")
                logger.debug(f"Bit info: {bit_info}")
                return ""
            
            if start_time >= end_time:
                logger.warning(f"Invalid time bounds for bit: {start_time} >= {end_time}")
                return ""
            
            # Check cache first
            cache_key = f"{start_time}_{end_time}"
            if cache_key in self._transcript_cache:
                return self._transcript_cache[cache_key]
            
            # Get relevant segments efficiently
            bit_segments = []
            for entry in transcript:
                if entry.get('type') != 'text':
                    continue
                    
                try:
                    seg_start = float(entry.get('start', 0))
                    seg_end = float(entry.get('end', 0))
                except (TypeError, ValueError):
                    continue
                
                # Skip segments outside our time range
                if seg_end < start_time or seg_start > end_time:
                    continue
                
                text = entry.get('text', '').strip()
                if text:
                    bit_segments.append((seg_start, text))
            
            # Sort segments by start time
            bit_segments.sort(key=lambda x: x[0])
            
            # Join segments
            bit_text = ' '.join(text for _, text in bit_segments).strip()
            
            # Cache result
            self._transcript_cache[cache_key] = bit_text
            
            if not bit_text:
                logger.warning(f"No transcript text found for bit in time range {start_time}-{end_time}")
                logger.debug(f"Bit title: {bit_info.get('title', 'Unknown')}")
                logger.debug(f"Found {len(bit_segments)} segments")
            else:
                logger.info(f"Found {len(bit_segments)} segments for bit: {bit_info.get('title', 'Unknown')}")
            
            return bit_text
            
        except Exception as e:
            logger.error(f"Error extracting bit text: {e}")
            logger.error(f"Bit data: {bit_data}")
            return ""


