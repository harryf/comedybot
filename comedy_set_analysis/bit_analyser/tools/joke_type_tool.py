#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import Field
from base_tool import BaseTool
from bit_utils import (
    read_bits_file, write_bits_file, select_bit,
    flatten_bit, should_process_bit
)

load_dotenv(find_dotenv(), override=True)
logger = logging.getLogger(__name__)

class JokeTypeValidator:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'joke_types.json'), 'r') as f:
            self.valid_types = set(json.load(f)['valid_joke_types'])
    
    def validate_response(self, response_data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate the response from the OpenAI assistant.
        
        Args:
            response_data: Dictionary containing the response data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if joke_types key exists
        if 'joke_types' not in response_data:
            return False, "Response missing 'joke_types' key"
            
        joke_types = response_data['joke_types']
        
        # Check if joke_types is a list
        if not isinstance(joke_types, list):
            return False, "'joke_types' must be a list"
            
        # Check if list is non-empty
        if not joke_types:
            return False, "No joke types provided"
            
        # Check if all items are strings and valid joke types
        invalid_types = []
        for joke_type in joke_types:
            if not isinstance(joke_type, str):
                return False, f"Invalid joke type format: {joke_type}"
            if joke_type not in self.valid_types:
                invalid_types.append(joke_type)
                
        if invalid_types:
            return False, f"Invalid joke types found: {', '.join(invalid_types)}"
            
        # Check for duplicates
        if len(joke_types) != len(set(joke_types)):
            return False, "Duplicate joke types found"
            
        return True, None

class JokeTypeTool(BaseTool):
    bits_file_path: str = Field(description="Path to the bits JSON file")
    transcript_file_path: Optional[str] = Field(None, description="Optional path to the transcript JSON file")
    validator: JokeTypeValidator = Field(default_factory=JokeTypeValidator, description="Validator for joke type responses")
    regenerate: bool = Field(default=False, description="Force regeneration of all joke types")
    
    def __init__(self, bits_file_path: str, transcript_file_path: str = None, regenerate: bool = False):
        super().__init__(bits_file_path=bits_file_path, transcript_file_path=transcript_file_path, regenerate=regenerate)

    def run(self):
        MAX_RETRIES = 3
        BASE_RETRY_DELAY = 5  # Base delay between retries in seconds
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        assistant_id = os.getenv("JOKE_TYPE_ASSISTANT_ID")
        client = OpenAI(api_key=openai_api_key)
        
        # Read bits file
        bits = read_bits_file(self.bits_file_path)
        
        # Read transcript file if provided
        transcript_data = None
        if self.transcript_file_path:
            with open(self.transcript_file_path, "r") as f:
                transcript_data = json.load(f)
        
        # Track if any bits were processed
        processed_any = False
        
        # Process each bit
        for bit in bits:
            # Skip if bit already has valid joke types and we're not regenerating
            if not should_process_bit(bit, 'joke_types', self.regenerate):
                logger.info(f"Skipping bit '{bit['title']}' - already has joke types")
                continue
                
            logger.info(f"Processing bit: {bit['title']}")
            processed_any = True
            
            # Get bit text
            if transcript_data:
                bit_items = select_bit(transcript_data, bit['start'], bit['end'])
                bit_text = flatten_bit(bit_items)
            else:
                bit_text = f"Title: {bit['title']}\nStart: {bit['start']}\nEnd: {bit['end']}"
            
            # Try to get joke types from assistant
            for attempt in range(MAX_RETRIES):
                try:
                    # Create thread and message
                    thread = client.beta.threads.create()
                    message = client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=bit_text
                    )
                    
                    # Run assistant
                    run = client.beta.threads.runs.create(
                        thread_id=thread.id,
                        assistant_id=assistant_id
                    )
                    
                    # Wait for completion
                    while run.status in ["queued", "in_progress"]:
                        run = client.beta.threads.runs.retrieve(
                            thread_id=thread.id,
                            run_id=run.id
                        )
                        time.sleep(1)
                    
                    if run.status == "completed":
                        # Get the assistant's response
                        messages = client.beta.threads.messages.list(thread_id=thread.id)
                        response = messages.data[0].content[0].text.value
                        
                        # Parse response
                        try:
                            response_data = json.loads(response)
                        except json.JSONDecodeError:
                            raise ValueError("Assistant response was not valid JSON")
                        
                        # Validate response
                        is_valid, error = self.validator.validate_response(response_data)
                        if not is_valid:
                            raise ValueError(error)
                        
                        # Add joke types to bit data
                        bit['joke_types'] = response_data['joke_types']
                        logger.info(f"Found joke types: {response_data['joke_types']}")
                        
                        # Write updated bits back to file after each successful processing
                        write_bits_file(self.bits_file_path, bits)
                        logger.info(f"Saved progress after processing bit: {bit['title']}")
                        break
                        
                    elif run.status == "failed":
                        raise Exception(f"Run failed: {run.last_error}")
                    else:
                        raise Exception(f"Unexpected run status: {run.status}")
                        
                except Exception as e:
                    error_message = str(e)
                    if "rate_limit_exceeded" in error_message.lower():
                        wait_time = float(error_message.split("try again in ")[1].split("s")[0])
                        logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        continue
                        
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_message}")
                        logger.info(f"Waiting {BASE_RETRY_DELAY} seconds before next attempt...")
                        time.sleep(BASE_RETRY_DELAY)
                        continue
                    else:
                        logger.error(f"Failed to process bit after {MAX_RETRIES} attempts")
                        raise
        
        if not processed_any:
            logger.info("No bits needed processing")
        else:
            logger.info("Successfully processed all required bits")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze joke types in comedy bits.')
    parser.add_argument('-b', '--bits-file', type=str, required=True,
                      help='Path to bits JSON file')
    parser.add_argument('-t', '--transcript-file', type=str,
                      help='Path to transcript JSON file')
    parser.add_argument('-r', '--regenerate', action='store_true',
                      help='Force regeneration of all joke types')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                      format='%(message)s')
    
    tool = JokeTypeTool(
        bits_file_path=args.bits_file,
        transcript_file_path=args.transcript_file,
        regenerate=args.regenerate
    )
    print(tool.run())
