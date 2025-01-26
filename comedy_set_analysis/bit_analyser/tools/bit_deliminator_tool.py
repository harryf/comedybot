import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import SimpleBaseTool
from pydantic import Field
import argparse, json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import logging
import time
import re
logger = logging.getLogger(__name__)
import jsonschema
from typing import Any

BITS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"}
                },
                "required": ["title", "start", "end"]
            }
        }
    },
    "required": ["items"]
}

def validate_bits_schema(bits_data):
    """Validate the bits data against the schema"""
    try:
        jsonschema.validate(instance=bits_data, schema=BITS_SCHEMA)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)

def check_minimum_bits(bits_data, total_duration):
    """Check if there are enough bits based on total duration"""
    min_bits = int(total_duration / 180)  # 3 minutes = 180 seconds
    if len(bits_data.get("items", [])) < min_bits:
        return False, f"Expected at least {min_bits} bits for a {total_duration:.1f} second transcript, but only found {len(bits_data.get('items', []))}"
    return True, None

def check_time_gaps(bits_data):
    """Check for gaps larger than 30 seconds between bits"""
    MAX_GAP = 30  # seconds
    gaps = []
    
    items = bits_data.get("items", [])

    if not items:
        return False, "No bits found in response"
    
    for i in range(len(items) - 1):
        current_bit = items[i]
        next_bit = items[i + 1]
        gap = next_bit["start"] - current_bit["end"]
        
        if gap > MAX_GAP:
            gaps.append((i, i + 1, gap))
    
    if gaps:
        gap_descriptions = [f"Gap of {gap:.1f} seconds between bits {i} and {i+1}" for i, i_next, gap in gaps]
        return False, "Large gaps found: " + "; ".join(gap_descriptions)
    return True, None

def check_bit_boundaries(bits_data, transcript_data):
    """Check if bit start and end times match transcript data points"""
    transcript_times = set((item["start"], item["end"]) for item in transcript_data)
    invalid_boundaries = []
    
    for i, bit in enumerate(bits_data.get("items", [])):
        start_found = any(abs(bit["start"] - t[0]) < 0.1 or abs(bit["start"] - t[1]) < 0.1 for t in transcript_times)
        end_found = any(abs(bit["end"] - t[0]) < 0.1 or abs(bit["end"] - t[1]) < 0.1 for t in transcript_times)
        
        if not start_found or not end_found:
            invalid_boundaries.append(i)
    
    if invalid_boundaries:
        bits = bits_data.get("items", [])
        error_details = [f"Bit {i} ({bits[i]['title']}) has invalid boundaries: start={bits[i]['start']}, end={bits[i]['end']}" 
                        for i in invalid_boundaries]
        return False, "Found bits with boundaries that don't match transcript data:\n" + "\n".join(error_details)
    return True, None

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text using a simple heuristic.
    This is a rough estimate - OpenAI uses tiktoken for exact counts.
    """
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4

def split_transcript_into_chunks(transcript_data: list, max_tokens: int = 25000) -> list:
    """
    Split transcript data into chunks that don't exceed the token limit.
    Ensures splits occur at transcript item boundaries to preserve timing information.
    
    Args:
        transcript_data: List of transcript items
        max_tokens: Maximum tokens per chunk (default 25000 to leave room for prompt)
        
    Returns:
        List of transcript chunks, where each chunk is a list of transcript items
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for item in transcript_data:
        # Estimate tokens in this item
        item_tokens = estimate_tokens(item["text"])
        
        # If adding this item would exceed the limit, start a new chunk
        if current_tokens + item_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(item)
        current_tokens += item_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def merge_bits_results(chunks_results: list) -> dict:
    """
    Merge bits results from multiple chunks into a single result.
    Ensures bits at chunk boundaries are properly handled.
    
    Args:
        chunks_results: List of bits results from each chunk
        
    Returns:
        Merged bits results
    """
    all_bits = []
    
    # Collect all bits from all chunks
    for chunk_result in chunks_results:
        if "items" in chunk_result:
            all_bits.extend(chunk_result["items"])
    
    # Sort bits by start time
    all_bits.sort(key=lambda x: x["start"])
    
    # Merge bits that are very close together (likely split across chunks)
    MERGE_THRESHOLD = 5  # seconds
    merged_bits = []
    
    for bit in all_bits:
        if not merged_bits or (bit["start"] - merged_bits[-1]["end"]) > MERGE_THRESHOLD:
            merged_bits.append(bit)
        else:
            # Merge with previous bit if they're close together
            merged_bits[-1]["end"] = bit["end"]
            merged_bits[-1]["title"] = f"{merged_bits[-1]['title']} (continued)"
    
    return {"items": merged_bits}

def extract_wait_time(error_message):
    """Extract wait time from OpenAI rate limit error message"""
    match = re.search(r'Please try again in (\d+\.?\d*)s', str(error_message))
    if match:
        return float(match.group(1))
    return 10.0  # Default wait time if we can't parse the message

def generate_default_bits_path(transcript_path: str) -> str:
    return os.path.join(os.path.dirname(transcript_path), "bits.json")

class BitDeliminatorTool(SimpleBaseTool):
    """
    Identifies and extracts bits from a transcription JSON file supported by the audience reactions JSON file
    using an OpenAI assistant.
    """
    transcript_file_path: str = Field(..., description="Path to the transcription JSON file.")
    bits_file_path: str = Field(default=None, description="Path to output the bits JSON file to.")
    model: str = Field(default="gpt-4-1106-preview", description="OpenAI model to use")
    temperature: float = Field(default=0.7, description="Temperature for OpenAI completion")
    max_retries: int = Field(default=3, description="Maximum number of retries for OpenAI API calls")
    client: Any = Field(default=None, description="OpenAI client instance")
    system_prompt: str = Field(default="""You are a comedy expert analyzing a transcript of a stand-up comedy performance.
Your task is to identify distinct comedy bits within the performance.

A comedy bit is a self-contained segment of the performance focused on a specific topic, story, or theme.
Bits typically last between 1-5 minutes, though they can be shorter or longer.

For each bit you identify, provide:
1. A brief, descriptive title that captures the main topic or theme
2. The start time (in seconds)
3. The end time (in seconds)

Format your response as a JSON object with an 'items' array containing the bits in chronological order.
Each bit should have 'title', 'start', and 'end' fields.

Example response format:
{
    "items": [
        {
            "title": "Dating Apps",
            "start": 0.0,
            "end": 120.5
        },
        {
            "title": "Living with Roommates",
            "start": 120.5,
            "end": 300.0
        }
    ]
}

Important guidelines:
1. Ensure bit boundaries align with natural transitions in the performance
2. Avoid gaps between bits unless there's a clear break or transition
3. Make titles concise but descriptive enough to understand the bit's content
4. Use the exact timestamps from the transcript""")

    def __init__(self, **data):
        super().__init__(**data)
        if self.bits_file_path is None:
            self.bits_file_path = generate_default_bits_path(self.transcript_file_path)
        self.client = OpenAI()

    def _get_openai_response(self, messages: list, retry_count: int = 0) -> str:
        """
        Get response from OpenAI API with retry logic for rate limits.
        
        Args:
            messages: List of message dictionaries for the conversation
            retry_count: Current retry attempt number
            
        Returns:
            Response content from OpenAI
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={ "type": "json_object" }  # Force JSON response
            )
            content = response.choices[0].message.content
            logger.debug(f"OpenAI response: {content}")
            
            # Validate it's proper JSON before returning
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError as e:
                logger.error(f"OpenAI returned invalid JSON: {content}")
                raise ValueError(f"OpenAI returned invalid JSON: {str(e)}")
            
        except Exception as e:
            if "rate limit" in str(e).lower() and retry_count < self.max_retries:
                wait_time = extract_wait_time(str(e))
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry {retry_count + 1}/{self.max_retries}")
                time.sleep(wait_time)
                return self._get_openai_response(messages, retry_count + 1)
            raise

    def process_transcript_chunk(self, chunk: list, chunk_number: int = None, total_chunks: int = None) -> dict:
        """
        Process a single transcript chunk with OpenAI.
        
        Args:
            chunk: List of transcript items to process
            chunk_number: Optional chunk number for logging
            total_chunks: Optional total number of chunks for logging
            
        Returns:
            Processed bits result for this chunk
        """
        # Prepare the chunk text
        chunk_text = "\n".join(f"{item['start']:.1f}-{item['end']:.1f}: {item['text']}" for item in chunk)
        
        # Log progress if processing multiple chunks
        if chunk_number is not None:
            logger.info(f"Processing chunk {chunk_number}/{total_chunks}")
        
        # Get the time range for this chunk
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        
        # Create the prompt for this chunk
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Here's a transcript segment from {chunk_start:.1f}s to {chunk_end:.1f}s. Each line shows the timestamp range and text:\n\n{chunk_text}"}
        ]
        
        # Get OpenAI response for this chunk
        response = self._get_openai_response(messages)
        chunk_result = json.loads(response)
        
        # Validate chunk result
        is_valid, error = validate_bits_schema(chunk_result)
        if not is_valid:
            raise ValueError(f"Chunk validation failed: {error}")
        
        return chunk_result

    def run(self) -> dict:
        """
        Run the bit deliminator tool on the transcript file.
        Handles large transcripts by splitting them into chunks only if necessary.
        """
        # Read transcript data
        with open(self.transcript_file_path, 'r') as f:
            transcript_data = json.load(f)
        
        # Calculate total duration for validation
        total_duration = max(item["end"] for item in transcript_data)
        
        # Check if we need to split the transcript
        full_text = "\n".join(item["text"] for item in transcript_data)
        estimated_tokens = estimate_tokens(full_text)
        
        try:
            if estimated_tokens <= 25000:  # Process as a single chunk if small enough
                result = self.process_transcript_chunk(transcript_data)
            else:
                # Split and process in chunks
                transcript_chunks = split_transcript_into_chunks(transcript_data)
                logger.info(f"Transcript size ({estimated_tokens} estimated tokens) exceeds limit. Processing in {len(transcript_chunks)} chunks.")
                
                chunks_results = []
                for i, chunk in enumerate(transcript_chunks, 1):
                    try:
                        chunk_result = self.process_transcript_chunk(chunk, i, len(transcript_chunks))
                        chunks_results.append(chunk_result)
                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {e}")
                        continue
                
                if not chunks_results:
                    raise ValueError("All chunks failed to process")
                
                # Merge results from all chunks
                result = merge_bits_results(chunks_results)
            
            # Validate the final results
            is_valid, error = validate_bits_schema(result)
            if not is_valid:
                raise ValueError(f"Results validation failed: {error}")
            
            # Additional validations
            is_valid, error = check_minimum_bits(result, total_duration)
            if not is_valid:
                logger.warning(f"Validation warning: {error}")
            
            is_valid, error = check_time_gaps(result)
            if not is_valid:
                logger.warning(f"Validation warning: {error}")
            
            is_valid, error = check_bit_boundaries(result, transcript_data)
            if not is_valid:
                logger.warning(f"Validation warning: {error}")
            
            # Write the results
            with open(self.bits_file_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process comedy transcript files and identify bits.')
    parser.add_argument('-t', '--transcript-file', type=str, required=True,
                      help='Path to the input transcript JSON file')
    parser.add_argument('-b', '--bits-file', type=str, required=False,
                      help='Path to output the bits JSON file to. If not provided, a default path will be generated which is <transcript_file_path>/bits.json.')
    args = parser.parse_args()
    
    tool_args = {
        'transcript_file_path': args.transcript_file,
    }
    if args.bits_file:
        tool_args['bits_file_path'] = args.bits_file

    logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
    
    tool = BitDeliminatorTool(**tool_args)
    print(tool.run())