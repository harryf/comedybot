from base_tool import BaseTool
from pydantic import Field
import os, argparse, json, sys
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

import logging
import time
import re
logger = logging.getLogger(__name__)
import jsonschema

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

def extract_wait_time(error_message):
    """Extract wait time from OpenAI rate limit error message"""
    match = re.search(r'Please try again in (\d+\.?\d*)s', str(error_message))
    if match:
        return float(match.group(1))
    return 10.0  # Default wait time if we can't parse the message

def generate_default_bits_path(transcript_path: str) -> str:
    return os.path.join(os.path.dirname(transcript_path), "bits.json")

class BitDeliminatorTool(BaseTool):
    """
    Identifies and extracts bits from a transcription JSON file supported by the audience reactions JSON file
    using an OpenAI assistant.
    """
    transcript_file_path: str = Field(..., description="Path to the transcription JSON file.")
    bits_file_path: str = Field(default=None, description="Path to output the bits JSON file to.")

    def __init__(self, **data):
        super().__init__(**data)
        if self.bits_file_path is None:
            self.bits_file_path = generate_default_bits_path(self.transcript_file_path)

    def run(self):
        MAX_RETRIES = 3
        BASE_RETRY_DELAY = 5  # Base delay between retries in seconds
        openai_api_key = os.getenv("OPENAI_API_KEY")
        assistant_id = os.getenv("BIT_FINDER_ASSISTANT_ID")
        client = OpenAI(api_key=openai_api_key)

        with open(self.transcript_file_path, "r") as f:
            transcript_data = json.load(f)
            
        # Remove 'type' and 'seek' properties from each element
        cleaned_transcript = [
            {k: v for k, v in item.items() if k not in ['type', 'seek']}
            for item in transcript_data
        ]
        
        transcript = json.dumps(cleaned_transcript, indent=2)

        thread = client.beta.threads.create()
        
        for attempt in range(MAX_RETRIES):
            try:
                if attempt == 0:
                    # First attempt - send original transcript
                    message = client.beta.threads.messages.create(
                        thread_id=thread.id,
                        content=transcript,
                        role="user"
                    )
                else:
                    # Retry attempt - send error message
                    message = client.beta.threads.messages.create(
                        thread_id=thread.id,
                        content=f"Please try again. The previous response had the following issue: {error_message}",
                        role="user"
                    )

                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant_id,
                )
                
                while True:
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                    if run.status == "completed":
                        break
                    elif run.status == "failed":
                        logger.error(f"Run failure reason: {run.last_error}")
                        raise Exception(f"Failed to generate text due to: {run.last_error}")
                    logger.info(f"Run status: {run.status}")
                
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id,
                    order="desc",
                    limit=1,
                )
                
                assistant_res = next(
                    (
                        content.text.value
                        for content in messages.data[0].content
                        if content.type == "text"
                    ),
                    None,
                )
                
                logger.info(f"Assistant response received, validating...")
                
                # Parse the response as JSON
                bits_data = json.loads(assistant_res)
                
                # Run all validations
                total_duration = transcript_data[-1]["end"] - transcript_data[0]["start"]
                
                validations = [
                    validate_bits_schema(bits_data),
                    check_minimum_bits(bits_data, total_duration),
                    check_time_gaps(bits_data),
                    check_bit_boundaries(bits_data, transcript_data)
                ]
                
                # Check all validations and collect errors
                validation_errors = []
                for is_valid, error in validations:
                    if not is_valid:
                        validation_errors.append(error)
                
                if validation_errors:
                    error_message = "\n".join([
                        f"- {error}" for error in validation_errors
                    ])
                    logger.warning(f"Validation failed (attempt {attempt + 1}/{MAX_RETRIES}). All errors:\n{error_message}")
                    raise ValueError(error_message)
                
                # If we get here, all validations passed
                logger.info("All validations passed, writing output file")
                with open(self.bits_file_path, 'w') as f:
                    json.dump(bits_data, f, indent=2)
                
                logger.info(f"Successfully wrote bits to {self.bits_file_path}")
                return  # Success!
                
            except (ValueError, json.JSONDecodeError) as e:
                error_message = str(e)
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed after {MAX_RETRIES} attempts. Last error: {error_message}")
                    raise Exception(f"Failed to generate valid bits after {MAX_RETRIES} attempts. Last error: {error_message}")
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed. Retrying...")
                continue

            except Exception as e:
                error_message = str(e)
                if "rate_limit_exceeded" in error_message.lower():
                    wait_time = extract_wait_time(error_message)
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                raise
            
            # Add delay between retries to avoid rate limits
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Waiting {BASE_RETRY_DELAY} seconds before next attempt...")
                time.sleep(BASE_RETRY_DELAY)

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