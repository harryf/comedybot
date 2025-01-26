from tools.bit_deliminator_tool import BitDeliminatorTool
from tools.joke_type_tool import JokeTypeTool
from tools.theme_identifier_tool import ThemeIdentifierTool
from tools.laughs_per_minute_tool import LaughsPerMinuteTool
from tools.bit_comparison_tool import BitComparisonTool
import argparse
import logging
import os
import sys
import time

logger = logging.getLogger()

def setup_logging():
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler for STDOUT (INFO and below)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)  # Filter out WARNING and above
    stdout_handler.setFormatter(formatter)

    # Handler for STDERR (WARNING and above)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)  # Only WARNING and above
    stderr_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

def process(input_directory_path):
    """
    Process a comedy set by running analysis tools in sequence:
    1. Bit Deliminator - identifies distinct bits in the set
    2. Joke Type - classifies the type of jokes in each bit
    3. Theme Identifier - identifies themes in each bit
    4. Laughs Per Minute - calculates audience reaction metrics
    5. Bit Comparison - finds similar bits and patterns
    
    Args:
        input_directory_path: Directory containing the required files:
            - transcript_clean.json: Cleaned transcript
            - sounds_clean.json: Cleaned audience reactions
            - bits.json: Will be created/updated by the tools
    """
    logger.info(f"Processing comedy set in directory: {input_directory_path}")
    
    # Define file paths
    transcript_path = os.path.join(input_directory_path, "transcript_clean.json")
    reactions_path = os.path.join(input_directory_path, "sounds_clean.json")
    bits_path = os.path.join(input_directory_path, "bits.json")
    
    # 1. Run Bit Deliminator Tool
    logger.info("Running Bit Deliminator Tool...")
    bit_tool = BitDeliminatorTool(transcript_file_path=transcript_path, bits_file_path=bits_path)
    bit_tool.run()
    
    # 2. Run Joke Type Tool
    logger.info("Running Joke Type Tool...")
    joke_tool = JokeTypeTool(bits_file_path=bits_path, transcript_file_path=transcript_path)
    joke_tool.run()
    
    # 3. Run Theme Identifier Tool
    logger.info("Running Theme Identifier Tool...")
    theme_tool = ThemeIdentifierTool(bits_file_path=bits_path, transcript_file_path=transcript_path)
    theme_tool.run()
    
    # 4. Run Laughs Per Minute Tool
    logger.info("Running Laughs Per Minute Tool...")
    lpm_tool = LaughsPerMinuteTool(bits_file_path=bits_path, reactions_file_path=reactions_path)
    lpm_tool.run()
    
    # 5. Run Bit Comparison Tool
    logger.info("Running Bit Comparison Tool...")
    comparison_tool = BitComparisonTool.initialize(directory=input_directory_path)
    comparison_tool.run()
    
    logger.info("Finished processing comedy set")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Analyse comedy set transcripts and audience reactions and identify and classify bits.")
    parser.add_argument("-i", "--inputdir", type=str, required=True, help="The path to the directory containing the transcript data.")
    args = parser.parse_args()

    process(args.inputdir)