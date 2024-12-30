from tools.bit_deliminator_tool import BitDeliminatorTool

import argparse
import logging, sys
import time

logger = logging.getLogger()

def setup_logging():
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.DEBUG)

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

def process(input_directory_path, output_directory_path):
    pass

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Analyse comedy set transcripts and audience reactions and identify and classify bits.")
    parser.add_argument("-i", "--inputdir", type=str, required=True, help="The path to the directory the transcript data is stored.")
    parser.add_argument("-o", "--outputdir", type=str, required=True, help="The path to the directory the analysis data will be saved.")
    args = parser.parse_args()

    process(args.inputdir, args.outputdir)