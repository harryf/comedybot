from tools.audio_file_finder_tool import AudioFileFinderTool
from tools.audio_preparation_tool import AudioPreparationTool
from tools.metadata_capture_tool import MetadataCaptureTool
from tools.sound_detection_tool import SoundDetectionTool
from tools.sound_analyser_tool import SoundAnalyserTool
from tools.metadata_update_tool import MetadataUpdateTool
from tools.audio_segment_tool import AudioSegmentTool
from tools.audio_postprocessing_tool import AudioPostprocessingTool

import argparse
import logging, sys
import time
import shutil

logger = logging.getLogger()

def is_installed(name: str) -> bool:
    """Check if a command is available in the system PATH."""
    return shutil.which(name) is not None

# Determine which transcription and analyzer tools to use based on whisper-cpp availability
if is_installed("whisper-cpp"):
    logger.debug("whisper-cpp is available, using version 2 of transcription tools")
    from tools.audio_transcription_tool_2 import AudioTranscriptionTool2 as AudioTranscriptionTool
    from tools.transcript_analyser_tool_2 import TranscriptAnalyserTool2 as TranscriptAnalyserTool
else:
    logger.debug("whisper-cpp not found, using default transcription tools")
    from tools.audio_transcription_tool import AudioTranscriptionTool
    from tools.transcript_analyser_tool import TranscriptAnalyserTool

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
    # Step 1: Find all audio files in the directory
    logger.debug(f"Searching for new files in {input_directory_path}")
    audio_files = AudioFileFinderTool(directory_path=input_directory_path).run()
    if len(audio_files) == 0:
        logger.info(f"No new audio files found in {input_directory_path}")
        return
    
    # Log the found audio files
    logger.debug(f"Found audio files: {audio_files}")
    
    for audio_file in audio_files:
        # Log the current audio file being processed
        logger.debug(f"Processing audio file: {audio_file}")
        
        # Step 2a: Prepare the audio file
        audio_preparation_tool = AudioPreparationTool(
            audio_file_path=audio_file,
            output_directory_path=output_directory_path
        )
        new_location = audio_preparation_tool.run()
        if not new_location:
            logger.info(f"Skipping {audio_file} as it has already been prepared.")
            continue
        
        new_subdirectory_path = new_location[0]
        new_audio_file_path = new_location[1]

        if not MetadataCaptureTool(
            audio_file_path=new_audio_file_path, 
            output_directory=new_subdirectory_path
        ).run():
            logger.error(f"Failed to capture metadata for {new_audio_file_path}")
            audio_preparation_tool.rollback(new_subdirectory_path)
            return
        
        # Step 2b: Transcribe the audio
        transcript_file_path = AudioTranscriptionTool(audio_file_path=new_audio_file_path, output_directory=new_subdirectory_path).run()
        
        # Step 2c: Detect sounds in the audio
        sounds_file_path = SoundDetectionTool(audio_file_path=new_audio_file_path, output_directory=new_subdirectory_path).run()
        
        # Step 2d: Clean the raw transcript file
        cleaned_transcript = TranscriptAnalyserTool(input_file=transcript_file_path, output_directory=new_subdirectory_path).run()
        
        # Step 2e: Clean the raw sounds file
        cleaned_sounds = SoundAnalyserTool(input_file=sounds_file_path, output_directory=new_subdirectory_path).run()
        
        # Step 2f: Update the metadata file
        MetadataUpdateTool(directory_path=new_subdirectory_path).run()

        # Step 2g: Split the audio file into segments
        AudioSegmentTool(input_directory=new_subdirectory_path).run()
        
        # Step 2h: Rename the directory based on the date_of_show in metadata
        new_directory_path = AudioPostprocessingTool(directory_path=new_subdirectory_path).run()
        if new_directory_path:
            logger.info(f"Directory renamed to: {new_directory_path}")
        else:
            logger.warning(f"Failed to rename directory: {new_subdirectory_path}")
        
        # Output or save the cleaned data as needed
        logger.info(f"Finished processing {audio_file}")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Process audio files in a directory and output the cleaned transcript and sound data.")
    parser.add_argument("-i", "--inputdir", type=str, required=True, help="The path to the directory containing audio files.")
    parser.add_argument("-o", "--outputdir", type=str, required=True, help="The path to the directory where the processed data will be saved.")
    args = parser.parse_args()

    process(args.inputdir, args.outputdir)
