from tools.audio_file_finder_tool import AudioFileFinderTool
from tools.audio_preparation_tool import AudioPreparationTool
from tools.audio_transcription_tool import AudioTranscriptionTool
from tools.sound_detection_tool import SoundDetectionTool
from tools.transcript_analyser_tool import TranscriptAnalyserTool
from tools.sound_analyser_tool import SoundAnalyserTool
from tools.metadata_capture_tool import MetadataCaptureTool
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

def process_audio_directory(input_directory_path, output_directory_path):
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
        
        # Output or save the cleaned data as needed
        logger.info(f"Processed {audio_file}:")
        logger.info(f"Cleaned Transcript: {cleaned_transcript}")
        logger.info(f"Cleaned Sounds: {cleaned_sounds}")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Process audio files in a directory and output the cleaned transcript and sound data.")
    parser.add_argument("-i", "--inputdir", type=str, required=True, help="The path to the directory containing audio files.")
    parser.add_argument("-o", "--outputdir", type=str, required=True, help="The path to the directory where the processed data will be saved.")
    args = parser.parse_args()

    process_audio_directory(args.inputdir, args.outputdir)
