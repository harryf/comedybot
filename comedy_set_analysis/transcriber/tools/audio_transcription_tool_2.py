#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import tempfile
import hashlib
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_tool import BaseTool
from pydantic import Field

# Configure logging
logger = logging.getLogger(__name__)

class AudioTranscriptionTool2(BaseTool):
    """
    Transcribes audio files using whisper-cpp CLI.
    This is a replacement for AudioTranscriptionTool that uses whisper-cpp instead of the Python whisper library.
    """
    audio_file_path: str = Field(..., description="Path to the input audio file.")
    output_directory: str = Field(..., description="Directory where the transcription output will be saved.")
    
    def run(self):
        """
        Transcribes the audio file and saves the result to a JSON file.

        Returns the path to the transcription JSON file.
        """
        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Define output file path
        output_basename = "transcript_raw_v2"
        output_file_path = os.path.join(self.output_directory, f"{output_basename}.json")
        
        # Generate a hash for the temporary WAV file (similar to the bash script)
        hash_value = hashlib.md5(output_basename.encode()).hexdigest()[:5]
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(prefix=f"transcribe_{hash_value}_", suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        
        try:
            # Convert to 16 kHz mono WAV using ffmpeg
            logger.info(f"Converting '{self.audio_file_path}' to 16 kHz mono WAV '{tmp_wav_path}'")
            ffmpeg_cmd = [
                "ffmpeg", 
                "-loglevel", "error", 
                "-y", 
                "-i", self.audio_file_path, 
                "-ar", "16000", 
                "-ac", "1", 
                tmp_wav_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            
            # Get the number of physical CPUs
            try:
                cpu_count = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
            except (subprocess.SubprocessError, ValueError):
                cpu_count = os.cpu_count() or 4  # Fallback to logical CPUs or 4 if that fails
            
            # Get the model path from environment or use default
            model_path = os.environ.get("WHISPER_MODEL_PATH", os.path.expanduser("~/.whisper/ggml-large-v2.bin"))
            
            # Get resources path from brew
            try:
                brew_prefix = subprocess.check_output(["brew", "--prefix", "whisper-cpp"]).decode().strip()
                resources_path = os.path.join(brew_prefix, "share/whisper-cpp")
            except subprocess.SubprocessError:
                resources_path = ""
            
            # Set environment variables for whisper-cli
            env = os.environ.copy()
            env["GGML_METAL_PATH_RESOURCES"] = resources_path
            env["GGML_METAL_NOSYNC"] = "1"
            
            # Run whisper-cli
            logger.info(f"Running whisper-cli, this may take a while...")
            output_base_path = os.path.join(self.output_directory, output_basename)
            whisper_cmd = [
                "whisper-cli",
                "-t", str(cpu_count),
                "--flash-attn",
                "--language", "en",
                "--model", model_path,
                "--file", tmp_wav_path,
                "--output-file", output_base_path,
                "--output-json",
                "--split-on-word",
                "--no-prints",
                "--max-len", "80"
            ]
            # Redirect stdout and stderr to /dev/null to prevent terminal output
            with open(os.devnull, 'w') as devnull:
                subprocess.run(whisper_cmd, check=True, env=env, stdout=devnull, stderr=devnull)
            
            logger.info(f"Transcribed {self.audio_file_path} and saved to {output_file_path}")
            return output_file_path
            
        finally:
            # Clean up the temporary WAV file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio files using whisper-cpp')
    parser.add_argument('--input', '-i', required=True, help='Path to the input audio file')
    parser.add_argument('--output', '-o', required=True, help='Directory where the transcription output will be saved')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    tool = AudioTranscriptionTool2(
        audio_file_path=args.input,
        output_directory=args.output
    )
    print(f"Transcription saved to: {tool.run()}")
