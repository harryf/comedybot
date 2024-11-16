from agency_swarm.tools import BaseTool
from pydantic import Field
import os
import whisper
from dotenv import load_dotenv
import json

load_dotenv()

class AudioTranscriptionTool(BaseTool):
    """
    Transcribes audio files using OpenAI's Whisper model.
    """
    audio_file_path: str = Field(..., description="Path to the input audio file (m4a format).")

    def run(self):
        model = whisper.load_model("base")
        result = model.transcribe(self.audio_file_path)
        
        output_file_path = f"{self.audio_file_path.rsplit('.', 1)[0]}_transcription.json"
        with open(output_file_path, "w") as output_file:
            json.dump(result, output_file, indent=4)
        
        return f"Transcription saved to {output_file_path}"

if __name__ == "__main__":
    tool = AudioTranscriptionTool(audio_file_path="example.m4a")
    print(tool.run()) 