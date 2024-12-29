from base_tool import BaseTool
from pydantic import Field
import json
import os

class BitDeliminatorTool(BaseTool):
    """
    Identifies and extracts bits from a transcription JSON file supported by the audience reactions JSON file.
    """
    transcription_file_path: str = Field(..., description="Path to the transcription JSON file.")
    audience_reactions_file_path: str = Field(..., description="Path to the audience reactions JSON file.")

    def run(self):
        pass

if __name__ == "__main__":
    tool = BitDeliminatorTool(transcription_file_path="example_transcription.json")
    print(tool.run()) 