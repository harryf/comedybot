from agency_swarm import Agent
from comedy_set_analysis.audio_transcription_agent.tools.audio_transcription_tool import AudioTranscriptionTool

class AudioTranscriptionAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Audio Transcription Agent",
            description="Transcribes audio files using OpenAI's Whisper API.",
            instructions="./instructions.md",
            tools=[AudioTranscriptionTool],
            temperature=0.5,
            max_prompt_tokens=25000,
        ) 