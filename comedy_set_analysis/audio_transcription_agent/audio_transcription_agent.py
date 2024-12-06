# import asyncio, sys
from agency_swarm import Agent
from audio_transcription_agent.tools.audio_preparation_tool import AudioPreparationTool
from audio_transcription_agent.tools.audio_transcription_tool import AudioTranscriptionTool
from audio_transcription_agent.tools.sound_detection_tool import SoundDetectionTool
from audio_transcription_agent.tools.transcript_analyser_tool import TranscriptAnalyserTool
from audio_transcription_agent.tools.sound_analyser_tool import SoundAnalyserTool
# from audio_transcription_agent.tools.transcript_sounds_merge_tool import TranscriptSoundsMergeTool

class AudioTranscriptionAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Audio Transcription Agent",
            description="Transcribes audio files using OpenAI's Whisper API.",
            instructions="./instructions.md",
            tools=[
                AudioPreparationTool,
                AudioTranscriptionTool,
                SoundDetectionTool,
                TranscriptAnalyserTool,
                SoundAnalyserTool
                # TranscriptSoundsMergeTool
            ],
            temperature=0.5,
            max_prompt_tokens=25000,
        )