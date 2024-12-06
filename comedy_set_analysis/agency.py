from agency_swarm import Agency
from audio_transcription_agent.audio_transcription_agent import AudioTranscriptionAgent
# from bit_deliminator_agent.bit_deliminator_agent import BitDeliminatorAgent
# from joke_type_classifier_agent.joke_type_classifier_agent import JokeTypeClassifierAgent

audio_transcription_agent = AudioTranscriptionAgent()
# bit_deliminator_agent = BitDeliminatorAgent()
# joke_type_classifier_agent = JokeTypeClassifierAgent()

agency = Agency(
    [
        audio_transcription_agent,
        # [audio_transcription_agent, bit_deliminator_agent],
        # [bit_deliminator_agent, joke_type_classifier_agent]
    ],
    shared_instructions='agency_manifesto.md',
    temperature=0.5,
    max_prompt_tokens=25000
)

if __name__ == "__main__":
    agency.run_demo() 