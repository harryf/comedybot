from agency_swarm import Agent
from comedy_set_analysis.bit_deliminator_agent.tools.bit_deliminator_tool import BitDeliminatorTool

class BitDeliminatorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Bit Deliminator Agent",
            description="Identifies and extracts bits from transcription JSON files.",
            instructions="./instructions.md",
            tools=[BitDeliminatorTool],
            temperature=0.5,
            max_prompt_tokens=25000,
        ) 