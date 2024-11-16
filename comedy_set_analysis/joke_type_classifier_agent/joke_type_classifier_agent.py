from agency_swarm import Agent
from comedy_set_analysis.joke_type_classifier_agent.tools.joke_type_classifier_tool import JokeTypeClassifierTool

class JokeTypeClassifierAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Joke Type Classifier Agent",
            description="Classifies the types of jokes in a bit.",
            instructions="./instructions.md",
            tools=[JokeTypeClassifierTool],
            temperature=0.5,
            max_prompt_tokens=25000,
        ) 