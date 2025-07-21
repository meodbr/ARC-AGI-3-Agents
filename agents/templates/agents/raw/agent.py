from google.adk.agents import Agent
import os

from .prompt import SYSTEM_INSTRUCTIONS

root_agent = Agent(
    name="raw_agent",
    model=os.environ.get('GEMINI_MODEL'),
    description=(
        "Agent solving 2D pixel puzzles"
    ),
    instruction=SYSTEM_INSTRUCTIONS,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)