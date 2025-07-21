import json
import logging
import os
import textwrap
from typing import Any, Optional

# import openai
# from openai import OpenAI as OpenAIClient
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, Tool, ToolConfig, FunctionCallingConfig, FunctionResponse

from ...agent import Agent
from ...structs import FrameData, GameAction, GameState
from .prompt import OBSERVATION_PROMPT, CHOOSE_ACTION_PROMPT, SYSTEM_INSTRUCTIONS

logger = logging.getLogger()


class CustomGemini(Agent):
    """Gemini custom agent."""

    MAX_ACTIONS: int = 80

    MESSAGE_LIMIT: int = 10
    MODEL: str = "gemini-2.0-flash"
    content_history: list[dict[str, Any]]
    token_counter: int
    client: genai.Client


    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        self.last_action = None
        self.content_history = []

        functions = self.build_functions()
        self.genai_tools = Tool(function_declarations=functions)


    @property
    def name(self) -> str:
        name = f"{super().name}.{self.MODEL}"
        return name

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )
    
    def gemini_generate_content(self, prompt: list[Content], tools: list[Tool] = None, tool_config_mode="AUTO"):
        config = GenerateContentConfig()
        if tools != None:
            tool_config = ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=tool_config_mode,
                )
            )
            config = GenerateContentConfig(tools=tools, tool_config=tool_config)
        config.system_instruction = SYSTEM_INSTRUCTIONS

        if os.environ.get("DEBUG") == "True":
            logger.info(prompt)
        logger.info("Sending to Gemini...")

        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=prompt,
            config=config,
        )

        response_content = response.candidates[0].content
        self.content_history.append(response_content)
        return response_content
    
    def gemini_choose_action(self, content_history: list[Content], observations: str = None) -> GameAction:
        name = GameAction.ACTION5.name
        arguments = None

        try:
            prompt = content_history + [Content(role='user', parts=[Part.from_text(text=CHOOSE_ACTION_PROMPT)])]

            response_content = self.gemini_generate_content(prompt, [self.genai_tools], tool_config_mode='ANY')
            tool_call = response_content.parts[0].function_call
            name = tool_call.name
            arguments = tool_call.args

            logger.info(f"Chose tool: {tool_call}")
        except Exception as e:
            logger.error("Gemini action generation failed")
            raise e

        try:
            data = json.loads(json.dumps(arguments)) if arguments else {}
        except Exception as e:
            data = {}
            logger.warning(f"JSON parsing error on LLM function response: {e}")

        action = GameAction.from_name(name)
        action.set_data(data)
        action.reasoning = {
            "action": action.name,
            "observations": observations,
        }
        return action

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("google.auth").setLevel(logging.CRITICAL)

        if len(self.content_history) == 0:
            self.last_action = GameAction.RESET
            self.content_history = [Content(role="user", parts=[Part.from_function_call(name=self.last_action.name, args={})])]
            return self.last_action

        logger.info("frame: " + self.small_pretty_print_3d(latest_frame.frame))

        function_response_text = self.build_func_resp_prompt(latest_frame)
        response_part = Part.from_function_response(name=self.last_action.name, response={"result": function_response_text})
        self.content_history.append(Content(role='user', parts=[response_part]))

        content_text=""
        try:
            prompt = self.content_history + [Content(role='user', parts=[Part(text=OBSERVATION_PROMPT)])]
            response_content = self.gemini_generate_content(prompt, [self.genai_tools])
            content_text = response_content.parts[0].text
            logger.info(f"Model observations: {content_text}")
        except Exception as e:
            logger.error("Gemini observation request failed")
            raise e

        action = self.gemini_choose_action(self.content_history, content_text)
        self.last_action = action
        return action
    
    def content_to_str(content: Content) -> str:
        if os.environ.get("DEBUG") == True:
            return str(content)

        result = Content()
        for part in content.parts:
            if part.function_response != None:
                continue
            result.parts.append(part)
        return str(result)

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (3, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (4, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (1, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (2, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# State:
{state}

# Score:
{score}

# Frame:
{latest_frame}

# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.score,
                state=latest_frame.state.name,
            )
        )
    
    def build_system_instructions(self):
        return Content(role="user", parts=[Part(text=SYSTEM_INSTRUCTIONS)])

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def pretty_print_3d_old(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                line = ""
                for num in row:
                    item = str(num) + (" " if num <= 9 else "")
                    line += item
                lines.append(line)
        return "\n".join(lines)
    
    def pretty_print_3d_custom(self, array_3d: list[list[list[Any]]]) -> str:
        colormap = {
            10: "a",
            11: "b",
            12: "c",
            13: "d",
            14: "e",
            15: "f",
        }
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                line = ""
                for num in row:
                    item = str(num) if num <= 9 else colormap[num]
                    line += item + " "
                lines.append(line)
        return "\n".join(lines)
    
    def small_pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        small_array_3d = []
        for i, block in enumerate(array_3d):
            small_block = []
            for x in range(len(block)):
                small_row = []
                for y in range(len(block[x])):
                    if x % 2 == 0 and y % 2 == 0:
                        small_row.append(block[x][y])
                if small_row:
                    small_block.append(small_row)
            small_array_3d.append(small_block)
        return self.pretty_print_3d(small_array_3d)
