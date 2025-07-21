import json
import logging
import os
import textwrap
from typing import Any, Optional

# import openai
# from openai import OpenAI as OpenAIClient
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, Tool, ToolConfig, FunctionCallingConfig

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

logger = logging.getLogger()


class Gemini(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 80
    DO_OBSERVATION: bool = True
    REASONING_EFFORT: Optional[str] = None
    MODEL_REQUIRES_TOOLS: bool = True

    MESSAGE_LIMIT: int = 10
    MODEL: str = "gemini-2.0-flash"
    messages: list[dict[str, Any]]
    token_counter: int

    _latest_tool_call_id: str = "call_12345"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        name = f"{super().name}.{self.MODEL}.{obs}"
        if self.REASONING_EFFORT:
            name += f".{self.REASONING_EFFORT}"
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

    def message_to_content(self, m):
        r = m['role']
        if r not in ['user', 'model']:
            m['role'] = 'model' if r == 'assistant' else 'user'

        if "content" in m:
            return Content(role=m["role"], parts=[Part(text=m["content"])])
        elif "function_call" in m:
            return Content(
                role=m["role"],
                parts=[
                    Part(function_call={
                        "name": m["function_call"]["name"],
                        "args": json.loads(m["function_call"]["arguments"])
                    })
                ]
            )
        elif "tool_calls" in m:
            return Content(
                role=m["role"],
                parts=[
                    Part(function_call={
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"])
                    })
                    for tc in m["tool_calls"]
                ]
            )
        elif "tool_call_id" in m:  # This is likely a tool's response
            return Content(
                role=m["role"],
                parts=[Part(text=m["content"])]
            )
        else:
            raise ValueError(f"Unsupported message format: {m}")

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("google.auth").setLevel(logging.CRITICAL)

        client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))

        tools = self.build_tools()
        functions = self.build_functions()

        if len(self.messages) == 0:
            user_prompt = self.build_user_prompt(latest_frame)
            message0 = {"role": "user", "content": user_prompt}
            self.push_message(message0)

            if self.MODEL_REQUIRES_TOOLS:
                message1 = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": self._latest_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": GameAction.RESET.name,
                                "arguments": json.dumps({}),
                            },
                        }
                    ],
                }
            else:
                message1 = {
                    "role": "assistant",
                    "function_call": {"name": "RESET", "arguments": json.dumps({})},
                }
            self.push_message(message1)
            return GameAction.RESET

        function_name = latest_frame.action_input.id.name
        function_response = self.build_func_resp_prompt(latest_frame)
        if self.MODEL_REQUIRES_TOOLS:
            message2 = {
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": str(function_response),
            }
        else:
            message2 = {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        self.push_message(message2)

        if self.DO_OBSERVATION:
            logger.info("Sending to Gemini for observation...")

            try:
                print("\n\n\n---------------------------------------------------------------------------- MMMEEESSAGSEFSES ---------------")
                print([m.keys() for m in self.messages])
                print(self.messages)
                prompt = [self.message_to_content(m) for m in self.messages]
                response = client.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                )
                content_text = response.candidates[0].content.parts[0].text
            except Exception as e:
                logger.error("Gemini observation request failed")
                logger.info(f"Message dump: {self.messages}")
                raise e

            self.track_tokens(response.usage_metadata.total_token_count, content_text)
            message3 = {"role": "assistant", "content": content_text}
            self.push_message(message3)
            logger.info(f"Assistant: {content_text}")

        user_prompt = self.build_user_prompt(latest_frame)
        message4 = {"role": "user", "content": user_prompt}
        self.push_message(message4)

        name = GameAction.ACTION5.name
        arguments = None
        message5 = None

        logger.info("Sending to Gemini for action...")

        try:
            prompt = [self.message_to_content(m) for m in self.messages]

            if self.MODEL_REQUIRES_TOOLS:
                genai_tools = Tool(function_declarations=functions)
                tool_config = ToolConfig(
                    function_calling_config=FunctionCallingConfig(
                        mode="ANY",
                    )
                )
                config = GenerateContentConfig(tools=[genai_tools], tool_config=tool_config)
                response = client.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                    config=config,
                )
                tool_call = response.candidates[0].content.parts[0].function_call
                name = tool_call.name
                arguments = tool_call.args
            else:
                response = client.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                )
                print("-------------------------\n\n-----------------------")
                print(response.text)
                print(response)
                fc = response.candidates[0].content.parts[0].function_call
                name = fc.name
                arguments = fc.args

            self.track_tokens(response.usage_metadata.total_token_count)

            message5 = {
                "role": "assistant",
                "function_call": {"name": name, "arguments": json.dumps(arguments)},
            }

        except Exception as e:
            logger.error("Gemini action generation failed")
            logger.info(f"Message dump: {self.messages}")
            raise e

        if message5:
            self.push_message(message5)

        try:
            data = json.loads(json.dumps(arguments)) if arguments else {}
        except Exception as e:
            data = {}
            logger.warning(f"JSON parsing error on LLM function response: {e}")

        action = GameAction.from_name(name)
        action.set_data(data)
        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": message,
                }
            )
        logger.info(f"Received {tokens} tokens, new total {self.token_counter}")
        # handle tool to debug messages:
        # with open("messages.json", "w") as f:
        #     json.dump(
        #         [
        #             msg if isinstance(msg, dict) else msg.model_dump()
        #             for msg in self.messages
        #         ],
        #         f,
        #         indent=2,
        #     )

    def push_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Push a message onto stack, store up to MESSAGE_LIMIT with FIFO."""
        self.messages.append(message)
        if len(self.messages) > self.MESSAGE_LIMIT:
            self.messages = self.messages[-self.MESSAGE_LIMIT :]
        if self.MODEL_REQUIRES_TOOLS:
            # cant clip the message list between tool
            # and tool_call else llm will error
            while (
                self.messages[0].get("role")
                if isinstance(self.messages[0], dict)
                else getattr(self.messages[0], "role", None)
            ) == "tool":
                self.messages.pop(0)
        return self.messages

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
                "description": "Send this simple input action (1, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, S, Down).",
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

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools

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

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

Use your tools to decide which action to take

# TURN:
Call exactly one action.
        """.format()
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "llm_user_prompt": self.build_user_prompt(self.frames[-1]),
                    "llm_tools": self.build_tools()
                    if self.MODEL_REQUIRES_TOOLS
                    else self.build_functions(),
                    "llm_tool_resp_prompt": self.build_func_resp_prompt(
                        self.frames[-1]
                    ),
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)