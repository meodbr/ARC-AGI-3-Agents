import random
import json
from typing import Any
from pydantic import BaseModel
from collections import deque

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

WORLD_FILE = "data/world.json"

class State(BaseModel):
    state_id: int
    frame: list[list[int]]
    neighboors: list[int]
    score: int
    distance_from_start: int

STATE_UNKNOWN = State(
    state_id=-1,
    frame=[],
    neighboors=[-1,-1,-1,-1],
    score=0,
    distance_from_start=-1
)


class DFS(Agent):
    """An agent that tries to explore every world state."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.world = []
        with open(WORLD_FILE, "r") as f:
            world_dict = json.loads(f.read())
            for state in world_dict:
                self.world.append(State.model_validate(state))
        
        self.current_state = STATE_UNKNOWN
        self.last_action = GameAction.RESET

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            return GameAction.RESET
        
        self.current_state = self.next_state(
            state=self.current_state,
            action_id=self._action_to_id(self.last_action),
            new_frame=latest_frame.frame[-1]
        )
        
        path_to_unkown_state = self.find_nearest_unknown_state(self.current_state)
        if path_to_unkown_state:
            self.last_action = self._id_to_action(path_to_unkown_state[0])
        else:
            self.last_action = GameAction.RESET
        
        self.save_world()

        return self.last_action
    
    def find_nearest_unknown_state(self, start_state: State) -> list[int]:
        """
        Perform BFS to find the nearest unknown state.
        Returns action_id path to do to go there
        """
        visited = set()
        queue = deque([(start_state.state_id, [])])

        while queue:
            current_id, path = queue.popleft()

            if current_id == -1:
                return path

            current_state = self._get_state(current_id)

            visited.add(current_id)

            for action_id, neighbor_id in enumerate(current_state.neighboors):  # Assuming state.n gives list of neighbor IDs
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [action_id]))

        return None  # No unknown state found
    
    def find_nearest_score_augmentation(self, start_state: State) -> list[int]:
        """
        Perform BFS to find the nearest unknown state.
        Returns action_id path to do to go there
        """
        visited = set()
        queue = deque([(start_state.state_id, [])])

        while queue:
            current_id, path = queue.popleft()

            current_state = self._get_state(current_id)

            if current_state.score > start_state.score:
                return path

            visited.add(current_id)

            for action_id, neighbor_id in enumerate(current_state.neighboors):  # Assuming state.n gives list of neighbor IDs
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [action_id]))

        return None  # No unknown state found

    
    @staticmethod
    def _action_to_id(action: GameAction):
        if action.value < 1 or action.value > 4:
            return -1
        return action.value - 1
    
    @staticmethod
    def _id_to_action(action_id: int):
        return GameAction.from_id(action_id + 1)
    
    def get_index_for_frame(self, frame: FrameData):
        for i, state in enumerate(self.world):
            if state.frame == frame:
                return i
        return -1
    
    def _create_state(self, frame: FrameData) -> State:
        new_id = len(self.world)
        self.world.append(State(
            state_id=new_id,
            frame=frame.frame,
            neighboors=[-1,-1,-1,-1]
        ))
        return self.world[-1]
    
    def _get_state(self, state_id: int) -> State:
        return self.world[state_id]
    
    def next_state(self, state: State, action_id: int, new_frame: FrameData):
        new_index = self.get_index_for_frame(new_frame)
        if new_index >= 0:
            new_state = self._get_state(new_index)
        else:
            new_state = self._create_state(frame=new_frame)

        if action_id in [0,1,2,3] and state.state_id >= 0:
            state.neighboors[action_id] = new_state.state_id
        return new_state
    
    def save_world(self):
        with open(WORLD_FILE, "w") as f:
            json.dump([s.model_dump() for s in self.world], f, indent=4)


        

