import os
import json
from typing import Any
from pydantic import BaseModel
from collections import deque

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

WORLD_FILE = "data/world.json"

INT_MAX = 2**31 - 1

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

class SaveFile(BaseModel):
    min_moves_for_score: dict[int, int] = {0: 0}
    world: list[State] = []

class DFS(Agent):
    """An agent that tries to explore every world state."""

    MAX_ACTIONS = 1000
    MAX_EXPLORATION_DEPTH = 4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.world = []
        save_file = SaveFile()

        if not os.path.exists(WORLD_FILE):
            with open(WORLD_FILE, 'w') as f:
                f.write(save_file.model_dump_json())
            
        with open(WORLD_FILE, "r") as f:
            file_dict = json.loads(f.read())
            save_file = SaveFile.model_validate(file_dict)

            self.world = save_file.world
            self.min_moves_for_score = save_file.min_moves_for_score
        
        for i, state in enumerate(self.world):
            if i != state.state_id:
                raise ValueError(f"Error: self.world[{i}].state_id should be {i} but is {state.state_id}")
        
        self.current_state = STATE_UNKNOWN
        self.last_action = GameAction.RESET
        self.nb_moves = 0
        self.nb_exploration_moves = 0

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
            return self._pick_action(GameAction.RESET)
        
        self.nb_moves += 1
        if self.last_action == GameAction.RESET:
            self.current_state = STATE_UNKNOWN
            self.nb_moves = 0
        
        self.current_state = self.next_state(
            state=self.current_state,
            action_id=self._action_to_id(self.last_action),
            new_frame=latest_frame
        )
        print(f"({self.current_state.state_id} - {self.nb_exploration_moves}) -> ", end="")
        if self.nb_exploration_moves > self.MAX_EXPLORATION_DEPTH:
            return self._pick_action(GameAction.RESET)

        # UPDATE DISTANCE TO SCORE KNOWLEDGE
        assert self.nb_moves >= self.current_state.distance_from_start
        score = self.current_state.score
        if score in self.min_moves_for_score.keys():
            self.min_moves_for_score[score] = min(self.min_moves_for_score[score], self.nb_moves)

        # SAVE TO FILE
        self.save_world()

        # ACTION CHOICE
        path_to_score_augmentation = self.find_nearest_score_augmentation(self.current_state)
        path_to_unkown_state = self.find_nearest_unknown_state(self.current_state)

        a_score_aug = path_to_score_augmentation[0] if path_to_score_augmentation else None
        a_unknown = path_to_unkown_state[0] if path_to_unkown_state else None
        dist_score_aug = (self.nb_moves + len(path_to_score_augmentation)) if path_to_score_augmentation else INT_MAX
        dist_unknown = (self.nb_moves + len(path_to_unkown_state)) if path_to_unkown_state else INT_MAX

        if not path_to_unkown_state:
            raise ValueError("WTF I know everything")

        if dist_unknown < self._min_moves_from_start_to_upgrade(score):
            return self._pick_action_id(a_unknown)
        
        if dist_score_aug < self._min_moves_from_start_to_upgrade(score):
            return self._pick_action_id(a_score_aug)
        
        return self._pick_action(GameAction.RESET)
    
    def _pick_action_id(self, action_id: int):
        self.last_action = self._id_to_action(action_id)
        return self.last_action
    
    def _pick_action(self, action: GameAction):
        self.last_action = action
        return self.last_action
    
    def _min_moves_from_start_to_upgrade(self, score: int):
        closest_upgrade = INT_MAX
        closest_upgrade_distance = INT_MAX
        for key, val in self.min_moves_for_score.items():
            if key > score and key < closest_upgrade:
                closest_upgrade = key
                closest_upgrade_distance = val
        return closest_upgrade_distance
    
    def find_nearest_unknown_state(self, start_state: State) -> list[int]:
        """
        Perform BFS to find the nearest unknown state.
        Returns action_id path to do to go there
        """
        visited = set()
        queue = deque([(start_state.state_id, [])])

        i = 0
        while queue:
            current_id, path = queue.popleft()

            if current_id == -1:
                return path

            i += 1
            if (i % 10 in [8, 9]):
                # print(f"nearest_unkn {i}: {current_id}")
                pass

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

        i = 0
        while queue:
            current_id, path = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            current_state = self._get_state(current_id)
            if current_state.state_id < 0:
                continue

            i += 1
            if (i % 10 in [8, 9]):
                # print(f"nearest_score {i}: {current_id}")
                # print(f"visited: {visited}")
                pass

            if current_state.score > start_state.score:
                return path


            for action_id, neighbor_id in enumerate(current_state.neighboors):  # Assuming state.n gives list of neighbor IDs
                queue.append((neighbor_id, path + [action_id]))

        return None  # No score augmentation state found

    
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
            if state.frame == frame.frame[-1]:
                return i
        return -1
    
    def _create_state(self, frame: FrameData, distance_from_start: int) -> State:
        new_id = len(self.world)
        self.world.append(State(
            state_id=new_id,
            frame=frame.frame[-1],
            neighboors=[-1,-1,-1,-1],
            score=frame.score,
            distance_from_start=distance_from_start,
        ))
        return self.world[-1]
    
    def _get_state(self, state_id: int) -> State:
        if state_id < 0 or state_id > len(self.world):
            return STATE_UNKNOWN
        return self.world[state_id]
    
    def next_state(self, state: State, action_id: int, new_frame: FrameData):
        if state.state_id == STATE_UNKNOWN.state_id: # START STATE
            distance_from_start = 0
        else:
            distance_from_start = state.distance_from_start + 1
        

        new_index = self.get_index_for_frame(new_frame)
        if new_index >= 0:
            new_state = self._get_state(new_index)
            new_state.distance_from_start = min(new_state.distance_from_start, distance_from_start) # Update distance_from_start
            if new_index != state.state_id:
                self.nb_exploration_moves = 0
        else:
            new_state = self._create_state(frame=new_frame, distance_from_start=distance_from_start)
            self.nb_exploration_moves += 1

        if action_id in [0,1,2,3] and state.state_id >= 0:
            state.neighboors[action_id] = new_state.state_id
        return new_state
    
    def save_world(self):
        with open(WORLD_FILE, "w") as f:
            save_file = SaveFile(
                min_moves_for_score=self.min_moves_for_score,
                world=self.world,
            )
            f.write(save_file.model_dump_json())


        

