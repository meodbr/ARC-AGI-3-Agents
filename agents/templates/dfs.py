import os
import json
from typing import Any
from pydantic import BaseModel
from collections import deque

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

WORLD_FILE = "data/world.json"

INT_MAX = 2**31 - 1
UNKNOWN_ID = -1
NOTHING_ID = -2
AVAILABLE_ACTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]

class State(BaseModel):
    id: int
    frame: list[list[int]]
    score: int

    _path: list[int] = []
    _neighboors: list[int] = [UNKNOWN_ID]*4

    def is_known(self):
        return self.id >= 0

    def get_neighboor(self, action: GameAction):
        if action not in AVAILABLE_ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        return self._neighboors[self._action_to_id(action)]

    def set_neighboor(self, action: GameAction, neighboor_id: int):
        if action not in AVAILABLE_ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        self._neighboors[self._action_to_id(action)] = neighboor_id
        
    def get_all_neighboors(self):
        return [self.neighboors[self._action_to_id(action)] for action in AVAILABLE_ACTIONS]
    
    def depth(self):
        return len(self.path)

    def update_path(self, new_path: list[GameAction]):
        new_depth = len(new_path)
        if not self.is_known():
            return
        if new_depth >= self.depth():
            return

        self.path = [self._action_to_id(action) for action in new_path]
        for action in AVAILABLE_ACTIONS:
            n = self.get_neighboor(action)
            n.update_path(new_path + [action])

    def get_path(self):
        return [self._id_to_action(action_id) for action_id in self.path]

    @staticmethod
    def _action_to_id(action: GameAction):
        if action.value < 1 or action.value > 4:
            return -1
        return action.value - 1
    
    @staticmethod
    def _id_to_action(action_id: int):
        return GameAction.from_id(action_id + 1)
    
    
STATE_UNKNOWN = State(
    id=UNKNOWN_ID,
    frame=[],
    score=0,
    path=[],
)

class Board(BaseModel):
    _level_start_state_ids: dict[int, int] = {}
    world: list[State] = []

    def level_start_state(self, score):
        if score not in self._level_start_state_ids.keys():
            return STATE_UNKNOWN
        return self.world[self._level_start_state_ids[score]]
    
    def next_level(self, score):
        known_superior_levels = [
            key
            for key in self._level_start_state_ids.keys()
            if key > score
        ]
        if len(known_superior_levels) == 0:
            return STATE_UNKNOWN
        next_level_score = min(known_superior_levels)
        return self.world[self._level_start_state_ids[next_level_score]]
    
            
    

class DFS(Agent):
    """An agent that tries to explore every world state."""

    MAX_ACTIONS = 1000
    MAX_EXPLORATION_DEPTH = 4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.board = Board()
        self.world = self.board.world

        if not os.path.exists(WORLD_FILE):
            with open(WORLD_FILE, 'w') as f:
                f.write(self.board.model_dump_json())
            
        with open(WORLD_FILE, "r") as f:
            file_dict = json.loads(f.read())
            self.board = Board.model_validate(file_dict)

            self.world = self.board.world
            self.min_moves_for_score = self.board.min_moves_for_score
        
        for i, state in enumerate(self.world):
            if i != state.id:
                raise ValueError(f"Error: self.world[{i}].id should be {i} but is {state.id}")
        
        self.current_state = STATE_UNKNOWN
        self.last_action = GameAction.RESET
        self.current_depth = 0
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
        
        self.current_depth += 1
        if self.last_action == GameAction.RESET:
            self.current_state = STATE_UNKNOWN
            self.current_depth = 0
        
        self.current_state = self.next_state(
            state=self.current_state,
            action_id=self._action_to_id(self.last_action),
            new_frame=latest_frame
        )
        print(f"({self.current_state.id} - {self.nb_exploration_moves}) -> ", end="")
        if self.nb_exploration_moves > self.MAX_EXPLORATION_DEPTH:
            return self._pick_action(GameAction.RESET)

        # UPDATE DISTANCE TO SCORE KNOWLEDGE
        assert self.current_depth >= self.current_state.depth
        score = self.current_state.score
        if score in self.min_moves_for_score.keys():
            self.min_moves_for_score[score] = min(self.min_moves_for_score[score], self.current_depth)

        # SAVE TO FILE
        self.save_board()

        # ACTION CHOICE
        path_to_score_augmentation = self.board.next_level()
        path_to_unkown_state = self.find_nearest_unknown_state(self.current_state)

        a_score_aug = path_to_score_augmentation[0] if path_to_score_augmentation else None
        a_unknown = path_to_unkown_state[0] if path_to_unkown_state else None
        dist_score_aug = (self.current_depth + len(path_to_score_augmentation)) if path_to_score_augmentation else INT_MAX
        dist_unknown = (self.current_depth + len(path_to_unkown_state)) if path_to_unkown_state else INT_MAX

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
        queue = deque([(start_state.id, [])])

        i = 0
        while queue:
            current_id, path = queue.popleft()

            if current_id == -1:
                return path
            
            if current_id in visited or len(path) > self.MAX_BFS_DEPTH:
                continue

            visited.add(current_id)

            current_state = self._get_state(current_id)

            i += 1
            if (i % 10 in [8, 9]):
                # print(f"nearest_unkn {i}: {current_id}")
                pass


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
        queue = deque([(start_state.id, [])])

        i = 0
        while queue:
            current_id, path = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            current_state = self._get_state(current_id)
            if current_state.id < 0:
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

    def get_index_for_frame(self, frame: FrameData):
        for i, state in enumerate(self.world):
            if state.frame == frame.frame[-1]:
                return i
        return -1
    
    def _create_state(self, frame: FrameData, path: list[GameAction]) -> State:
        new_id = len(self.world)
        new_state = State(
            id=new_id,
            frame=frame.frame[-1],
            score=frame.score,
        )
        new_state.update_path(path)
        self.world.append(new_state)
        return self.world[-1]
    
    def _get_state(self, id: int) -> State:
        if id < 0 or id > len(self.world):
            return STATE_UNKNOWN
        return self.world[id]
    
    def next_state(self, state: State, action: GameAction, new_frame: FrameData):
        new_index = self.get_index_for_frame(new_frame)
        if new_index >= 0:
            new_state = self._get_state(new_index)
            new_state.update_path(state.get_path + [action])
        else:
            new_state = self._create_state(
                frame=new_frame,
                path=(state.get_path() + [action])
            )
            self.nb_exploration_moves += 1

        if action in AVAILABLE_ACTIONS and state.id >= 0:
            state.set_neighboor(action, new_state.id)
        return new_state
    
    
    def save_board(self):
        with open(WORLD_FILE, "w") as f:
            save_file = Board(
                min_moves_for_score=self.min_moves_for_score,
                world=self.world,
            )
            f.write(save_file.model_dump_json())
    


        

