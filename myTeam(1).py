"""
Pac-Man Capture the Flag Agent: MCTS Offense + Rule-Based Defense

V2 cleanup of the original contest agent.

Key changes in v2:
- Replaced eval() team creation with a safe agent registry.
- Added clearer class and method names.
- Added a UCT-style child selection rule for MCTS.
- Separated score tracking into a lightweight ScoreTracker class.
- Reduced global state.
- Kept compatibility with the UC Berkeley Capture the Flag agent API.

Expected filename in the contest framework:
    myTeam.py

Example run from the Pac-Man Capture the Flag framework:
    python capture.py -r myTeam -b baselineTeam -n 10
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import util
from captureAgents import CaptureAgent
from game import Directions


# -----------------------------------------------------------------------------
# Team creation

def _as_bool(value) -> bool:
    """Interpret Capture-the-Flag command-line options safely."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="PacmanAgent",
    second="GhostAgent",
    **kwargs,
):
    """
    Contest framework entrypoint.

    ``use_mcts`` is an optional evaluation switch. It changes only the offensive
    return-home planner:

    - ``use_mcts=true``: UCT-style MCTS return planning (default).
    - ``use_mcts=false``: one-step greedy border return.

    The latter is the ablation condition. Food seeking, threat thresholds and
    defence remain unchanged, so the comparison isolates the contribution of
    search rather than comparing unrelated agents.
    """
    agent_registry = {
        "PacmanAgent": PacmanAgent,
        "GhostAgent": GhostAgent,
    }

    if first not in agent_registry or second not in agent_registry:
        raise ValueError(
            f"Unknown agent class. Available agents: {list(agent_registry.keys())}"
        )

    def build_agent(agent_name, index):
        agent_class = agent_registry[agent_name]
        if agent_class is PacmanAgent:
            return agent_class(index, use_mcts=kwargs.get("use_mcts", True))
        return agent_class(index)

    return [
        build_agent(first, firstIndex),
        build_agent(second, secondIndex),
    ]


# -----------------------------------------------------------------------------
# Score tracking

@dataclass
class ScoreTracker:
    """
    Tracks red and blue score changes across games.

    This keeps evaluation-related state out of the action-selection logic.
    """
    max_games: int = 10
    game_counter: int = 0
    previous_score: int = 0
    red_scores: List[int] = field(default_factory=lambda: [0] * 10)
    blue_scores: List[int] = field(default_factory=lambda: [0] * 10)
    current_red_score: int = 0
    current_blue_score: int = 0
    started_current_game: bool = False

    def start_game_if_needed(self):
        """Initialize tracking for a new game."""
        if not self.started_current_game:
            if self.game_counter < self.max_games:
                self.game_counter += 1

            self.previous_score = 0
            self.current_red_score = 0
            self.current_blue_score = 0
            self.started_current_game = True

    def update(self, current_score: int):
        """Update score arrays from the current game score."""
        self.start_game_if_needed()

        score_delta = current_score - self.previous_score

        if score_delta == 0:
            return

        idx = self.game_counter - 1

        if 0 <= idx < self.max_games:
            if score_delta > 0:
                self.current_red_score += score_delta
                self.red_scores[idx] = self.current_red_score
            else:
                self.current_blue_score += abs(score_delta)
                self.blue_scores[idx] = self.current_blue_score

        self.previous_score = current_score

    def finish_game(self):
        """Mark the current game as finished."""
        self.started_current_game = False

    def plot(self):
        """Plot red and blue scores over tracked games."""
        games = np.arange(1, self.max_games + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(games, self.blue_scores, label="Blue Team", color="blue", marker="o")
        plt.plot(games, self.red_scores, label="Red Team", color="red", marker="o")
        plt.xlabel("Game Number")
        plt.ylabel("Score")
        plt.title("Scores of Red and Blue Teams")
        plt.xticks(games)
        plt.yticks(np.arange(0, max(self.blue_scores + self.red_scores) + 1))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


SCORE_TRACKER = ScoreTracker(max_games=10)


# -----------------------------------------------------------------------------
# MCTS node

class MCTSNode:
    """
    MCTS-inspired search node for return-home planning.

    This implementation uses:
    - Expansion over legal non-stop actions
    - UCT-style child selection
    - A simple reward based on distance to the home border
    - Backpropagation of rewards through parent nodes

    It is intentionally lightweight to respect the Capture the Flag time budget.
    """

    def __init__(
        self,
        game_state,
        agent: "PacmanAgent",
        action: Optional[str],
        parent: Optional["MCTSNode"],
        center_line: List[Tuple[int, int]],
        depth_limit: int = 12,
    ):
        self.game_state = game_state.deepCopy()
        self.agent = agent
        self.action = action
        self.parent = parent
        self.center_line = center_line
        self.depth_limit = depth_limit

        self.depth = parent.depth + 1 if parent else 0
        self.visit_count = 1
        self.total_value = 0.0
        self.children: List[MCTSNode] = []

        legal_actions = game_state.getLegalActions(agent.index)
        self.unexpanded_actions = [
            action for action in legal_actions if action != Directions.STOP
        ]

    def run_search(self, time_budget_seconds: float = 0.82, max_iterations: int = 5000) -> str:
        """Run MCTS-style search within a fixed time budget."""
        start_time = time.time()

        for _ in range(max_iterations):
            if time.time() - start_time >= time_budget_seconds:
                break

            selected_node = self.select_or_expand()
            reward = selected_node.evaluate_reward()
            selected_node.backpropagate(reward)

        if not self.children:
            actions = self.game_state.getLegalActions(self.agent.index)
            non_stop_actions = [action for action in actions if action != Directions.STOP]
            return random.choice(non_stop_actions or actions)

        return self.best_child(exploration_weight=0.0).action

    def select_or_expand(self) -> "MCTSNode":
        """Select a node to evaluate or expand a new child."""
        if self.depth >= self.depth_limit:
            return self

        if self.unexpanded_actions:
            return self.expand()

        if not self.children:
            return self

        return self.best_child().select_or_expand()

    def expand(self) -> "MCTSNode":
        """Expand one unvisited action."""
        action = self.unexpanded_actions.pop()
        successor = self.game_state.generateSuccessor(self.agent.index, action)

        child = MCTSNode(
            game_state=successor,
            agent=self.agent,
            action=action,
            parent=self,
            center_line=self.center_line,
            depth_limit=self.depth_limit,
        )

        self.children.append(child)
        return child

    def best_child(self, exploration_weight: float = 1.4) -> "MCTSNode":
        """
        Select child using UCT.

        UCT score:
            average_value + c * sqrt(log(parent_visits) / child_visits)
        """
        best_score = -float("inf")
        best_nodes = []

        for child in self.children:
            average_value = child.total_value / max(child.visit_count, 1)

            exploration_bonus = exploration_weight * math.sqrt(
                math.log(max(self.visit_count, 2)) / max(child.visit_count, 1)
            )

            score = average_value + exploration_bonus

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        return random.choice(best_nodes)

    def evaluate_reward(self) -> float:
        """
        Reward moving toward the home border.

        The MCTS planner is used when retreating or under threat, so the core
        objective is to reduce distance to the center line.
        """
        current_position = self.game_state.getAgentPosition(self.agent.index)

        if current_position is None:
            return -1000.0

        if current_position == self.game_state.getInitialAgentPosition(self.agent.index):
            return -1000.0

        if not self.center_line:
            return -100.0

        min_distance_to_border = min(
            self.agent.getMazeDistance(current_position, border_position)
            for border_position in self.center_line
        )

        return -float(min_distance_to_border)

    def backpropagate(self, reward: float):
        """Backpropagate reward to root."""
        self.total_value += reward
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(reward)


# -----------------------------------------------------------------------------
# Offensive Pacman agent

class PacmanAgent(CaptureAgent):
    """
    Offensive agent.

    Strategy:
    - If safe and carrying little food: greedily seek food using features.
    - If ghosts are nearby or enough food is carried: use MCTS to return home.
    - If enemy ghosts are scared: continue aggressive food collection.
    - If not currently Pacman: reuse defensive behavior.
    """

    FOOD_RETURN_THRESHOLD = 5
    GHOST_THREAT_DISTANCE = 3

    def __init__(self, index, use_mcts=True):
        super().__init__(index)
        self.use_mcts = _as_bool(use_mcts)

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

        self.map_height = gameState.data.layout.height
        self.map_width = gameState.data.layout.width
        self.center_line = self.compute_home_border(gameState)

    def chooseAction(self, gameState):
        SCORE_TRACKER.update(gameState.data.score)

        agent_state = gameState.getAgentState(self.index)
        is_pacman = agent_state.isPacman

        if not is_pacman:
            return self.choose_defensive_action(gameState)

        carried_food = agent_state.numCarrying
        actions = gameState.getLegalActions(self.index)
        non_stop_actions = [action for action in actions if action != Directions.STOP]

        nearby_ghosts = self.get_nearby_enemy_ghosts(gameState)
        available_food = self.getFood(gameState).asList()

        if self.any_enemy_ghost_scared(gameState, scared_threshold=10):
            return self.choose_best_food_action(gameState, non_stop_actions or actions)

        if not nearby_ghosts and carried_food <= self.FOOD_RETURN_THRESHOLD:
            return self.choose_best_food_action(gameState, non_stop_actions or actions)

        should_return_home = (
            len(available_food) < 2
            or carried_food > self.FOOD_RETURN_THRESHOLD
            or bool(nearby_ghosts)
        )

        if should_return_home:
            if self.use_mcts:
                return self.choose_return_home_action(gameState)
            return self.choose_greedy_return_home_action(gameState)

        return self.choose_best_food_action(gameState, non_stop_actions or actions)

    def choose_best_food_action(self, gameState, actions: List[str]) -> str:
        """Choose action with highest food-seeking feature score."""
        action_scores = [
            self.evaluate_food_action(gameState, action)
            for action in actions
        ]

        best_score = max(action_scores)
        best_actions = [
            action for action, score in zip(actions, action_scores)
            if score == best_score
        ]

        return random.choice(best_actions)

    def choose_return_home_action(self, gameState) -> str:
        """Use MCTS-style search to choose a retreat action."""
        root = MCTSNode(
            game_state=gameState,
            agent=self,
            action=None,
            parent=None,
            center_line=self.center_line,
        )

        return root.run_search()

    def choose_greedy_return_home_action(self, gameState) -> str:
        """
        One-step return policy used only for the MCTS ablation.

        It shares the same return trigger and home-border objective as the full
        agent, but deliberately removes tree search. That makes the ablation a
        fair test of whether look-ahead improves retreat decisions.
        """
        actions = gameState.getLegalActions(self.index)
        candidate_actions = [
            action for action in actions if action != Directions.STOP
        ] or actions

        if not candidate_actions:
            return Directions.STOP

        if not self.center_line:
            return random.choice(candidate_actions)

        action_distances = []
        for action in candidate_actions:
            successor = self.get_successor(gameState, action)
            position = successor.getAgentState(self.index).getPosition()

            if position is None:
                distance = float("inf")
            else:
                distance = min(
                    self.getMazeDistance(position, border_position)
                    for border_position in self.center_line
                )

            action_distances.append((action, distance))

        best_distance = min(distance for _, distance in action_distances)
        best_actions = [
            action for action, distance in action_distances
            if distance == best_distance
        ]
        return random.choice(best_actions)

    def evaluate_food_action(self, gameState, action: str) -> float:
        """Evaluate food-seeking action with simple linear features."""
        successor = self.get_successor(gameState, action)

        current_food_carried = gameState.getAgentState(self.index).numCarrying
        next_food_carried = successor.getAgentState(self.index).numCarrying

        features = util.Counter()

        if next_food_carried > current_food_carried:
            features["eat_food"] = 1
        else:
            remaining_food = self.getFood(successor).asList()
            if remaining_food:
                current_position = successor.getAgentState(self.index).getPosition()
                features["distance_to_food"] = min(
                    self.getMazeDistance(current_position, food)
                    for food in remaining_food
                )

        weights = {
            "eat_food": 100,
            "distance_to_food": -1,
        }

        return features * weights

    def choose_defensive_action(self, gameState) -> str:
        """Defensive fallback when the offensive agent is on home territory."""
        actions = gameState.getLegalActions(self.index)
        non_stop_actions = [action for action in actions if action != Directions.STOP]
        candidate_actions = non_stop_actions or actions

        action_scores = [
            self.evaluate_defensive_action(gameState, action)
            for action in candidate_actions
        ]

        best_score = max(action_scores)
        best_actions = [
            action for action, score in zip(candidate_actions, action_scores)
            if score == best_score
        ]

        return random.choice(best_actions)

    def evaluate_defensive_action(self, gameState, action: str) -> float:
        features = self.get_defensive_features(gameState, action)
        weights = self.get_defensive_weights()
        return features * weights

    def get_successor(self, gameState, action: str):
        return gameState.generateSuccessor(self.index, action)

    def get_nearby_enemy_ghosts(self, gameState) -> List[int]:
        """Return enemy ghosts within threat distance."""
        enemy_ghosts = self.get_visible_enemy_ghosts(gameState)
        my_position = gameState.getAgentPosition(self.index)

        if my_position is None:
            return []

        nearby_ghosts = []

        for ghost_index in enemy_ghosts:
            ghost_position = gameState.getAgentPosition(ghost_index)

            if ghost_position is None:
                continue

            distance = self.getMazeDistance(my_position, ghost_position)

            if distance <= self.GHOST_THREAT_DISTANCE:
                nearby_ghosts.append(ghost_index)

        return nearby_ghosts

    def get_visible_enemy_ghosts(self, gameState) -> List[int]:
        """Return visible enemy ghosts that are not scared."""
        enemy_ghosts = []

        for opponent_index in self.getOpponents(gameState):
            opponent_state = gameState.getAgentState(opponent_index)
            opponent_position = gameState.getAgentPosition(opponent_index)

            if (
                not opponent_state.isPacman
                and opponent_state.scaredTimer == 0
                and opponent_position is not None
            ):
                enemy_ghosts.append(opponent_index)

        return enemy_ghosts

    def any_enemy_ghost_scared(self, gameState, scared_threshold: int = 10) -> bool:
        """Check whether any opponent ghost is scared for long enough to attack safely."""
        for opponent_index in self.getOpponents(gameState):
            opponent_state = gameState.getAgentState(opponent_index)

            if not opponent_state.isPacman and opponent_state.scaredTimer > scared_threshold:
                return True

        return False

    def compute_home_border(self, gameState) -> List[Tuple[int, int]]:
        """Compute legal center-line positions that lead back to home territory."""
        walls = gameState.getWalls().asList()

        if self.red:
            center_x = (self.map_width // 2) - 1
        else:
            center_x = self.map_width // 2

        center_positions = []

        for y in range(self.map_height):
            current_position = (center_x, y)
            adjacent_position = (center_x + 1 - 2 * self.red, y)

            if current_position not in walls and adjacent_position not in walls:
                center_positions.append(current_position)

        return center_positions

    def get_defensive_features(self, gameState, action: str):
        successor = self.get_successor(gameState, action)
        agent_state = successor.getAgentState(self.index)
        agent_position = agent_state.getPosition()

        features = util.Counter()
        features["on_defense"] = 0 if agent_state.isPacman else 1

        invaders = []
        for opponent_index in self.getOpponents(successor):
            opponent_state = successor.getAgentState(opponent_index)
            if opponent_state.isPacman and opponent_state.getPosition() is not None:
                invaders.append(opponent_state)

        features["num_invaders"] = len(invaders)

        if invaders and agent_position is not None:
            features["invader_distance"] = min(
                self.getMazeDistance(agent_position, invader.getPosition())
                for invader in invaders
            )

        if action == Directions.STOP:
            features["stop"] = 1

        reverse_direction = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]

        if action == reverse_direction:
            features["reverse"] = 1

        return features

    def get_defensive_weights(self):
        return {
            "on_defense": 100,
            "num_invaders": -1000,
            "invader_distance": -10,
            "stop": -100,
            "reverse": -2,
        }


# -----------------------------------------------------------------------------
# Defensive Ghost agent

class GhostAgent(CaptureAgent):
    """
    Defensive Ghost agent.

    Strategy:
    - Stay on defense.
    - Chase visible enemy Pacmen.
    - Avoid stopping.
    - Slightly discourage reversing.
    """

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        if not actions:
            return Directions.STOP

        action_scores = [
            self.evaluate_defensive_action(gameState, action)
            for action in actions
        ]

        best_score = max(action_scores)
        best_actions = [
            action for action, score in zip(actions, action_scores)
            if score == best_score
        ]

        return random.choice(best_actions)

    def evaluate_defensive_action(self, gameState, action: str) -> float:
        features = self.get_defensive_features(gameState, action)
        weights = self.get_defensive_weights()
        return features * weights

    def get_defensive_features(self, gameState, action: str):
        successor = self.get_successor(gameState, action)
        agent_state = successor.getAgentState(self.index)
        agent_position = agent_state.getPosition()

        features = util.Counter()
        features["on_defense"] = 0 if agent_state.isPacman else 1

        invaders = []

        for opponent_index in self.getOpponents(successor):
            opponent_state = successor.getAgentState(opponent_index)

            if opponent_state.isPacman and opponent_state.getPosition() is not None:
                invaders.append(opponent_state)

        features["num_invaders"] = len(invaders)

        if invaders and agent_position is not None:
            features["invader_distance"] = min(
                self.getMazeDistance(agent_position, invader.getPosition())
                for invader in invaders
            )

        if action == Directions.STOP:
            features["stop"] = 1

        reverse_direction = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]

        if action == reverse_direction:
            features["reverse"] = 1

        return features

    def get_defensive_weights(self):
        return {
            "on_defense": 100,
            "num_invaders": -1000,
            "invader_distance": -10,
            "stop": -100,
            "reverse": -2,
        }

    def get_successor(self, gameState, action: str):
        return gameState.generateSuccessor(self.index, action)
