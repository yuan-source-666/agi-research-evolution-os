"""
AGI World Model v0.1
====================
Latent-space state prediction model for planning and imagination.

Core concept (JEPA-inspired):
  Instead of predicting raw observations, predict in a compact latent space.
  The world model learns: state_t -> action -> state_{t+1} transitions.

Architecture:
  1. StateEncoder: encodes a situation description into a latent vector
  2. ActionPredictor: predicts next state given current state + action
  3. StateDecoder: decodes latent vector back to description
  4. TransitionModel: combines encoder + predictor for full transition
  5. ImaginationEngine: simulates future trajectories in latent space
  6. PlanningModule: uses imagination to find good action sequences

This is a lightweight prototype using simple TF-IDF style encoding.
No neural network training required -- works with the LLM inference engine.
The LLM acts as the "physics engine" for the world model.

Dependencies: standard library + requests (for inference engine)
"""

import json
import time
import os
import hashlib
import threading
import math
from typing import Any, Optional, Callable, List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class WorldPhase(Enum):
    IDLE = "idle"
    ENCODING = "encoding"
    PREDICTING = "predicting"
    IMAGINING = "imagining"
    PLANNING = "planning"


@dataclass
class WorldState:
    """A state in the world model."""
    id: str = ""
    description: str = ""
    latent_vector: list = field(default_factory=list)  # compact representation
    properties: dict = field(default_factory=dict)     # structured attributes
    timestamp: float = 0.0
    parent_id: str = ""                                 # parent state
    action_taken: str = ""                              # action from parent


@dataclass
class WorldAction:
    """An action in the world model."""
    id: str = ""
    name: str = ""
    description: str = ""
    preconditions: dict = field(default_factory=dict)
    effects: dict = field(default_factory=dict)
    success_rate: float = 0.5


@dataclass
class Trajectory:
    """A simulated trajectory through the world."""
    id: str = ""
    states: list = field(default_factory=list)     # list of state descriptions
    actions: list = field(default_factory=list)   # list of action descriptions
    total_reward: float = 0.0
    length: int = 0
    success: bool = False
    timestamp: float = 0.0


@dataclass
class PredictionResult:
    """Result of a state transition prediction."""
    predicted_state: str = ""
    confidence: float = 0.0
    latent_vector: list = field(default_factory=list)
    reasoning: str = ""
    latency: float = 0.0


class StateEncoder:
    """Encodes world states into compact latent vectors.

    Uses a hybrid approach:
    1. TF-IDF style bag-of-words for text features
    2. Property extraction for structured features
    3. Dimensionality reduction via hashing (simulates latent space)
    """

    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim
        self.vocab = {}  # word -> index
        self.lock = threading.Lock()

    def _tokenize(self, text: str) -> list:
        """Simple tokenization."""
        import re
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens += cn_chars
        for i in range(len(cn_chars) - 1):
            tokens.append(cn_chars[i] + cn_chars[i + 1])
        return tokens

    def encode(self, description: str, properties: dict = None) -> list:
        """Encode a state description into a latent vector."""
        tokens = self._tokenize(description)
        # TF features
        tf = {}
        for t in tokens:
            with self.lock:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab) % self.latent_dim
            idx = self.vocab[t]
            tf[idx] = tf.get(idx, 0) + 1

        # Hash into latent_dim dimensions
        latent = [0.0] * self.latent_dim
        for idx, count in tf.items():
            latent[idx] = float(count)

        # Normalize
        norm = math.sqrt(sum(v * v for v in latent))
        if norm > 0:
            latent = [v / norm for v in latent]

        # Add property features (append to latent if room)
        if properties:
            for key, value in properties.items():
                h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % self.latent_dim
                if isinstance(value, (int, float)):
                    latent[h] = (latent[h] + float(value)) / 2.0
                elif isinstance(value, str):
                    # Simple string hash feature
                    vh = int(hashlib.md5(value.encode()).hexdigest()[:8], 16) % 100
                    latent[h] = (latent[h] + vh / 100.0) / 2.0

        return latent

    @staticmethod
    def similarity(vec_a: list, vec_b: list) -> float:
        """Cosine similarity between two latent vectors."""
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        na = math.sqrt(sum(a * a for a in vec_a))
        nb = math.sqrt(sum(b * b for b in vec_b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


class TransitionModel:
    """Predicts state transitions using the LLM inference engine as physics engine.

    The LLM is prompted with current state + action and asked to predict the
    next state. This is a "world model as language model" approach.
    """

    def __init__(self, inference_url: str = "http://127.0.0.1:8080",
                 inference_timeout: int = 120):
        self.inference_url = inference_url.rstrip("/")
        self.inference_timeout = inference_timeout
        self.encoder = StateEncoder(latent_dim=64)
        self.lock = threading.RLock()

    def predict(self, current_state: str, action: str,
                context: str = "") -> PredictionResult:
        """Predict next state given current state and action.
        Uses the LLM inference engine as the world dynamics predictor.
        """
        t_start = time.time()
        import requests as req

        system_prompt = (
            "You are a world model simulator. Given a current state and an action, "
            "predict the most likely next state. Be concise (2-3 sentences). "
            "Output only the predicted next state description."
        )

        user_prompt = (
            f"Current state: {current_state}\n"
            f"Action taken: {action}\n"
        )
        if context:
            user_prompt += f"Context: {context}\n"
        user_prompt += "Predict the next state:"

        try:
            payload = {
                "model": "qwen2.5-32b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 256,
                "temperature": 0.3,  # low temperature for deterministic prediction
                "top_p": 0.9,
            }
            resp = req.post(
                f"{self.inference_url}/v1/chat/completions",
                json=payload,
                timeout=self.inference_timeout
            )

            latency = time.time() - t_start

            if resp.status_code != 200:
                return PredictionResult(
                    predicted_state="[prediction failed]",
                    confidence=0.0,
                    latency=latency,
                    reasoning=f"HTTP {resp.status_code}"
                )

            data = resp.json()
            predicted = data["choices"][0]["message"]["content"].strip()

            # Encode predicted state
            latent = self.encoder.encode(predicted)

            # Confidence based on response length and determinism
            confidence = min(len(predicted) / 100.0, 0.9)

            return PredictionResult(
                predicted_state=predicted,
                confidence=confidence,
                latent_vector=latent,
                reasoning="LLM-based prediction",
                latency=round(latency, 2)
            )

        except Exception as e:
            return PredictionResult(
                predicted_state=f"[error: {str(e)[:100]}]",
                confidence=0.0,
                latency=round(time.time() - t_start, 2),
                reasoning=str(e)[:200]
            )


class ImaginationEngine:
    """Simulates future trajectories in latent space.

    Given an initial state, imagines multiple possible futures by
    simulating actions and predicting outcomes using the transition model.
    """

    def __init__(self, transition_model: TransitionModel,
                 max_depth: int = 5, max_branches: int = 3):
        self.transition_model = transition_model
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.lock = threading.RLock()
        self.trajectory_history: List[Trajectory] = []

    def imagine(self, initial_state: str, possible_actions: List[str],
                depth: int = None, context: str = "") -> List[Trajectory]:
        """Imagine multiple future trajectories from initial state.
        For each action, predict next state, then branch recursively.
        """
        depth = depth or self.max_depth
        trajectories = []

        def _simulate(state: str, actions_so_far: list, states_so_far: list,
                      current_depth: int, current_reward: float):
            if current_depth >= depth or not possible_actions:
                traj = Trajectory(
                    id=hashlib.md5(f"{time.time()}{len(trajectories)}".encode()).hexdigest()[:12],
                    states=states_so_far,
                    actions=actions_so_far,
                    total_reward=current_reward,
                    length=len(actions_so_far),
                    success=True,
                    timestamp=time.time()
                )
                trajectories.append(traj)
                return

            for action in possible_actions[:self.max_branches]:
                prediction = self.transition_model.predict(state, action, context)
                new_state = prediction.predicted_state

                # Simple reward: confidence of prediction
                reward = prediction.confidence

                _simulate(
                    new_state,
                    actions_so_far + [action],
                    states_so_far + [new_state],
                    current_depth + 1,
                    current_reward + reward
                )

                if len(trajectories) >= self.max_branches * 2:
                    break

        _simulate(initial_state, [], [initial_state], 0, 0.0)

        with self.lock:
            self.trajectory_history.extend(trajectories)

        return trajectories

    def best_trajectory(self, trajectories: List[Trajectory]) -> Optional[Trajectory]:
        """Select the trajectory with highest total reward."""
        if not trajectories:
            return None
        return max(trajectories, key=lambda t: t.total_reward)


class PlanningModule:
    """Uses imagination to find good action sequences.

    Combines world model imagination with evaluation to plan
    the best course of action.
    """

    def __init__(self, imagination_engine: ImaginationEngine):
        self.imagination = imagination_engine
        self.lock = threading.RLock()
        self.plan_history: List[dict] = []

    def plan(self, initial_state: str, goal: str,
             possible_actions: List[str], max_depth: int = 3) -> dict:
        """Plan a sequence of actions to reach a goal from initial state.

        Uses imagination to simulate, then evaluates trajectories
        against the goal.
        """
        t_start = time.time()

        # Imagine possible futures
        trajectories = self.imagination.imagine(
            initial_state, possible_actions, depth=max_depth,
            context=f"Goal: {goal}"
        )

        # Evaluate each trajectory against the goal
        best = self.imagination.best_trajectory(trajectories)

        # Simple goal alignment scoring (keyword overlap)
        goal_words = set(goal.lower().split())
        if best:
            final_state_words = set(best.states[-1].lower().split())
            alignment = len(goal_words & final_state_words) / max(len(goal_words), 1)
        else:
            alignment = 0.0

        plan_result = {
            "initial_state": initial_state,
            "goal": goal,
            "best_trajectory": asdict(best) if best else None,
            "total_trajectories": len(trajectories),
            "goal_alignment": round(alignment, 4),
            "planning_time": round(time.time() - t_start, 2),
            "timestamp": time.time(),
        }

        with self.lock:
            self.plan_history.append(plan_result)

        return plan_result

    def get_history(self, n: int = 10) -> list:
        with self.lock:
            return self.plan_history[-n:]


class WorldModel:
    """Main World Model coordinator.

    Integrates state encoding, transition prediction, imagination, and planning.
    """

    def __init__(self, inference_url: str = "http://127.0.0.1:8080",
                 inference_timeout: int = 120,
                 log_dir: str = None):
        self.encoder = StateEncoder(latent_dim=64)
        self.transition = TransitionModel(inference_url, inference_timeout)
        self.imagination = ImaginationEngine(self.transition)
        self.planner = PlanningModule(self.imagination)

        self.states: Dict[str, WorldState] = {}
        self.actions: Dict[str, WorldAction] = {}
        self.phase = WorldPhase.IDLE
        self.lock = threading.RLock()
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "WM_LOGS"
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def register_state(self, description: str, properties: dict = None,
                       parent_id: str = "", action_taken: str = "") -> WorldState:
        """Register a new world state."""
        with self.lock:
            latent = self.encoder.encode(description, properties)
            state = WorldState(
                id=hashlib.md5(f"{description}{time.time()}".encode()).hexdigest()[:12],
                description=description,
                latent_vector=latent,
                properties=properties or {},
                timestamp=time.time(),
                parent_id=parent_id,
                action_taken=action_taken,
            )
            self.states[state.id] = state
            return state

    def register_action(self, name: str, description: str = "",
                        preconditions: dict = None, effects: dict = None) -> WorldAction:
        """Register a new world action."""
        with self.lock:
            action = WorldAction(
                id=hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8],
                name=name,
                description=description,
                preconditions=preconditions or {},
                effects=effects or {},
            )
            self.actions[action.id] = action
            return action

    def predict_transition(self, state_id: str, action_description: str) -> PredictionResult:
        """Predict the next state given a registered state and an action."""
        with self.lock:
            state = self.states.get(state_id)
            if not state:
                return PredictionResult(predicted_state="[state not found]", confidence=0.0)
        return self.transition.predict(state.description, action_description)

    def imagine_futures(self, state_id: str, actions: List[str],
                        depth: int = 3) -> List[Trajectory]:
        """Imagine future trajectories from a registered state."""
        with self.lock:
            state = self.states.get(state_id)
            if not state:
                return []
        return self.imagination.imagine(state.description, actions, depth=depth)

    def plan_to_goal(self, initial_state: str, goal: str,
                     possible_actions: List[str], depth: int = 3) -> dict:
        """Plan actions to reach a goal."""
        return self.planner.plan(initial_state, goal, possible_actions, depth)

    def find_similar_states(self, description: str, top_k: int = 5) -> List[Tuple[float, WorldState]]:
        """Find states similar to a description."""
        with self.lock:
            query_vec = self.encoder.encode(description)
            scored = []
            for state in self.states.values():
                sim = StateEncoder.similarity(query_vec, state.latent_vector)
                scored.append((sim, state))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]

    def stats(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase.value,
                "total_states": len(self.states),
                "total_actions": len(self.actions),
                "trajectories_simulated": len(self.imagination.trajectory_history),
                "plans_generated": len(self.planner.plan_history),
            }

    def get_state_history(self, n: int = 20) -> list:
        with self.lock:
            sorted_states = sorted(self.states.values(), key=lambda s: s.timestamp, reverse=True)
            return [asdict(s) for s in sorted_states[:n]]

    def _log(self, event: str, message: str):
        log_path = os.path.join(self.log_dir, "world_model_log.jsonl")
        entry = {"timestamp": time.time(), "event": event, "message": message}
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
