from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces
from twin_runtime.twin_runtime_core import LogLevel, TwinRuntime


class TwinBuilderEnv(gymnasium.Env):
    def __init__(self, env_config):
        # Following attributes should be adapted to your use case
        self.twin_model_file = Path(__file__).parent / "TwinModel.twin"
        self.observation_space = spaces.Dict(
            {
                "PC": spaces.Box(low=float("-inf"), high=float("inf")),
                "PCabin": spaces.Box(low=float("-inf"), high=float("inf")),
                "ceiling_t": spaces.Box(low=float("-inf"), high=float("inf")),
                "altitude_out": spaces.Box(low=float("-inf"), high=float("inf")),
                "velocity_out": spaces.Box(low=float("-inf"), high=float("inf")),
            }
        )
        self.action_space = spaces.Dict(
            {
                "controlFlow": spaces.Box(low=0, high=6),
                "outflow": spaces.Box(low=0, high=350),
            }
        )

        # These should be left as-is or modified if you need to
        self.number_of_warm_up_steps = env_config.get("number_of_warm_up_steps", 0)
        self.warmup_action_vals = env_config.get("warm_up_action_variable_values")
        self.state = {}
        self.twin_runtime = None  # assigned in reset
        self.done = False
        self.step_size = 0.5
        self.time_index = 0.0

    def reset(self, *, seed=None, options=None):
        self.done = False
        runtime_log = str(self.twin_model_file.with_suffix(".log"))

        if self.twin_runtime is not None:
            self.twin_runtime.twin_close()

        # Load Twin, set the parameters values, initialize, generate snapshots, output)
        self.twin_runtime = TwinRuntime(
            self.twin_model_file, runtime_log, log_level=LogLevel.TWIN_LOG_ALL
        )
        self.twin_runtime.twin_instantiate()
        self.twin_runtime.twin_initialize()

        for state_variable_name in self.observation_space.keys():
            self.state[state_variable_name] = 0

        self.time_index = 0.0

        # Run initial steps to "warm up" the simulation
        if self.number_of_warm_up_steps > 0:
            action = {
                key: val for key, val in zip(self.action_space, self.warmup_action_vals)
            }
            for i in range(self.number_of_warm_up_steps):
                self.step(action)
        return self._get_obs(), {}

    def _get_obs(self):
        """Get the observable state."""
        print(f"returning state: {self.state}")
        return {key: np.array([val]) for key, val in self.state.items()}

    def reward(self, state):
        return 1

    def step(self, action):
        """Called for each step of the episode"""

        if self.twin_runtime is None:
            raise ValueError("twin_runtime cannot be None.")
        for f in action.keys():
            self.twin_runtime.twin_set_input_by_name(f, action[f])

        self.twin_runtime.twin_simulate(self.time_index)

        for state_variable_name in self.observation_space.keys():
            value = self.twin_runtime.twin_get_output_by_name(state_variable_name).value
            self.state[state_variable_name] = value

        # increase the index
        self.time_index += self.step_size
        return self._get_obs(), self.reward(self.state), self.done, self.done, {}

    def episode_finish(self):
        """Called at the end of each episode"""
        self.twin_runtime.twin_close()
        self.twin_runtime = None


env_config = {
    "number_of_warm_up_steps": 5,
    "warm_up_action_variable_values": [(6.0 - 0.0) / 2.0, (350.0 - 0.0) / 2.0],
}
