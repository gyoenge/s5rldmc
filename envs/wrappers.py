import dm_env 
# from dm_env import specs, TimeStep 
from dm_env import StepType 
import jax 
import jax.numpy as jnp 
import numpy as np 
from flax import struct 
from typing import Tuple 
import chex 
from gymnax.environments import spaces 


### 


@struct.dataclass 
class EnvState: 
    timestep: int 
    state: Dict[str, jnp.ndarray] = struct.field(default_factory=dict) 

class PopJaxWrapper(): 
    """Convert DMC env -> PopJax env format"""    
    def __init__(self, env: dm_env.Environment):
        self._env = env 
        
    def _dmc_obs_converter(self, dmc_obs) -> Tuple[chex.Array, EnvState]:
        jnp_arrays = []
        state_dict = {}
        for name, np_array in dmc_obs.items(): 
            jnp_array = jnp.array(np_array)
            jnp_arrays.append(jnp_array)
            state_dict[name] = jnp_array
        obs = jnp.concatenate(jnp_arrays, axis=0)
        state = EnvState(
            timestep = 0, 
            state = state_dict
        )
        return obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> Tuple[chex.Array, EnvState]: 
        dmc_time_step = self._env.reset() 
        
        dmc_obs = dmc_time_step.observation 
        obs, state = self._dmc_obs_converter(dmc_obs)
        
        return obs, state 
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: chex.Array) -> Tuple[chex.Array, EnvState, float, bool, dict]: 
        # if isinstance(action, np.ndarray):
        #     action = jnp.array(action)
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        dmc_time_step = self._env.step(action) 
        
        dmc_obs = dmc_time_step.observation 
        obs, state = self._dmc_obs_converter(dmc_obs) 
        reward = dmc_time_step.reward 
        done = (dmc_time_step.step_type == StepType.LAST)  #### 
        info = {}  
        
        return obs, state, reward, done, info
    
    def observation_space(self) -> spaces.Box:
        dmc_observation_spec = self._env.observation_spec()
        
        total_length = sum(spec['shape'][0] for spec in dmc_observation_spec.values())
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_length,), dtype=np.float64
        )
        
        return obs_space
    
    def action_space(self) -> spaces.Box:
        dmc_action_spec = self._env.action_spec()
        
        action_space = spaces.Box(
            low=dmc_action_spec['minimum'], 
            high=dmc_action_spec['maximum'], 
            shape=dmc_action_spec['shape'], 
            dtype=np.float64
        )
        
        return action_space


### 


class RandomProjectionWrapper():
    """Random Observation & Action Projection"""

    def __init__(self, env: PopJaxWrapper, O, A):
        self._env = env 
        
        self.O : int = O
        self.A : int = A
        self.E_o : int = self._env.observation_space().shape[0]
        self.E_a : int = self._env.action_space().shape[0]
        
        key = jax.random.PRNGKey(0)
        self.M_o = jax.random.normal(key, (self.O, self.E_o))
        self.M_a = jax.random.normal(key, (self.E_a, self.A))
        
        self.action_minimums = self._env.action_space.low
        self.action_maximums = self._env.action_space.high

    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> Tuple[chex.Array, EnvState]: 
        obs, state = self._env.reset()
        
        # projected obs (E_o -> O)
        obs = self.M_o @ obs  
          
        return obs, state 
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: chex.Array) -> Tuple[chex.Array, EnvState, float, bool, dict]: 
        assert action.shape[0] == self.A
        
        # projected action (A -> E_a)
        action = self.M_a @ action 
        action = self._scale_vector(action, self.action_minimums, self.action_maximums)

        obs, state, reward, done, info = self._env.step(state, action)
        
        # projected obs (E_o -> O)
        obs = self.M_o @ obs    
        
        return obs, state, reward, done, info
    
    def _scale_vector(self, vector, minimums, maximums):
        length = vector.shape[0]
        assert minimums.shape[0] == length
        assert maximums.shape[0] == length
        
        # rescale each vector[i] from ~(-1,1) to ~(min[i],max[i])
        scaled_vector = minimums + (vector-(-1))*(maximums-minimums)/(1-(-1))  
        return scaled_vector  
    
    def observation_space(self) -> spaces.Box:
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.O,), dtype=np.float64
        )
        return obs_space
    
    def action_space(self) -> spaces.Box:
        action_space = spaces.Box(
            low=jnp.full((self.A,), -1), 
            high=jnp.full((self.A,), 1), 
            shape=(self.A,), 
            dtype=np.float64
        )
        return action_space


###

        
@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper():
    """Log the episode returns and lengths."""

    def __init__(self, env: PopJaxWrapper):
        self._env = env 

    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset()
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LogEnvState, action: chex.Array) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(state.env_state, action)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state = env_state,
            episode_returns = new_episode_return * (1 - done),
            episode_lengths = new_episode_length * (1 - done),
            returned_episode_returns = state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths = state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep = state.timestep + 1,
        )
        # info["returned_episode_returns"] = state.returned_episode_returns
        # info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        info["return_info"] = jnp.stack([state.timestep, state.returned_episode_returns])
        # info["timestep"] = state.timestep
        return obs, state, reward, done, info

    def observation_space(self) -> spaces.Box:
        return self._env.observation_space()
    
    def action_space(self) -> spaces.Box:
        return self._env.action_space()
    
