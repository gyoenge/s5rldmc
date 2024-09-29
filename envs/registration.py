from dm_control import suite 
import jax.numpy as jnp
import distrax


class DMControlEnvironmentSampler(): 
    """DMControl environment sampler""" 
    
    def __init__(self, env_list, lo_E=None): 
        self.env_list = env_list
        # env list distribution 
        self.lo_E = lo_E if lo_E else jnp.ones(len(env_list)) / len(env_list)

    def sample(self, key: chex.PRNGKey): 
        # sample random environment 
        probabilities = distrax.Categorical(probs=self.lo_E).probs
        
        index = jrandom.choice(key, a=len(self.env_list), p=probabilities)
        domain, task = self.env_list[index]
        
        key, _key = jrandom.split(key)
        env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': _key})
        return env

