import jax 
import jax.numpy as jnp 
import time 
from envs import DMControlEnvironmentSampler
from envs.wrappers import ObservationRandomProjectWrapper, ActionRandomProjectWrapper
from algorithms.ppo_s5 import make_train as make_train_s5
import argparse

def run(num_runs, mode="train", arch="s5", file_tag=""):
    print("*"*10)
    print(f"Running {num_runs} runs of {mode} with arch {arch}")    
    
    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split()
    
    TRAIN_ENVS = [
        ('ball_in_cup', 'catch'),
        ('cartpole', 'swingup'), 
        ('finger', 'spin'),
        ('finger', 'turn_hard'), 
        ('point_mass', 'easy'), 
        ('reacher', 'easy')
    ]
    TEST_ENVS = [
        ('cartpole', 'balance'), 
        ('finger', 'turn_easy'),
        ('pendulum', 'swingup'),
        ('point_mass', 'hard'),
        ('reacher', 'hard')
    ]
    if mode=="train": 
        env_sampler = DMControlEnvironmentSampler(env_list=TRAIN_ENVS)
    elif mode=="test":
        raise NotImplementedError
        # env_sampler = DMControlEnvironmentSampler(env_list=TEST_ENVS)
    else: 
        raise ValueError(f"Invalid mode: {mode}. Mode should be 'train' or 'test'.")
    
    config = {
        # env setting 
        "PROJ_OBS_SIZE": 12,  # projected observation size (*)
        "PROJ_ACT_SIZE": 2,  # projected action size (*) 
        "MODE": mode, 
        "ENV_SAMPLER": env_sampler, 
        # train hyperparameters, setting  
        "LR": 5e-5,  # adam learning rate 
        "NUM_ENVS": 1 # 64,  # number of environments 
        "NUM_STEPS": 1024,  # unroll length 
        "TOTAL_TIMESTEPS": 15e6,  # number of timesteps 
        "UPDATE_EPOCHS": 30,  # number of epochs 
        "NUM_MINIBATCHES": 1 # 8,  # number of minibatches 
        "GAMMA": 0.99,  # discount 
        "GAE_LAMBDA": 1.0,  # GAE 
        "CLIP_EPS": 0.2,  # clipping coefficient 
        "ENT_COEF": 0.0,  # entropy coefficient 
        "VF_COEF": 1.0,  # value function weight 
        "MAX_GRAD_NORM": 0.5,  # maximum gradient norm 
        "ANNEAL_LR": False,  # learning rate annealing 
        "DEBUG": True, 
        # S5 model 
        "S5_D_MODEL": 256,  
        "S5_SSM_SIZE": 256,  # s5 hidden size 
        "S5_N_LAYERS": 4,  # s5 layers 
        "S5_BLOCKS": 1,  
        "S5_ACTIVATION": "full_glu",  
        "S5_DO_NORM": False,
        "S5_PRENORM": False,
        "S5_DO_GTRXL_NORM": False,
    }
    
    train_vjit_lstm = _
    train_vjit_s5 = jax.jit(jax.vmap(make_train_s5(config)))
    rngs = jax.random.split(rng, num_runs)
    info_dict = {}

    if arch == "s5":
        t0 = time.time()
        compiled_s5 = train_vjit_s5.lower(rngs).compile()
        compile_s5_time = time.time() - t0
        print(f"s5 compile time: {compile_s5_time}")

        t0 = time.time()
        out_s5 = jax.block_until_ready(compiled_s5(rngs))
        run_s5_time = time.time() - t0
        print(f"s5 time: {run_s5_time}")
        info_dict["s5"] = {
            "compile_s5_time": compile_s5_time,
            "run_s5_time": run_s5_time,
            "out": out_s5[1],
        }
    
    elif arch == "lstm": 
        raise NotImplementedError
    
    else: 
        raise NotImplementedError
    
    jnp.save(f"results/{num_runs}_{mode}_{arch}_{file_tag}.npy", info_dict)

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", type=int, required=True)
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--arch", type=str, default="s5")
args = parser.parse_args()

if __name__ == "__main__":
    run(args.num_runs, args.mode, args.arch)

