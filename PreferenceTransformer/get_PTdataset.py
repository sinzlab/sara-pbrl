import datetime
import os
import pickle
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn

from absl import app, flags


import sys
sys.path.append(os.path.join(os.path.expanduser('~'),'rltransfer/PreferenceTransformer'))
from dataset_utils import D4RLDataset, reward_from_preference, reward_from_preference_transformer
import wrappers

# Import Bradley-Terry inference function
sys.path.append('/mnt/vast-react/projects/rl_pref_constraint/SARA_PbRL/pbrl')
from train_bradley_terry_from_latents import bradley_terry_inference


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'hopper-medium-replay-v2', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_string('model_type', 'PrefTransformer', 'type of reward model.')
flags.DEFINE_string('ckpt_dir',
                    './logs/pref_reward',
                    'ckpt path for reward model.')
flags.DEFINE_integer('seq_len', 100, 'sequence length for relabeling reward in Transformer.')
flags.DEFINE_bool('use_diff', False, 'boolean whether use difference in sequence for reward relabeling.')
flags.DEFINE_string('label_mode', 'last', 'mode for relabeling reward with tranformer.')

def reward_from_bradley_terry(env_name, dataset, ckpt_dir, batch_size=256):
    """
    Replace dataset rewards with Bradley-Terry model predictions.
    
    Args:
        env_name: Environment name
        dataset: D4RL dataset object
        ckpt_dir: Path to Bradley-Terry checkpoint directory
        batch_size: Batch size for processing
    
    Returns:
        dataset: Dataset with replaced rewards
    """
    # Import split_into_trajectories from dataset_utils
    from dataset_utils import split_into_trajectories
    
    # Split dataset into trajectories (same as preference transformer)
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations
    )
    
    # Convert trajectories to the format expected by bradley_terry_inference
    trajectories = []
    trj_mapper = []
    
    for trj_idx, traj in enumerate(trajs):
        _obs, _act = [], []
        for _o, _a, _r, _m, _d, _no in traj:
            _obs.append(_o)
            _act.append(_a)
        
        traj_len = len(traj)
        _obs, _act = np.asarray(_obs), np.asarray(_act)
        trajectories.append((_obs, _act))
        
        for seg_idx in range(traj_len):
            trj_mapper.append((trj_idx, seg_idx))
    
    # Use bradley_terry_inference to get rewards for all trajectories
    all_traj_rewards = bradley_terry_inference(ckpt_dir, trajectories, env_name, batch_size=batch_size)
    
    # Now map the trajectory rewards back to the dataset format using trj_mapper
    # (same logic as reward_from_preference_transformer)
    data_size = dataset.rewards.shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset.rewards)
    
    for i in range(interval):
        start_pt = i * batch_size
        end_pt = min((i + 1) * batch_size, data_size)
        
        # For each timestep in this batch, get the corresponding reward
        batch_rewards = []
        for pt in range(start_pt, end_pt):
            _trj_idx, _seg_idx = trj_mapper[pt]
            # Get reward for this specific timestep from the trajectory rewards
            timestep_reward = all_traj_rewards[_trj_idx][_seg_idx]
            batch_rewards.append(timestep_reward)
        
        # Assign rewards back to dataset
        new_r[start_pt:end_pt] = np.array(batch_rewards)
    
    # Replace dataset rewards
    dataset.rewards = new_r.copy()
    
    return dataset

def make_env_and_dataset(env_name, seed, ckpt_dir, model_type='PrefTransformer', batch_size=256, seq_len=100, use_diff=False, label_mode='last') :
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if model_type=="GroundTruth":
        return dataset
    elif model_type=='BradleyTerryContrastive':
        dataset = reward_from_bradley_terry(env_name, dataset, ckpt_dir, batch_size=batch_size)
        return dataset
    else:    
        reward_model = initialize_model(ckpt_dir)
        if model_type == "MR":
            dataset = reward_from_preference(env_name, dataset, reward_model, batch_size=batch_size)
        elif model_type=='PrefTransformer':
            dataset = reward_from_preference_transformer(
                env_name,
                dataset,
                reward_model,
                batch_size=batch_size,
                seq_len=seq_len,
                use_diff=use_diff,
                label_mode=label_mode
                )
        return dataset


def initialize_model(ckpt_dir):
    if os.path.exists(os.path.join(ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(ckpt_dir, "model.pkl")
    
    #a very hacky work around that can be deleted later. We had run the preftrans reward model using numpy 2.2.3. But the pbrl container must have a lower version of numpy (1.26.4) or error results on d4rl import. But this lower version of python is missing np.bool attribute which causes issue on pickle.load
    # try:
    #     with open(model_path, "rb") as f:
    #         ckpt = pickle.load(f)
    # except Exception:
    #     np.bool=np.bool_
    #     with open(model_path, "rb") as f:
    #         ckpt = pickle.load(f)
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    
    reward_model = ckpt['reward_model']
    return reward_model


def main(_):

    dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.ckpt_dir, FLAGS.model_type, FLAGS.batch_size, FLAGS.seq_len, FLAGS.use_diff, FLAGS.label_mode)
    

    

    

if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    app.run(main)
