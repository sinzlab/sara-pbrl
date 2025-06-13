import datetime
import os
import pickle
from typing import Tuple

import gym
import numpy as np

from absl import app, flags


import sys
sys.path.append(os.path.join(os.path.expanduser('~'),'rltransfer/PreferenceTransformer'))
from dataset_utils import D4RLDataset, reward_from_preference, reward_from_preference_transformer
import wrappers


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
