import datetime
import os
import pickle
from typing import Tuple
###needed for kitchen
#import gym
import collections
import collections.abc

# Restore the old name so D4RL’s isinstance(...) check will work
collections.Mapping = collections.abc.Mapping

# now disable dm_control if you still want that
from d4rl.kitchen.adept_envs import mujoco_env
mujoco_env.USE_DM_CONTROL = False

import gym, d4rl
from d4rl import hand_manipulation_suite
#######
import numpy as np
import absl
from collections import deque

import wrappers
from evaluation import evaluate
from learner import Learner

from viskit.logging import logger, setup_logger
from JaxPref.utils import WandBLogger, define_flags_with_default, get_user_flags, \
    set_random_seed, Timer, prefix_metrics

from JaxPref.dataset_utils import PrefD4RLDataset

from JaxPref.PrefTransformer import PrefTransformer
import wandb
import faulthandler; faulthandler.enable()

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

FLAGS_DEF = define_flags_with_default(
    env_name='halfcheetah-medium-v2',
    seed=42,
    tqdm=True,
    eval_episodes=10,
    log_interval=1000,
    eval_interval=5000,
    batch_size=256,
    max_steps=int(1e6),
    model_type="PrefTransformerDPPO",
    seq_len=100,
    min_seq_len=0,
    dropout=0.25, #HP from paper for gym 

    lambd=0.5, #HP from paper for gym 
    dist_temperature=0.1,
    logging=WandBLogger.get_default_config(),

    # params for loading preference transformer
    ckpt_base_dir="/mnt/vast-react/projects/rl_pref_constraint/PbRL/",
    ckpt_type="last",
    transformer=PrefTransformer.get_default_config(),
    smooth_sigma=20.0,
    smooth_in=True,
    dataset_name='',
    
)
    
FLAGS = absl.flags.FLAGS

def get_completed_run_name():
    project='PrefTransformerDPPO_{}'.format(FLAGS.env_name)
    group='{}_seed{}_FakeEval'.format(FLAGS.dataset_name,FLAGS.seed)
    if 'script' in FLAGS.dataset_name:
        project +='_scriptLabel'
    if 'mistake' in FLAGS.dataset_name:
        project += '_error'
    
    # Initialize the API client
    api = wandb.Api()
    
    
    filters = {
        "state": "finished",
        "group": group
    }

    runs = api.runs(project, filters=filters)

    if not runs:
        raise ValueError("No finished runs found for that {} & {}.".format(project,group))


    if len(runs)>1:
        raise ValueError("More than 1 finished run for {} & {}.".format(project,group))
    return runs[0].name

    



def initialize_model():
    run_name=get_completed_run_name()
    ckpt_dir = os.path.join(FLAGS.ckpt_base_dir, FLAGS.env_name, FLAGS.model_type, FLAGS.dataset_name, "seed{}".format(FLAGS.seed),run_name)
    if FLAGS.ckpt_type == "best":
        model_path = os.path.join(ckpt_dir, "best_model.pkl")
    elif FLAGS.ckpt_type == "last":
        model_path = os.path.join(ckpt_dir, "model.pkl")
    else:
        raise NotImplementedError

    print("Loading score model from", model_path)
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']
    return reward_model

def make_env_and_dataset(env_name: str,
                         seed: int,
                         ) -> Tuple[gym.Env, PrefD4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    reward_model = initialize_model()

    dataset = PrefD4RLDataset(
        env=env,
        seq_len=FLAGS.seq_len,
        min_seq_len=FLAGS.min_seq_len,
        reward_model=reward_model,
    )

    return env, dataset


def main(_):
    VARIANT = get_user_flags(FLAGS, FLAGS_DEF)
    policy_run_name=datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S") 
    FLAGS.logging.output_dir = os.path.join(FLAGS.ckpt_base_dir, FLAGS.env_name,"IQL", FLAGS.dataset_name, 'DPPOPolicy',"seed{}".format(FLAGS.seed),policy_run_name) #though it's not IQL, I put it in this folder structure just for comparison with SimRewards and PT
    FLAGS.logging.project='IQL_{}'.format(FLAGS.env_name) #likewise for the project, it's not an IQL policy but put it here so we can easily compare in wandb
    if 'script' in FLAGS.dataset_name:
        FLAGS.logging.project +='_scriptLabel'
    if 'mistake' in FLAGS.dataset_name:
        FLAGS.logging.project += '_error'

    FLAGS.logging.group = "{}_{}".format(FLAGS.dataset_name,'DPPOPolicy')


    FLAGS.logging.experiment_id = 'seed{}_{}'.format(FLAGS.seed,policy_run_name)
    

    save_dir = FLAGS.logging.output_dir

    setup_logger(
        variant=VARIANT,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False
    )

    
    wb_logger = WandBLogger(FLAGS.logging, variant=VARIANT)

    set_random_seed(int(FLAGS.seed))

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    lambd=FLAGS.lambd,
                    dist_temperature=FLAGS.dist_temperature,
                    dropout_rate=FLAGS.dropout if (FLAGS.dropout > 0) else None,
                    )

    last_8_performance_every5 = deque(maxlen=8)
    for i in range(FLAGS.max_steps + 1):
        metrics = dict()
        metrics["step"] = i
        with Timer() as timer:
            batch = dataset.sample(FLAGS.batch_size)
            train_info = prefix_metrics(agent.update(batch), 'train')

            if i % FLAGS.log_interval == 0:
                for k, v in train_info.items():
                    metrics[k] = v

            if i % FLAGS.eval_interval == 0:
                eval_info = prefix_metrics(evaluate(agent, env, FLAGS.eval_episodes), 'eval')

                last_8_performance_every5.append(eval_info['eval/return_list'])
                last8Flattened = [env.get_normalized_score(value) * 100 for sublist in last_8_performance_every5 for value in sublist]
                if len(last8Flattened)<8*FLAGS.eval_episodes:
                    running_epRewards_mean=0.0
                    running_epRewards_std=0.0
                else:
                    if len(last8Flattened)>8*FLAGS.eval_episodes:
                        raise ValueError("Too many evaluation episodes")
                    running_epRewards_mean=np.mean(last8Flattened)
                    running_epRewards_std=np.std(last8Flattened)
                metrics['eval/maxReward']=np.max(last8Flattened)
                metrics['eval/minReward']=np.min(last8Flattened)
                metrics['eval/running_epRewards_mean']=running_epRewards_mean
                metrics['eval/running_epRewards_std']=running_epRewards_std
                for k, v in eval_info.items():
                    metrics[k] = v
        
        keep=("step",'eval/maxReward','eval/minReward','eval/running_epRewards_mean','eval/running_epRewards_std','eval/normalized_episode_reward','eval/normalized_episode_reward_std','eval/episode_length','eval/episode_length_std','eval/success')
        wandbMetrics={k: metrics[k] for k in keep if k in metrics}
        if len(metrics) > 1: # has something to log
            metrics["time"] = timer()
            logger.record_dict(metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=True)
            wb_logger.log(wandbMetrics, step=int(i/1000))

    # save model
    agent.actor.save(os.path.join(save_dir, "model.pkl"))


if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    absl.app.run(main)
