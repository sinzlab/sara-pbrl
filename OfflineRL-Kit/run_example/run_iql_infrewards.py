import argparse
import random

# import gym
# import d4rl

import collections
import collections.abc

# Restore the old name so D4RL’s isinstance(...) check will work
collections.Mapping = collections.abc.Mapping

# now disable dm_control if you still want that
from d4rl.kitchen.adept_envs import mujoco_env
mujoco_env.USE_DM_CONTROL = False
import gym, d4rl
from d4rl import hand_manipulation_suite
import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, DiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import IQLPolicy
import pickle
import os
import datetime

import sys

from pbrl.make_offlinedataset import make_offline_dataset, normalize
from PreferenceTransformer.get_PTdataset import make_env_and_dataset
"""
suggested hypers
expectile=0.7, temperature=3.0 for all D4RL-Gym tasks
All the defaults for the IQL training (dropout, tau, gamma, actor crtic lrs, hidden dims, etc) are defaults provided with this package as well as those using in PreferenceTransformer model, according to the mujoco_config on the github
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="iql")
    parser.add_argument("--task", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-q-lr", type=float, default=3e-4)
    parser.add_argument("--critic-v-lr", type=float, default=3e-4)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--lr-decay", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    #added for inferred rewards with my model
    parser.add_argument("--simWRestrictedWeight", type=float, default=None)
    parser.add_argument("--simWUnrestrictedWeight", type=float, default=None)
    parser.add_argument("--causal_pool1", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--causal_pool2", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--capacityEncoderFilepath", type=str, default=None) 
    parser.add_argument("--windowRewards",type=lambda x: int(x) if x.lower() != "none" else None, default=None)#parser.add_argument("--windowRewards", type=int, default=None)
    parser.add_argument("--src_mask_decoder", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_vary_seqLens", type=lambda x: x.lower() == "true", default=False)
    

    #added for Preference Transformer Rewards
    parser.add_argument("--PrefTrans_ckpt_dir", type=str, default=None) #the checkpoint of the trained reward model from PreferenceTransformer, down to level (but not including) the folder seed name. This will be added below to match the seed input

    #specify whether we use my similarity rewards model or Preference Transformer model 
    parser.add_argument("--reward_model", type=str, default='SimilarityRewards') #either SimilarityRewards or PreferenceTransformer

    #specify save dir
    parser.add_argument("--save_dir", type=str, default='/mnt/vast-react/projects/rl_pref_constraint/PbRL')
    parser.add_argument("--dataset_name", type=str, default='dataset_name')
    parser.add_argument("--job_type", type=str, default='')

    return parser.parse_args()

def make_PT_dataset(env_name,seed,ckpt_dir,max_ep_len):
    dataset=make_env_and_dataset(env_name=env_name, seed=seed, ckpt_dir=ckpt_dir,model_type='PrefTransformer')
    #note that make_env_and_dataset both replaces the rewards with PT rewards from teh trained model and it inputs into the D4RLDataset class. In the init of the D4RL dataset class, a few preprocessing steps (done_floats and clip_to_eps) occur, which is the same as what we use when we make the offline dataset for our model
    #this d4rl dataset now needs to be converted to the dict dataset and normalized in accordance to what we use for the offline rl toolkit

    datasetOffline={}
    datasetOffline['observations']=dataset.observations
    datasetOffline['actions']=dataset.actions
    datasetOffline['rewards']=dataset.rewards
    datasetOffline['masks']=dataset.masks
    datasetOffline['terminals']=dataset.dones_float
    datasetOffline['next_observations']=dataset.next_observations

    datasetOffline=normalize(datasetOffline, env_name, max_episode_steps=max_ep_len)
    if 'antmaze' in env_name:
        datasetOffline['rewards'] -= 1.0
    if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        datasetOffline['rewards'] += 0.5

    return datasetOffline

def make_GT_dataset(env_name,seed,max_ep_len):
    dataset=make_env_and_dataset(env_name=env_name, seed=seed, ckpt_dir=None,model_type="GroundTruth")
    #same as above for PT dataset, except we pass in model_type GroundTruth, so it does not replace rewards with PT rewards but rather uses the original task rewards. Then normalization proceeds the same
    #this d4rl dataset now needs to be converted to the dict dataset and normalized in accordance to what we use for the offline rl toolkit

    datasetOffline={}
    datasetOffline['observations']=dataset.observations
    datasetOffline['actions']=dataset.actions
    datasetOffline['rewards']=dataset.rewards
    datasetOffline['masks']=dataset.masks
    datasetOffline['terminals']=dataset.dones_float
    datasetOffline['next_observations']=dataset.next_observations

    datasetOffline=normalize(datasetOffline, env_name, max_episode_steps=max_ep_len)
    if 'antmaze' in env_name:
        datasetOffline['rewards'] -= 1.0
    if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        datasetOffline['rewards'] += 0.5

    return datasetOffline


def make_offlinedataset_fromargs(cfg,reward_model):
    if reward_model=="GroundTruth":
        dataset=make_GT_dataset(env_name=cfg['task'],seed=cfg['seed'],max_ep_len=cfg['max_ep_len'])
    elif reward_model=='SimilarityRewards':
        dataset=make_offline_dataset(cfg)
    elif reward_model=='PreferenceTransformer':
        dataset=make_PT_dataset(env_name=cfg['task'],seed=cfg['seed'],ckpt_dir=cfg['PrefTrans_ckpt_dir'],max_ep_len=cfg['max_ep_len'])
    else:
        raise ValueError("Reward Model must be either SimilarityRewards or PreferenceTransformer")

    return dataset

def set_seed(seed) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Ensures deterministic cuBLAS behavior

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)  # Ensure deterministic algorithms
    torch.set_float32_matmul_precision('high')


    print(f"Random seed set as {seed}")

def train(args=get_args()):
    if args.seed==-1:
        raise ValueError("Must specify seed used") #the seed used should match the seed from training the reward model. If it is not specified then throw error
    # create env and dataset
    env = gym.make(args.task)

        
    # seed
    set_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    print("Making offline dataset with weights {}, {}, {}, {}".format(args.simWRestrictedWeight,args.simWUnrestrictedWeight,args.causal_pool1,args.causal_pool2))
    cfg={'task':args.task, 'max_ep_len':args.max_ep_len}
    if args.capacityEncoderFilepath is not None:
        if 'seed{}'.format(args.seed) in args.capacityEncoderFilepath:
            cfg['capacityEncoderFilepath']=args.capacityEncoderFilepath
            cfg['causal_pool1']=args.causal_pool1
            cfg['causal_pool2']=args.causal_pool2
            cfg['alpha']=args.simWRestrictedWeight
            cfg['beta']=args.simWUnrestrictedWeight
            cfg['windowRewards']=args.windowRewards
            cfg['src_mask_decoder']=args.src_mask_decoder
            cfg['use_vary_seqLens']=args.use_vary_seqLens
            
        else:
            raise ValueError("Seed value given and Capacity Encoder filepath do not match")

    cfg['seed']=args.seed
    
    if args.PrefTrans_ckpt_dir is not None:
        if 'seed{}'.format(args.seed) in args.PrefTrans_ckpt_dir:
            cfg['PrefTrans_ckpt_dir']=args.PrefTrans_ckpt_dir
        else:
            raise ValueError("Seed value given and Preference Transformer filepath do not match")

    dataset=make_offlinedataset_fromargs(cfg,args.reward_model)
    
    #dataset['rewards']=np.floor(dataset['rewards'] * 1e5) / 1e5 #remove this and in cap enc use fullqueryset

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]



    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate)
    critic_q1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_q2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_v_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic_q1 = Critic(critic_q1_backbone, args.device)
    critic_q2 = Critic(critic_q2_backbone, args.device)
    critic_v = Critic(critic_v_backbone, args.device)
    
    for m in list(actor.modules()) + list(critic_q1.modules()) + list(critic_q2.modules()) + list(critic_v.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=args.critic_q_lr)
    critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=args.critic_q_lr)
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_v_lr)

    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None
    
    # create IQL policy
    policy = IQLPolicy(
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optim,
        critic_q1_optim,
        critic_q2_optim,
        critic_v_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature=args.temperature
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # log
    timestamp=datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")
    if args.reward_model!="GroundTruth":
        save_dir=os.path.join(args.save_dir,args.task,'IQL',args.dataset_name, args.reward_model, 'seed{}'.format(args.seed),timestamp)
    else:
        save_dir=os.path.join(args.save_dir,args.task,'IQL', args.reward_model, 'seed{}'.format(args.seed),timestamp)
    os.makedirs(save_dir)
    log_dirs = save_dir
    # key: output file name, value: output handler type
    cfg['save_dir']=save_dir #for wandb logging 
    if 'scriptLabel' in args.dataset_name:
        projectName='{}_{}'.format(args.task,'scriptLabel')
    else:
        projectName='{}'.format(args.task)
    
    if 'mistake' in args.dataset_name:
        projectName += '_error'
    if args.reward_model!="GroundTruth":
        groupName='{}_{}'.format(args.dataset_name, args.reward_model)
    else:
        groupName=args.reward_model
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard",
        "wandb": {'project':'IQL_{}'.format(projectName),'group':groupName,'name':'seed{}_{}'.format(args.seed,timestamp),'job_type':args.job_type,'cfg':cfg, 'wandb_dir':save_dir}
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()