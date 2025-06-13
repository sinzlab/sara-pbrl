from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tqdm import trange


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'normalized_episode_reward': [], 'episode_length': [], 'success': []}
    
    # for _ in trange(num_episodes, desc='evaluation', leave=False):
    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            if k=='normalized_episode_reward':
                infoKey='return'
            elif k=='episode_length':
                infoKey='length'
            else:
                infoKey='success'
            stats[k].append(info['episode'][infoKey]) #this is temporarly assigning normalized ep reward to the return list, which is not yet normalized
    
    return_list=stats['normalized_episode_reward']
    episode_len_list=stats['episode_length']
    
    stats['return_list']=return_list #preserving the return list so that we can compute running average in train.py. Note this isn't normalized yet
    
    return_list_normalized=[env.get_normalized_score(rewVal) * 100 for rewVal in  return_list] #converting it to a normalized score
    print("#######")
    print(return_list_normalized)
    stats['normalized_episode_reward']=np.mean(return_list_normalized)
    stats['normalized_episode_reward_std']=np.std(return_list_normalized)
    
    stats['episode_length']=np.mean(episode_len_list)
    stats['episode_length_std']=np.std(episode_len_list)
    
    stats['success']=np.mean(stats['success'])
      
    return stats
