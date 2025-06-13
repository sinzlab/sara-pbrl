import sys
import os
# sys.path.append(os.path.expanduser('~'))
# sys.path.append(os.path.join(os.path.expanduser('~'),'rltransfer'))
# sys.path.append(os.path.join(os.path.expanduser('~'),'rltransfer','url_benchmark'))
# sys.path.append(os.path.join(os.path.expanduser('~'),'rltransfer','OfflineRL-Kit'))
# sys.path.append(os.path.join(os.path.expanduser('~'),'rltransfer','cpl-human/d4rl'))


import d4rl


import url_benchmark
from url_benchmark.pbrl.CottonDecoderV2pbrl import TransformerForInference
from collections import OrderedDict
import datetime
import gym
import os
import pickle
import PreferenceTransformer
from PreferenceTransformer.dataset_utils import split_into_trajectories  
import wandb


from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

def get_best_sim(latents,evalLatents,seqLenList,agent,max_ep_len,use_max=True):
    #latents is shape batch_size, max_ep_len, embedDim
    #evalLatents is shape (1,embedDim)
    #seqLenList is a list of sequence lengths
    cos=torch.nn.CosineSimilarity(dim=2)
    simValues=[]
    for seqLen in seqLenList:
        simValue=cos(latents,evalLatents['seqLen{}'.format(seqLen)][agent].repeat(max_ep_len,1)) #batch_size by max_ep_len
        simValues.append(simValue)

    # Stack along a new dimension (seqLen dimension) and compute max over that dimension
    simValues = torch.stack(simValues, dim=0)  # Shape: (num_seqLen, batch_size, max_ep_len)
    if use_max:
        best_simValues, _ = torch.max(simValues, dim=0)  # Max over seqLen dimension
    else:
        best_simValues, _ = torch.min(simValues, dim=0)
    return best_simValues

#almost same as the PreferenceTransformer (see train_offline.py). Just changed dataset to be dict instead of the dataset class that the PreferenceTransformer uses
def normalize(dataset, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(dataset['observations'], dataset['actions'],
                                    dataset['rewards'], dataset['masks'],
                                    dataset['terminals'],
                                    dataset['next_observations'])
    trj_mapper = []
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        traj_len = len(traj)

        for _ in range(traj_len):
            trj_mapper.append((trj_idx, traj_len))

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(sorted_trajs[-1])

    normalized_rewards = []
    for i in range(dataset['rewards'].shape[0]):
        _reward = dataset['rewards'][i]
        if 'antmaze' in env_name:
            _, len_trj = trj_mapper[i]
            _reward -= min_return / len_trj
        _reward /= max_return - min_return
        # if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)

    dataset['rewards'] = np.array(normalized_rewards)
    return dataset

def make_offline_dataset(cfg):
    task=cfg['task']
    capacityEncoderFilepath=cfg['capacityEncoderFilepath']
    device = torch.device('cuda')

    # save_dir=cfg['save_dir']
    # save_dir = Path(save_dir)
    # save_dir.mkdir(exist_ok=True, parents=True)
    env = gym.make(task)
    eps= 1e-5 #default taken from Preference Transformer
    clip_to_eps=True #default taken from Preference Transformer
    dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                            dataset['next_observations'][i]
                            ) > 1e-5 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    dones_float[-1] = 1
    dataset['observations']=dataset['observations'].astype(np.float32)
    dataset['actions']=dataset['actions'].astype(np.float32)
    dataset['rewards']=dataset['rewards'].astype(np.float32)
    dataset['masks']=1.0-dataset['terminals'].astype(np.float32)
    dataset['terminals']=dones_float.astype(np.float32)
    dataset['next_observations']=dataset['next_observations'].astype(
                            np.float32)

    trajs = split_into_trajectories(
        dataset['observations'],
        dataset['actions'],
        dataset['rewards'],
        dataset['masks'], #masks
        dataset['terminals'],
        dataset['next_observations']
    ) #list of length num trajectories. Each trajectory is a list of tuples (number of tuples is the trajectory length). Each tuple is obs, action, rewards, masks, dones, next_obs


    max_ep_len=cfg['max_ep_len']
    obsDim=dataset['observations'].shape[1]
    actionDim=dataset['actions'].shape[1]
    dataForEncoder = torch.zeros((len(trajs),max_ep_len,obsDim+actionDim+1)) #the +1 is for the mask 
    dataset_idx_mapper = torch.zeros(len(trajs),max_ep_len)

    dataset_idx=0
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        _obs=torch.zeros((len(traj),obsDim))
        _action=torch.zeros((len(traj),actionDim))
        
        for stepIdx,transitionTuple in enumerate(traj):
            _o, _a, _r, _m, _d, _no=transitionTuple
            _obs[stepIdx,:]=torch.tensor(_o)
            _action[stepIdx,:]=torch.tensor(_a)
            dataset_idx_mapper[trj_idx,stepIdx]=dataset_idx #need this to go back to the buffer of transitions
            dataset_idx+=1
        
        epLen=_obs.shape[0]
        dataForEncoder[trj_idx,:epLen,:obsDim]=_obs
        dataForEncoder[trj_idx,:epLen,obsDim:obsDim+actionDim]=_action
        #src key padding mask: 0s where we have real data, 1 from onwards
        columns = torch.arange(max_ep_len)
        src_key_padding_maskCol = columns >= epLen 
        dataForEncoder[trj_idx,:,-1]=src_key_padding_maskCol
        ######################


    with open(os.path.join(capacityEncoderFilepath,'cfg.pkl'), 'rb') as f:
        capEncCfg = pickle.load(f) 

    
    capacity_inference=TransformerForInference(capEncCfg, causal_pool1=cfg['causal_pool1'],causal_pool2=cfg['causal_pool2'],window_size=cfg['windowRewards'],device='cuda').to(device)
    capacity_inference.eval()

    with open(os.path.join(capacityEncoderFilepath,'train_set.pkl'), 'rb') as f:
        train_set = pickle.load(f)
    
    trainDataloader=DataLoader(train_set, batch_size=128, shuffle=False) #creating dataloader only because the relevant function (getEvalLatents) requires it as input, but will just be using the full dataset in that function. Thus the batch size we put here doesn't actually matter
    #we get eval latents on the full training set

    #get fixed latents from the two classes of agents (preferred and not preferred)
    capacityEncoder=url_benchmark.pbrl.ContrastiveCapacityEncoderV2pbrl.CapacityEncoderV2(capEncCfg,train_set=train_set,test_set=[],use_wandb=False) #the train set and test set must be passed in to initialize the class but we don't use it here. 
    capacityEncoder.capacity_encoder.load_state_dict(torch.load(os.path.join(capacityEncoderFilepath,'capacityEncoder.pt'), weights_only=True))
    capacityEncoder.capacity_encoder.eval()
    capacityEncoder=capacityEncoder
    allAgentLatents={} #only used in policy eval rollouts

    if cfg['use_vary_seqLens']:
        for seqLen in capEncCfg['seqLenList']:
            allAgentLatents['seqLen{}'.format(seqLen)]={}
            for i in range(len(capacityEncoder.agent_list)):
                latentsFullAgent=capacityEncoder.getEvalLatents(i,trainDataloader,seqLen=seqLen).clone().detach()
                if i==0:
                    agent='unpreferred'
                if i==1:
                    agent='preferred'
                allAgentLatents['seqLen{}'.format(seqLen)][agent]=latentsFullAgent
    else:
        for i in range(len(capacityEncoder.agent_list)):
            latents=capacityEncoder.getEvalLatents(i,trainDataloader).clone().detach()
            if i==0:
                agent='unpreferred'
            if i==1:
                agent='preferred'
            allAgentLatents[agent]=latents
    ###########################################

    ##### get capacity latents######
    dataForEncoder=dataForEncoder.to(device)

    encoderDataloader=DataLoader(dataForEncoder, batch_size=64, shuffle=False)



    cos=torch.nn.CosineSimilarity(dim=2)
    for i, encoderBatch in enumerate(encoderDataloader):
       
        batchForCapacity=encoderBatch[:,:,:capacity_inference.inputDim] #needed for the capacity latent inference, not distinguishing by agent type
        batchMask=encoderBatch[:,:,obsDim+actionDim] #padding mask, needed for both the decoder loss and the capacity latent inference 
        latents_batch=capacity_inference.forward(batchForCapacity, src_key_padding_mask=batchMask, use_src_mask=cfg['src_mask_decoder']).detach() #size is batch_size, max_ep_len, embedDim
        
        if not cfg['use_vary_seqLens']:
            simWUnpreferred=cos(latents_batch,allAgentLatents['unpreferred'].repeat(max_ep_len,1)) #
            simWPreferred=cos(latents_batch,allAgentLatents['preferred'].repeat(max_ep_len,1)) 
        else:
            simWPreferred=get_best_sim(latents_batch,allAgentLatents,capEncCfg['seqLenList'],'preferred',max_ep_len=max_ep_len,use_max=True) 
            simWUnpreferred=get_best_sim(latents_batch,allAgentLatents,capEncCfg['seqLenList'],'unpreferred',max_ep_len=max_ep_len,use_max=False)
        new_rewards_batch=cfg['alpha']*simWPreferred-cfg['beta']*simWUnpreferred #batch_size by max_ep_len 
        if i==0:
            new_rewards=new_rewards_batch
        else:
            new_rewards=torch.cat([new_rewards,new_rewards_batch],dim=0)

        
        # if i==0:
        #     capacityLatents=latents_batch
        # else:
        #     capacityLatents=torch.cat([capacityLatents,latents_batch],dim=0)
        
    # capacityLatents=capacityLatents.detach().cpu()
    # with open(os.path.join(cfg['save_dir'],'capacityLatents.pkl'), 'wb') as f: 
    #     pickle.dump(capacityLatents, f)

    ####################################



    new_rewards=new_rewards.detach().cpu().numpy() #shape number of trajectories, max_ep_len, this will include rewards for timesteps of a trajectory that were padded on. But we'll take care of that in rematching to the dataset
        


    newRewArray=np.zeros(dataset['terminals'].shape)
    if new_rewards.shape!=dataset_idx_mapper.shape:
        raise ValueError("New rewards and dataset idx mapper should have same shape")
    for trj_idx in range(dataset_idx_mapper.shape[0]):
        firstIdxInRewArr=int(dataset_idx_mapper[trj_idx][0].detach().item()) 
        lastNonzeroIdx=dataset_idx_mapper[trj_idx].nonzero()[-1].item()
        lastIdxInRewArr=int(dataset_idx_mapper[trj_idx][lastNonzeroIdx].detach().item()) #take the last nonzero index then get the value of dataset_idx_mapper at that idx. That is the last index for the corresponding trajectory in the qlearning dataset
        newRewArray[firstIdxInRewArr:lastIdxInRewArr+1]=new_rewards[trj_idx][:lastNonzeroIdx+1]
        
    #check that no elements in newRewArray are zero
    if sum(newRewArray==0)>5: #allow a handful for testing
        print("Num rew 0 is {}".format(sum(newRewArray==0)))
        if cfg['alpha']!=0 or cfg['beta']!=0: #if both weights are 0 then it's expected to have all zero rewards, but if either weight is nonzero then we should not have zero rewards anywehre
            raise ValueError("reward assignment is faulty somewhere. Should not have zero rewards")

    dataset['rewards']=newRewArray #assign new rewards

    #now need to do reward normalization, this follows the PreferenceTransformer code
    dataset=normalize(dataset, task, max_episode_steps=max_ep_len)
    if 'antmaze' in task:
        dataset['rewards'] -= 1.0
    if ('halfcheetah' in task or 'walker2d' in task or 'hopper' in task):
        dataset['rewards'] += 0.5
    
    # with open(os.path.join(cfg['save_dir'],'offline_dataset_modifiedreward.pkl'), 'wb') as f: 
    #     pickle.dump(dataset, f)
    return dataset


def main():
    
    capacityEncoderFilepath='/user/rajaram/u13657/103727'#'/home/srajaram/rltransfer/exp_local/2025.02.01/_HopperMediumReplayCapacityNo05_smallerdims/103727'

    
    cfg={'capacityEncoderFilepath':capacityEncoderFilepath, 'task':'hopper-medium-replay-v2', 'max_ep_len':1000}
    cfg['causal_pool1']=True
    cfg['causal_pool2']=False
    cfg['alpha']=.5
    cfg['beta']=.5
    # currDateTime=datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")
    # dmod='OfflineDataset'
    
    #filepath=os.path.join(os.path.expanduser('~'),'rltransfer/exp_local/',currDateTime.split('_')[0],"HopperMediumReplayCapacity"+dmod)
    # cfg['save_dir']=filepath
    make_offline_dataset(cfg)
                    


if __name__ == '__main__':
    main()
    