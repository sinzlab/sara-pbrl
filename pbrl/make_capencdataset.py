import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import random
import torch
from pathlib import Path
import gym
import sys
sys.path.append(os.path.join(os.path.expanduser('~'),'rltransfer/PreferenceTransformer'))
import wrappers
import collections
import collections.abc

# Restore the old name so D4RL’s isinstance(...) check will work
collections.Mapping = collections.abc.Mapping

# now disable dm_control if you still want that
from d4rl.kitchen.adept_envs import mujoco_env
mujoco_env.USE_DM_CONTROL = False

import gym, d4rl
from d4rl import hand_manipulation_suite

from PreferenceTransformer.JaxPref.replay_buffer import get_d4rl_dataset
from PreferenceTransformer.JaxPref.new_preference_reward_main import create_preference_dataset
from PreferenceTransformer.JaxPref.sampler import TrajSampler
from PreferenceTransformer.JaxPref import reward_transform as r_tf

def set_seed(seed) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def remove_05_pref_dataset(data_dict,labelKey):
    labels = np.array(data_dict[labelKey])
    mask = ~(np.all(labels == [0.5, 0.5], axis=1))
    if sum(mask)!=len(labels): #if the number of True (ie no 05 rows) is equal to the number of labels, we don't need to filter anything
        filtered_dict={}
        for key, value in data_dict.items():
            filtered_dict[key] = value[mask]
    print(f"From preference dattaset removed {len(labels) - sum(mask)} rows.")
    return filtered_dict


def update_pref_datasets(data_folder,labelKey):


    # Iterate over all subdirectories
    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)

        # Check if the folder name contains "False" and is indeed a directory
        if "False" in folder and os.path.isdir(folder_path):
            print(folder)
            # Define file paths
            files_to_process = [
                os.path.join(folder_path, "pref_eval_dataset_False.pkl"),
                os.path.join(folder_path, "preference_dataset.pkl")
            ]

            for file_path in files_to_process:
                print(file_path)
                if os.path.exists(file_path):
                    # Load the pickle file
                    with open(file_path, "rb") as f:
                        data_dict = pickle.load(f)

                    # Get the indices where labels are [.5, .5]
                    labels = np.array(data_dict[labelKey])
                    mask = ~(np.all(labels == [0.5, 0.5], axis=1))
                    if sum(mask)!=len(labels): #if the number of True (ie no 05 rows) is equal to the number of labels, we don't need to filter anything
                        # Save a backup of the original file
                        old_file_path = file_path.replace(".pkl", "_OLD.pkl")
                        with open(old_file_path, "wb") as f:
                            pickle.dump(data_dict, f)


                        print(mask.shape)
                        filtered_dict={}
                        for key, value in data_dict.items():
                            if key in ['start_indices', 'start_indices_2']:
                                continue
                            filtered_dict[key] = value[mask]

                        # Save the updated dict back to the original file
                        with open(file_path, "wb") as f:
                            pickle.dump(filtered_dict, f)

                        print(f"Processed {file_path}, removed {len(labels) - sum(mask)} rows. Original saved as {old_file_path}.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--use05", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--scriptLabel", type=lambda x: x.lower() == "true", default=False)
    
    parser.add_argument("--mod_hopper", type=lambda x: x.lower() == "true", default=None)

    parser.add_argument("--mistake_rate", type=float, default=0.0)
    
    #defaults
    parser.add_argument("--num_query", type=int, default=500)
    parser.add_argument("--query_len", type=int, default=100)
    parser.add_argument("--train_split_size", type=float, default=0.8)
    parser.add_argument("--save_dir", type=str, default='/mnt/vast-react/projects/rl_pref_constraint/PbRL')
    parser.add_argument("--data_seed", type=int, default=3407)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.expanduser('~'),'rltransfer/PreferenceTransformer/human_label')) 
  

    return parser.parse_args()

def return_trajs(indices,preferredSuffix,notPreferredSuffix,datasetTorch):
    prefObsKey,prefActionKey='observations'+preferredSuffix,'actions'+preferredSuffix
    notPrefObsKey,notPrefActionKey='observations'+notPreferredSuffix,'actions'+notPreferredSuffix
    prefObs, prefAction= datasetTorch[prefObsKey][indices], datasetTorch[prefActionKey][indices]
    prefAgIdx=torch.zeros(prefObs.shape[:-1])+1
    prefAgIdx=prefAgIdx.unsqueeze(2)

    

    notPrefObs, notPrefAction= datasetTorch[notPrefObsKey][indices], datasetTorch[notPrefActionKey][indices]
    notPrefAgIdx=torch.zeros_like(prefAgIdx)

    

    maskShape=prefObs.shape[:-1]
    src_key_padding_mask=torch.zeros(maskShape,dtype=torch.bool).unsqueeze(2) # assumed not truncating, so set to False. Both trajs use the same mask (all False anyway)

    trajsPreferred=torch.cat([prefObs,prefAction,src_key_padding_mask, prefAgIdx],dim=2) #numTrajectories x stepMax(100) x obsDim+actionDim+3 (mask+agentId+taskId)
    trajsNotPreferred=torch.cat([notPrefObs,notPrefAction,src_key_padding_mask,notPrefAgIdx],dim=2)

    all_agent_trajectories=torch.cat([trajsPreferred,trajsNotPreferred],dim=0)
    return all_agent_trajectories

def modified_hopper_data(prefDataset):
    actions_repeated=np.tile(prefDataset['actions'],(1,1,2))
    prefDataset['actions']=actions_repeated

    actions_repeated2=np.tile(prefDataset['actions_2'],(1,1,2))
    prefDataset['actions_2']=actions_repeated2

    for key in ['observations', 'next_observations','observations_2', 'next_observations_2']:
        # Initialize newObs array 
        obs=prefDataset[key]
        newObs = np.zeros((obs.shape[0], obs.shape[1], 17))

        # Assign values based on the given rules
        newObs[:, :, 0:5] = obs[:, :, 0:5]    # Elements 0-4 remain the same
        newObs[:, :, 5:8] = obs[:, :, 2:5]    # Elements 5-7 = Elements 2-4 of obs
        newObs[:, :, 8:11] = obs[:, :, 5:8]   # Elements 8-10 = Elements 5-7 of obs
        newObs[:, :, 11:14] = obs[:, :, 8:11] # Elements 11-13 = Elements 8-10 of obs
        newObs[:, :, 14:17] = obs[:, :, 8:11] # Elements 14-16 = Elements 8-10 of obs

        prefDataset[key]=newObs
    return prefDataset

def create_mistakes(prefData, mistake_rate,labelKey):
    
    labels = prefData[labelKey]
    assert labels.shape[1] == 2, "Expected labels of shape (N, 2)"

    # Step 1: Find indices of hard labels (i.e., not [0.5, 0.5])
    hard_label_mask = ~np.isclose(labels, 0.5).all(axis=1)
    hard_indices = np.where(hard_label_mask)[0]

    # Step 2: Choose a random subset to flip
    num_to_flip = int(len(hard_indices) * mistake_rate)
    flip_indices = np.random.choice(hard_indices, size=num_to_flip, replace=False)

    # Step 3: Flip the labels (1 - label) for selected indices
    prefData[labelKey][flip_indices] = 1 - prefData[labelKey][flip_indices]

    return prefData

def make_capEncDataset_fromargs(args):
    set_seed(args.data_seed)
    ##this is identical to how the preference dataset is created in new_preference_reward_main, with the exception that we exclude code for the robosuite###

    if 'ant' in args.task:
        gym_env = gym.make(args.task)
        gym_env = wrappers.EpisodeMonitor(gym_env)
        gym_env = wrappers.SinglePrecision(gym_env)
        gym_env.seed(args.data_seed) #Note: PreferenceTransformer uses same seed for training the reward model rather than the data seed. I don't think this should need a seed anyway, but I changed it to data_seed in both here and PreferenceTransformer new_preference_reward_main 
        gym_env.action_space.seed(args.data_seed)
        gym_env.observation_space.seed(args.data_seed)
        dataset = r_tf.qlearning_ant_dataset(gym_env)
        label_type = 1
    else:
        gym_env = gym.make(args.task)
        eval_sampler = TrajSampler(gym_env.unwrapped, args.max_ep_len)
        dataset = get_d4rl_dataset(eval_sampler.env)
        label_type = 0

    dataset['actions'] = np.clip(dataset['actions'], -0.999, 0.999) #the clip action value was taken from PreferenceTransformer new_preference_reward_main.py, we won't change it
    
    print("load saved indices.")
    if 'dense' in args.task:
        env = "-".join(args.task.split("-")[:-2] + [args.task.split("-")[-1]])
    else:
        env = args.task
    base_path = os.path.join(args.data_dir, env)

    #pref_eval_dataset and true_eval is only used for the PreferenceTransformer (not my similarity rewards model). As we are using the full set of human queries available, the pref_eval_dataset is actually a subset of the preference_dataset and not a true eval dataset 
    #note: this is hacky, but I set use_human_label equal to True even if the script_label arg of this script is True. In subsequent steps, it is reassigned so that label is equal to script_label if args.script_label (could be cleaner if I just set use_human_label to be false), but I did it this way originally
    preference_dataset,pref_eval_dataset,true_eval=create_preference_dataset(base_path,gym_env,dataset,args.num_query,args.query_len,label_type,balance=False,use_human_label=True) #balance False and use_human_label true are defaults for Preference Transformer new_preference_reward main 
    
    if args.mod_hopper:
        preference_dataset=modified_hopper_data(preference_dataset)
        pref_eval_dataset=modified_hopper_data(pref_eval_dataset)

    ####get a random subset#####
    if args.fraction<1.0:
        k = int(args.fraction * args.num_query)
        indicesFracSample = np.random.choice(args.num_query, k, replace=False)
        preference_dataset = {key: value[indicesFracSample] for key, value in preference_dataset.items()}
    ###########################
    if args.scriptLabel:
        labelKey='script_labels'
    else:
        labelKey='labels'
    
    if args.mistake_rate>0.0:
        preference_dataset=create_mistakes(preference_dataset,args.mistake_rate,labelKey)
        pref_eval_dataset=create_mistakes(pref_eval_dataset,args.mistake_rate,labelKey)

    if not args.use05:
        preference_dataset=remove_05_pref_dataset(preference_dataset,labelKey)
        pref_eval_dataset=remove_05_pref_dataset(pref_eval_dataset,labelKey)

    datasetTorch={}
    for key in preference_dataset:
        datasetTorch[key]=torch.from_numpy(preference_dataset[key])

    if args.use05:
        indices10=np.union1d((preference_dataset[labelKey]==[.5,.5]).all(axis=1).nonzero()[0], (preference_dataset[labelKey]==[1,0]).all(axis=1).nonzero()[0]) #here we include both trajs where neither is preferred ([.5, .5]) in both agent types
    else:
        indices10=(preference_dataset[labelKey]==[1,0]).all(axis=1).nonzero()[0]
    trajs10=return_trajs(indices10,'','_2',datasetTorch=datasetTorch)

    indices01=(preference_dataset[labelKey]==[0,1]).all(axis=1).nonzero()[0]# #here we only include the [0,1] indices. If we included [.5,.5] again we would double count indices. 
    trajs01=return_trajs(indices01,'_2','',datasetTorch=datasetTorch)

    all_trajectories=torch.cat([trajs10,trajs01],dim=0)
    all_trajectories=all_trajectories[torch.randperm(all_trajectories.size()[0])]
    all_trajectories=all_trajectories.float()


    if args.train_split_size<1.0:
        train_set, test_set = train_test_split(all_trajectories, train_size=args.train_split_size, shuffle=True, random_state=args.data_seed)
    else:
        train_set=all_trajectories
        test_set=[]
    print('Num preference queries {}'.format(preference_dataset['observations'].shape[0]))
    print("Train set size {}".format(len(train_set)))
    print("Test set size {}".format(len(test_set)))

    if args.mod_hopper:
        folderName='Percent{}_05{}_modForWalker'.format(int(args.fraction*100),args.use05)
    else:
        folderName='Percent{}_05{}'.format(int(args.fraction*100),args.use05)
    if args.scriptLabel:
        folderName=folderName+'_scriptLabel'
        preference_dataset['labels']=preference_dataset['script_labels']
        pref_eval_dataset['labels']=pref_eval_dataset['script_labels']
    if args.mistake_rate>0.0:
        folderName=folderName+'_mistake{}'.format(int(args.mistake_rate*100))

    save_dir=os.path.join(args.save_dir,args.task,'Data',folderName)
    os.makedirs(save_dir,exist_ok=True)
    with open(os.path.join(save_dir,'train_set.pkl'), 'wb') as f:  # open a text file
        pickle.dump(train_set, f)
    with open(os.path.join(save_dir,'test_set.pkl'), 'wb') as f:  # open a text file
        pickle.dump(test_set, f)
    with open(os.path.join(save_dir,'preference_dataset.pkl'), 'wb') as f:  # open a text file
        pickle.dump(preference_dataset, f)
    with open(os.path.join(save_dir,'pref_eval_dataset_{}.pkl'.format(true_eval)), 'wb') as f:  # open a text file
        pickle.dump(pref_eval_dataset, f)



if __name__ == "__main__":
    args=get_args()
    make_capEncDataset_fromargs(args)