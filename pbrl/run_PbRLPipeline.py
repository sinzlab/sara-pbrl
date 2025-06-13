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

from PreferenceTransformer.JaxPref.replay_buffer import get_d4rl_dataset
from PreferenceTransformer.JaxPref.new_preference_reward_main import create_preference_dataset
from PreferenceTransformer.JaxPref.sampler import TrajSampler
from PreferenceTransformer.JaxPref import reward_transform as r_tf
from pbrl.ContrastiveCapacityEncoderpbrl import set_seed,CapacityEncoderV2
import datetime
import wandb
import copy
import subprocess 
import json
import yaml
os.environ['WANDB_INIT_TIMEOUT'] = '600'





#rerun to see if reproduce results
#make data utilities file for functions that are shared
#make readme

def get_run_max_epochs(project,group):
    #get runs that are finished for the given project and
    api = wandb.Api()
    runs = api.runs(
        project,
        filters={
            "group": {"$regex": group},
            "state": "finished"
        }
    )

    # Get the run with the maximum number of completed epochs
    # Replace "epoch" with the actual key if it's different (e.g., "final_epoch")
    max_epoch_run = max(
        (run for run in runs if "epoch" in run.summary),
        key=lambda run: run.summary.get("epoch", 0),
        default=None
    )
    if max_epoch_run:
        return max_epoch_run
    else:
        raise ValueError("No finished runs for {} and {}".format(project,group))


def has_completed_run_with_config(project: str, group: str, target_config: dict) -> bool:
    """
    Check if a completed run in the given project and group exists with the specified config,
    ignoring 'filepath' and 'exp_name' and 'device' keys.
    
    :param project: The W&B project name.
    :param group: The W&B group name.
    :param target_config: The configuration dictionary to match.
    :return: True if a completed run matches the config (excluding 'filepath' and 'exp_name'), else False.
    """
    api = wandb.Api()

    # Remove ignored keys from the target config
    target_config_filtered = {k: v for k, v in target_config.items() if k not in {"filepath", "exp_name", 'device'}}

    # Build W&B Mongo-style filters
    filters = {
        "group": group,
        "state": "finished",  # Only fetch completed runs
    }
    
    # Add config filters dynamically
    for key, value in target_config_filtered.items():
        filters[f"config.{key}"] = value

    try:
        # Fetch runs using the optimized filter
        runs = api.runs(path=project, filters=filters)
        # If there is at least one matching run, return True
        all_runs=[run.name for run in runs]
        return sorted(all_runs, reverse=True)
    except wandb.errors.CommError:
        return []
    except ValueError as e:
        return []

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hopper-medium-replay-v2")    
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--use05", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--mistake_rate", type=float, default=0.0) 
    parser.add_argument("--run_type", type=str, default=None) #run_IQL_CapEnc, run_IQL_PrefTrans, run_IQL_PrefTransADT
    parser.add_argument("--script_label", type=lambda x: x.lower() == "true", default=False) #to use reward models run using the script labels
    
    
    parser.add_argument("--enc_epochs", type=int, default=4000) 
    parser.add_argument("--windowRewards",type=lambda x: int(x) if x.lower() != "none" else None, default=None)
    parser.add_argument("--src_mask_decoder", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_vary_seqLens", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--causal_pool",type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=None) #specify seed for IQL run. If not specified then will run over the 8 seeds in a for loop
    parser.add_argument("--enc_configpath", type=str, default=None) #Not mandatory to specify but rather only specify this if running IQL with a preferred capacity encoder congif. This will run over seeds (or seed if args.seed specified) 
    parser.add_argument("--jobInfo", type=str, default='')


    parser.add_argument("--mod_hopper", type=lambda x: x.lower() == "true", default=None) #if doing the hopper task and use this flag, then use hopper dataset that was transformed to the walker dim
    
        
    #defaults for all runs
    parser.add_argument("--save_dir", type=str, default='PbRL_results')
    
    #for IQL Note: this value can affect the capacity encoder training as well, but it is assumed that the task config is set to take care of that (there may also be a reason why one would make them differ between capacity encoder training and the following IQL  so want to allow for that flexibility )
    parser.add_argument("--max_ep_len", type=int, default=1000)

    return parser.parse_args()


    
def get_root_value_PT(run_id, project_name):
    api = wandb.Api()
    run = api.run(f"rlunigoe/{project_name}/{run_id}")

    # Explicitly fetch and download the metadata file
    metadata_file = run.file("wandb-metadata.json")
    temp_path = f"/tmp/{run_id}/wandb-metadata.json"  # Temporary path to store the file
    
    try:
        metadata_file.download(root=os.path.dirname(temp_path), replace=True)
        with open(temp_path, "r") as f:
            metadata = json.load(f)
            return metadata.get("root", None)
    except Exception as e:
        print(f"Error reading metadata for run {run_id}: {e}")
        return None


    

def get_task_dims(task,dataset,save_dir):
    pref_set_path=os.path.join(save_dir,task,'Data',dataset,'preference_dataset.pkl')
    with open(pref_set_path, 'rb') as f:
        preference_set = pickle.load(f)
    obs_dim=preference_set['observations'].shape[2]
    action_dim=preference_set['actions'].shape[2]
    return obs_dim, action_dim



def run_pipeline(args):
    percent=int(args.fraction*100)
    use05=args.use05
    

    if args.mod_hopper:
        dataset_name='Percent{}_05{}_modForWalker'.format(percent,use05)
        taskForData='hopper-medium-replay-v2'
    else:
        dataset_name='Percent{}_05{}'.format(percent,use05)
        taskForData=args.task
    if args.script_label:
        dataset_name += '_scriptLabel'
    if args.mistake_rate>0.0:
        dataset_name += '_mistake{}'.format(int(args.mistake_rate*100))

    
    if args.seed is None:
        seed_list=[231, 107, 93, 1, 123, 827, 67, 42]
        seed_list = seed_list[::-1]
    else:
        if int(args.seed) not in [42, 231, 107, 93, 1, 123, 827, 67]:
            raise ValueError("New seed given as argument")
        seed_list=[int(args.seed)]

    if args.run_type=='run_IQL_CapEnc':
        data_dir=os.path.join(args.save_dir,taskForData,'Data',dataset_name) #if mod_hopper set to true, then taskForData is the hopper dataset (modified) while the IQL task is the walker
        with open(os.path.join(data_dir,'train_set.pkl'), 'rb') as f:
            train_set = pickle.load(f) 
        with open(os.path.join(data_dir,'test_set.pkl'), 'rb') as f:
            test_set = pickle.load(f) 
                
        with open(os.path.join(args.enc_configpath,'enc_config.yaml'), 'rb') as f:
            base_config=yaml.safe_load(f)
        job_type='{}'.format(args.jobInfo)
        for seed in seed_list:
            
            filepath=os.path.join(args.save_dir,taskForData,'CapacityEncoder',dataset_name,'seed{}'.format(seed))
            exp_name=datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")
            fp=os.path.join(filepath,exp_name)
            new_config = copy.deepcopy(base_config)
            new_config['task_name']=args.task
            obs_dim, action_dim=get_task_dims(args.task,dataset_name,args.save_dir)
            new_config['action_dim']=action_dim
            new_config['obs_dim']=obs_dim
            new_config['epochs']=args.enc_epochs
            new_config['stepMax']=args.max_ep_len

            new_config['filepath']=fp
            new_config['exp_name']=exp_name
            new_config['seed']=seed
            
        
            new_config['useFullQuerySet']=True #set to true, may have been set to false when tuning, but now we want the fully query set as our training set
            newGroup='{}_seed{}'.format(dataset_name,seed)+'_FakeEval'
            new_config['group']=newGroup
            
            new_config['job_type']=job_type #just ensure consistency between the entered argument of args.timestep_mask and what the config is. 

            #####check if run wiht config already completed###
            projectName=new_config['task_name']
            if 'scriptLabel' in new_config['group']:
                projectName += '_scriptLabel'
            if 'mistake' in new_config['group']:
                projectName += '_error'
            project="CapacityEncoder_{}".format(projectName)
            runs=has_completed_run_with_config(project, new_config['group'], new_config)
            set_seed(seed=seed)
            if len(runs)==0:
                capacityEncoder=CapacityEncoderV2(new_config,train_set=train_set,test_set=test_set,use_wandb=True)
                capacityEncoder.fit()
            else:
                fp=os.path.join(filepath,runs[0])
                new_config['filepath']=fp #these two lines are not actually needed now but just to be consistent and clean
                new_config['exp_name']=runs[0]

            #Run IQL 
            script_path = os.path.join("OfflineRL-Kit/run_example/run_iql_infrewards.py")
            iqlCommand=["python", script_path, "--task", args.task, '--seed', str(seed), '--simWRestrictedWeight', str(1.0), '--simWUnrestrictedWeight', str(0.0) ,'--causal_pool1', 'False', '--causal_pool2', str(args.causal_pool), '--reward_model', 'SimilarityRewards', "--capacityEncoderFilepath", fp, '--dataset_name',dataset_name, '--job_type', job_type, '--windowRewards', str(args.windowRewards), '--src_mask_decoder', str(args.src_mask_decoder),'--use_vary_seqLens', str(args.use_vary_seqLens)]
            iqlCommand.append('--max_ep_len')
            iqlCommand.append(str(args.max_ep_len))
            if 'kitchen' in args.task or 'pen' in args.task:
                iqlCommand.append('--dropout_rate')
                iqlCommand.append('.1')
                iqlCommand.append('--temperature')
                iqlCommand.append('.5')

            IQlResult = subprocess.run(iqlCommand,  check=True)
            print(IQlResult.stdout) 
    if args.run_type in ['run_IQL_PrefTrans', 'run_IQL_PrefTransformerADT']:
        if 'ADT' not in args.run_type:
            PTproject="PreferenceTransformer_{}".format(taskForData)
            jobInfoPT=args.jobInfo
        else:
            PTproject="PrefTransformerADT_{}".format(taskForData)
            jobInfoPT='ADT_'+args.jobInfo
        if args.script_label:
            PTproject += '_{}'.format('scriptLabel')
        if args.mistake_rate>0.0:
            PTproject += '_error'

        for seed in seed_list:
            prefTransGroup='{}_seed{}_FakeEval'.format(dataset_name,seed)
            best_run=get_run_max_epochs(PTproject,prefTransGroup)
            prefTransFilepath=get_root_value_PT(best_run,PTproject)
            script_path = os.path.expandvars("OfflineRL-Kit/run_example/run_iql_infrewards.py")
            iqlCommand=["python", script_path, "--task", args.task, '--seed', str(seed), '--reward_model', 'PreferenceTransformer', "--PrefTrans_ckpt_dir", prefTransFilepath, '--dataset_name', dataset_name, '--job_type', jobInfoPT]
            iqlCommand.append('--max_ep_len')
            iqlCommand.append(str(args.max_ep_len))
            if 'kitchen' in args.task or 'pen' in args.task:
                iqlCommand.append('--dropout_rate')
                iqlCommand.append('.1')
                iqlCommand.append('--temperature')
                iqlCommand.append('.5')
            
            IQlResult = subprocess.run(iqlCommand, check=True)
            print(IQlResult.stdout)


    
                

    


if __name__ == "__main__":
    args=get_args()
    run_pipeline(args)