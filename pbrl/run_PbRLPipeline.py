import pickle
import os
import argparse


### imports needed for environments###
import collections
import collections.abc

# Restore the old name so D4RL’s isinstance(...) check will work
collections.Mapping = collections.abc.Mapping

# now disable dm_control if you still want that
from d4rl.kitchen.adept_envs import mujoco_env
mujoco_env.USE_DM_CONTROL = False
import gym, d4rl
from d4rl import hand_manipulation_suite
###################################


from pbrl.ContrastiveCapacityEncoderpbrl import set_seed,CapacityEncoderV2
import datetime
import wandb
import copy
import subprocess 
import json
import yaml
os.environ['WANDB_INIT_TIMEOUT'] = '600'



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
        return max_epoch_run, max_epoch_run.summary.get("epoch", 0)
    else:
        return None


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
    parser.add_argument("--run_type", type=str, default=None) #run_IQL_SARA, run_IQL_PrefTrans, run_IQL_PrefTransADT, run_DPPO
    parser.add_argument("--script_label", type=lambda x: x.lower() == "true", default=False) #to use reward models run using the script labels
    
    
    parser.add_argument("--enc_epochs", type=int, default=None) 
    parser.add_argument("--enc_lr", type=float, default=None, help="Learning rate for capacity encoder. If not specified, uses the default from config file.")
    parser.add_argument("--windowRewards",type=lambda x: int(x) if x.lower() != "none" else None, default=None)
    parser.add_argument("--src_mask_decoder", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_vary_seqLens", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--causal_pool",type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=None) #specify seed for IQL run. If not specified then will run over the 8 seeds in a for loop
    parser.add_argument("--enc_configpath", type=str, default=None) #Not mandatory to specify but rather only specify this if running IQL with a preferred capacity encoder congif. This will run over seeds (or seed if args.seed specified) 
    parser.add_argument("--jobInfo", type=str, default='')

    parser.add_argument("--pt_epochs", type=int, default=10000) 


    parser.add_argument("--mod_hopper", type=lambda x: x.lower() == "true", default=None) #if doing the hopper task and use this flag, then use hopper dataset that was transformed to the walker dim
    
        
    #defaults for all runs
    parser.add_argument("--save_dir", type=str, default='/mnt/vast-react/projects/rl_pref_constraint/PbRL')
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for preferred similarity")
    parser.add_argument("--beta", type=float, default=0.0, help="Weight for unpreferred similarity")

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


def get_env_max_len(task):
    env=gym.make(task)
    return env._max_episode_steps


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

    max_ep_len=get_env_max_len(args.task)
    if args.run_type=='run_IQL_SARA':
        # Set default enc_epochs based on task if not provided
        if args.enc_epochs is None:
            if args.task == 'hopper-medium-expert-v2':
                args.enc_epochs = 2000
            elif args.task == 'walker2d-medium-replay-v2' or args.task == 'halfcheetah-medium-replay-v2':
                args.enc_epochs = 20000
            else:
                args.enc_epochs = 4000
        
        data_dir=os.path.join(args.save_dir,taskForData,'Data',dataset_name) #if mod_hopper set to true, then taskForData is the hopper dataset (modified) while the IQL task is the walker
        with open(os.path.join(data_dir,'train_set.pkl'), 'rb') as f:
            train_set = pickle.load(f) 
        with open(os.path.join(data_dir,'test_set.pkl'), 'rb') as f:
            test_set = pickle.load(f) 

        if 'yaml' not in args.enc_configpath:
            configPath=os.path.join(args.enc_configpath,'enc_config.yaml')
        else:
            configPath=args.enc_configpath   
        with open(configPath, 'rb') as f:
            base_config=yaml.safe_load(f)
        job_type='{}'.format(args.jobInfo)
        for seed in seed_list:
            
            filepath=os.path.join(args.save_dir,taskForData,'CapacityEncoder',dataset_name,'seed{}'.format(seed))
            exp_name=datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")
            fp=os.path.join(filepath,exp_name)
            new_config = copy.deepcopy(base_config)
            new_config['task_name']=args.task
            obs_dim, action_dim=get_task_dims(args.task,dataset_name,args.save_dir)
            print('obsdim {} action dim {}'.format(obs_dim,action_dim))
            new_config['action_dim']=action_dim
            new_config['obs_dim']=obs_dim
            new_config['epochs']=args.enc_epochs
            new_config['stepMax']=max_ep_len

            # Override learning rate if specified
            if args.enc_lr is not None:
                new_config['lr'] = args.enc_lr

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
            iqlCommand=["python", script_path, "--task", args.task, '--seed', str(seed), '--simWRestrictedWeight', str(args.alpha), '--simWUnrestrictedWeight', str(args.beta) ,'--causal_pool1', 'False', '--causal_pool2', str(args.causal_pool), '--reward_model', 'SimilarityRewards', "--capacityEncoderFilepath", fp, '--dataset_name',dataset_name, '--job_type', job_type, '--windowRewards', str(args.windowRewards), '--src_mask_decoder', str(args.src_mask_decoder),'--use_vary_seqLens', str(args.use_vary_seqLens)]
            iqlCommand.append('--max_ep_len')
            iqlCommand.append(str(max_ep_len))
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
            model_type="PrefTransformer"
        else:
            PTproject="PrefTransformerADTHighG_{}".format(taskForData)
            jobInfoPT='ADT_'+args.jobInfo
            model_type="PrefTransformerADT"
        if args.script_label:
            PTproject += '_{}'.format('scriptLabel')
        if args.mistake_rate>0.0:
            PTproject += '_error'

        for seed in seed_list:
            prefTransGroup='{}_seed{}_FakeEval'.format(dataset_name,seed)
            result = None#get_run_max_epochs(PTproject,prefTransGroup)
            runPTPretrain=False
            if result is None:
                runPTPretrain=True
            else:
                best_run, epoch_count = result
                if epoch_count<args.pt_epochs-1:
                    runPTPretrain=True
            if runPTPretrain:
                # First step: Train the preference transformer
                ptCommand = ["python", "-m", "JaxPref.new_preference_reward_main", 
                            "--env", taskForData, 
                            "--transformer.embd_dim", "256", 
                            "--transformer.n_layer", "1", 
                            "--transformer.n_head", "4", 
                            "--logging.output_dir", args.save_dir, 
                            "--batch_size", "256", 
                            "--skip_flag", "0", 
                            "--n_epochs", str(args.pt_epochs), 
                            "--seed", str(seed), 
                            "--fraction", str(args.fraction),  
                            "--mistake_rate", str(args.mistake_rate), 
                            "--model_type", model_type]
                if args.use05:
                    ptCommand.append("--use05")
                    ptCommand.append(str(args.use05))
                if not args.script_label:
                    ptCommand.append("--use_human_label")
                    ptCommand.append(str(not args.script_label))

                # Change to PreferenceTransformer directory and run the command
                original_cwd = os.getcwd()
                os.chdir("PreferenceTransformer")
                try:
                    PTResult = subprocess.run(ptCommand, check=True)
                    print("Preference Transformer training completed successfully")
                    print(PTResult.stdout)
                finally:
                    os.chdir(original_cwd)
            
            # Second step: Run IQL with the trained preference transformer
            result = get_run_max_epochs(PTproject,prefTransGroup)
            if result is None:
                raise ValueError(f"No finished runs found for project {PTproject} and group {prefTransGroup}")
            else:
                best_run, epoch_count = result
                if epoch_count<args.pt_epochs-1: 
                    raise ValueError(f"No finished runs found for project {PTproject} and group {prefTransGroup} with {args.pt_epochs-1} epochs")
            prefTransFilepath=get_root_value_PT(best_run.id,PTproject)
            script_path = os.path.expandvars("OfflineRL-Kit/run_example/run_iql_infrewards.py")
            iqlCommand=["python", script_path, "--task", args.task, '--seed', str(seed), '--reward_model', 'PreferenceTransformer', "--PrefTrans_ckpt_dir", prefTransFilepath, '--dataset_name', dataset_name, '--job_type', jobInfoPT]
            iqlCommand.append('--max_ep_len')
            iqlCommand.append(str(max_ep_len))
            if 'kitchen' in args.task or 'pen' in args.task:
                iqlCommand.append('--dropout_rate')
                iqlCommand.append('.1')
                iqlCommand.append('--temperature')
                iqlCommand.append('.5')
            
            IQlResult = subprocess.run(iqlCommand, check=True)
            print(IQlResult.stdout)

    if args.run_type == 'run_DPPO':
        for seed in seed_list:
            # First step: Train the preference transformer with DPPO model type
            dpCommand = ["python", "-m", "JaxPref.new_preference_reward_main", 
                        "--env", taskForData, 
                        "--seed", str(seed), 
                        "--fraction", str(args.fraction),  
                        "--mistake_rate", str(args.mistake_rate), 
                        "--model_type", "PrefTransformerDPPO"]
            if args.use05:
                dpCommand.append("--use05")
                dpCommand.append("True")
            if not args.script_label:
                dpCommand.append("--use_human_label")
                dpCommand.append("True")

            # Change to DPPO directory and run the command
            original_cwd = os.getcwd()
            os.chdir("DPPO")
            try:
                PredResult = subprocess.run(dpCommand, check=True)
                print("DPPO Preference Predictor training completed successfully")
                print(PredResult.stdout)
            finally:
                os.chdir(original_cwd)
            
            # Second step: Run DPPO training
            dppoCommand = ["python", "DPPO/train.py", 
                          "--env_name", args.task, 
                          "--seed", str(seed), 
                          "--dataset_name", dataset_name]
            
            # Add lambda parameter for kitchen and pen tasks
            # Default value of .5 is used for D4RL locomotion tasks
            if 'kitchen' in args.task or 'pen' in args.task:
                dppoCommand.append('--lambd')
                dppoCommand.append('.1')
            
            DPPOResult = subprocess.run(dppoCommand, check=True)
            print(DPPOResult.stdout)


if __name__ == "__main__":
    args=get_args()
    run_pipeline(args)