#!/usr/bin/env python3
"""
Script to compute correlations and outliers between model data and ground truth data,
and save the datasets as pickle files.

Usage:
python compute_correlations_save_datasets.py --project <project> --dataset_name <dataset_name> 
                                            --seed <seed> --model <model> --job_type <job_type>
                                            [--pt_job_type <pt_job_type>] [--pt_adt_job_type <pt_adt_job_type>]
                                            [--sara_job_type <sara_job_type>]
"""

import argparse
import os
import pickle
import sys
import numpy as np
import wandb
import yaml
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
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

# Add the SARA_PbRL directory to the path
sys.path.append('/mnt/vast-react/projects/rl_pref_constraint/SARA_PbRL')

# Import required modules
from PreferenceTransformer.get_PTdataset import make_env_and_dataset
from pbrl.make_offlinedataset import make_offline_dataset, normalize


def get_wandb_config(project, group, job_type, seed):
    """Get wandb config for a specific run."""
    api = wandb.Api()

    # Fetch the run directly using filters
    filters = {"group": group}
    if job_type is None:
        filters["jobType"] = {"$eq": None}  # Handle None job_type properly
    else:
        filters["jobType"] = job_type
    
    runs = api.runs(f"{project}", filters=filters)
    target_run = None
    for run in runs:
        if f'seed{seed}' in run.name:
            target_run = run
            break
    if target_run is None:
        raise ValueError(f"No run found with seed '{seed}' in group '{group}' and jobType {job_type}")

    # Extract config as a dictionary
    config_dict = dict(target_run.config)
    print(f"Found run: {target_run.name}")

    # Convert to YAML format (if needed)
    config_yaml = yaml.dump(config_dict, default_flow_style=False)

    return yaml.safe_load(config_yaml)  # Return as a dictionary


def make_PT_dataset(env_name, seed, ckpt_dir, max_ep_len):
    """Create PT dataset from checkpoint directory."""
    dataset = make_env_and_dataset(env_name=env_name, seed=seed, ckpt_dir=ckpt_dir, model_type='PrefTransformer')
    
    datasetOffline = {}
    datasetOffline['observations'] = dataset.observations
    datasetOffline['actions'] = dataset.actions
    datasetOffline['rewards'] = dataset.rewards
    datasetOffline['masks'] = dataset.masks
    datasetOffline['terminals'] = dataset.dones_float
    datasetOffline['next_observations'] = dataset.next_observations

    datasetOffline = normalize(datasetOffline, env_name, max_episode_steps=max_ep_len)
    if 'antmaze' in env_name:
        datasetOffline['rewards'] -= 1.0
    if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        datasetOffline['rewards'] += 0.5

    return datasetOffline


def make_GT_dataset(env_name, seed, max_ep_len):
    """Create ground truth dataset."""
    dataset = make_env_and_dataset(env_name=env_name, seed=seed, ckpt_dir=None, model_type="GroundTruth")
    
    datasetOffline = {}
    datasetOffline['observations'] = dataset.observations
    datasetOffline['actions'] = dataset.actions
    datasetOffline['rewards'] = dataset.rewards
    datasetOffline['masks'] = dataset.masks
    datasetOffline['terminals'] = dataset.dones_float
    datasetOffline['next_observations'] = dataset.next_observations

    datasetOffline = normalize(datasetOffline, env_name, max_episode_steps=max_ep_len)
    if 'antmaze' in env_name:
        datasetOffline['rewards'] -= 1.0
    if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        datasetOffline['rewards'] += 0.5

    return datasetOffline


def make_offlinedataset_fromargs(cfg, reward_model):
    """Create offline dataset from config based on reward model type."""
    if reward_model == "GroundTruth":
        dataset = make_GT_dataset(env_name=cfg['task'], seed=cfg['seed'], max_ep_len=cfg['max_ep_len'])
    elif reward_model == 'SimilarityRewards':
        dataset = make_offline_dataset(cfg)
    elif reward_model == 'PreferenceTransformer':
        dataset = make_PT_dataset(env_name=cfg['task'], seed=cfg['seed'], 
                                ckpt_dir=cfg['PrefTrans_ckpt_dir'], max_ep_len=cfg['max_ep_len'])
    else:
        raise ValueError("Reward Model must be either GroundTruth, SimilarityRewards or PreferenceTransformer")

    return dataset


def get_PT_data(project, group, job_type, seed):
    """Get PT dataset from wandb config."""
    iqlConfig = get_wandb_config(project, group, job_type, seed)
    dataset = make_offlinedataset_fromargs(iqlConfig, 'PreferenceTransformer')
    return dataset


def get_sim_dataset(project, group, job_type, seed):
    """Get similarity rewards dataset from wandb config."""
    iqlConfig = get_wandb_config(project, group, job_type, seed)
    dataset = make_offlinedataset_fromargs(iqlConfig, 'SimilarityRewards')
    return dataset


def get_groundtruth_data(project, group, job_type, seed):
    """Get ground truth dataset from wandb config."""
    iqlConfig = get_wandb_config(project, group, job_type, seed)
    dataset = make_offlinedataset_fromargs(iqlConfig, 'GroundTruth')
    return dataset


def check_obs_act_aligned(modelData, gtData):
    """Check if model data and ground truth data are aligned."""
    aligned = True
    for key in ['observations', 'actions', 'next_observations']:
        if not (np.round(modelData[key], 4) == np.round(gtData[key], 4)).all():
            aligned = False
    return aligned


def compute_correlations_outliers(model_data, ground_truth_data):
    """Compute correlations and outliers between model and ground truth data."""
    # first check the obs, action, next_obs are aligned (so we're computing correlations for the same input)
    if not check_obs_act_aligned(model_data, ground_truth_data):
        raise ValueError("model data not aligned with ground truth obs, act, next_obs")

    ground_truth = ground_truth_data['rewards']
    model = model_data['rewards']
    
    ### 1. Pearson Correlation
    pearson_corr, pValPears = pearsonr(ground_truth, model)

    ### 2. Spearman Correlation
    spearman_corr, pValSpear = spearmanr(ground_truth, model)

    ### 3. IQR-Based Outlier Detection (based on ground truth)
    Q1 = np.percentile(ground_truth, 25)
    Q3 = np.percentile(ground_truth, 75)
    IQR = Q3 - Q1

    # Define IQR outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outlier detection in the model distribution
    upper_outliers = model > upper_bound
    lower_outliers = model < lower_bound

    # Count
    num_upper_outliers = np.sum(upper_outliers)
    num_lower_outliers = np.sum(lower_outliers)
    numTransitions = ground_truth.shape[0]
    
    return (pearson_corr, pValPears), (spearman_corr, pValSpear), num_lower_outliers/numTransitions, num_upper_outliers/numTransitions


def save_dataset(dataset, save_path, filename):
    """Save dataset as pickle file."""
    os.makedirs(save_path, exist_ok=True)  # Create directory structure if it doesn't exist
    full_path = os.path.join(save_path, filename)
    with open(full_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to: {full_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute correlations and save datasets')
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--dataset_name', required=True, help='Dataset name')
    parser.add_argument('--seed', type=int, required=True, help='Seed value')
    parser.add_argument('--model', required=True, choices=['PT', 'PT+ADT', 'SARA'], help='Model type')
    parser.add_argument('--job_type', required=True, help='Job type for the specified model')
    parser.add_argument('--gt_job_type', default='NewNorm', help='Job type for ground truth (default: NewNorm)')  # Added for ground truth
    parser.add_argument('--mistake_rate', type=float, default=0.0, help='Mistake rate for error experiments (default: 0.0)')
    
    args = parser.parse_args()
    
    # Determine model_name based on model type
    if args.model in ['PT', 'PT+ADT']:
        model_name = 'PreferenceTransformer'
    elif args.model == 'SARA':
        model_name = 'SimilarityRewards'
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Handle mistake rate modifications
    dataset_name = args.dataset_name
    project_name = args.project
    project_name_gt = args.project  # Ground truth uses original project name
    
    if args.mistake_rate > 0.0:
        dataset_name += '_mistake{}'.format(int(args.mistake_rate*100))
        project_name += '_error'  # Only for model data, not ground truth
    
    # Construct group names
    model_group = f"{dataset_name}_{model_name}"
    gt_group = "GroundTruth"  # Ground truth group name
    
    print(f"Processing: Project={args.project}, Dataset={args.dataset_name}, Model={args.model}, Seed={args.seed}")
    print(f"Mistake rate: {args.mistake_rate}")
    print(f"Modified dataset name: {dataset_name}")
    print(f"Model project: {project_name}")
    print(f"GT project: {project_name_gt}")
    print(f"Model group: {model_group}")
    print(f"Ground truth group: {gt_group}")
    
    try:
        # Ensure base directory structure exists
        base_save_path = "relabeled_offlinedatasets"
        # Handle None gt_job_type for directory structure
        if args.gt_job_type=="None":
            args.gt_job_type=None
        gt_job_folder = "null" if args.gt_job_type is None else args.gt_job_type
        gt_save_path = os.path.join(base_save_path, project_name_gt, args.dataset_name, "GroundTruth", gt_job_folder)
        os.makedirs(gt_save_path, exist_ok=True)  # Create directory structure if it doesn't exist
        
        # Check if ground truth data already exists, if so load it
        gt_filename = f"groundtruth_seed{args.seed}.pkl"
        gt_full_path = os.path.join(gt_save_path, gt_filename)
        
        if os.path.exists(gt_full_path):
            print(f"Loading existing ground truth data from: {gt_full_path}")
            with open(gt_full_path, 'rb') as f:
                gt_data = pickle.load(f)
        else:
            print("Ground truth data not found, creating new...")
            gt_data = get_groundtruth_data(project_name_gt, gt_group, args.gt_job_type, args.seed)
        
        # Get model data based on model type
        print(f"Loading {args.model} model data...")
        if args.job_type=="None":
            args.job_type=None
        if args.model in ['PT', 'PT+ADT']:
            model_data = get_PT_data(project_name, model_group, args.job_type, args.seed)
        elif args.model == 'SARA':
            model_data = get_sim_dataset(project_name, model_group, args.job_type, args.seed)
        
        # Compute correlations and outliers
        print("Computing correlations and outliers...")
        results = compute_correlations_outliers(model_data, gt_data)
        
        # Print results
        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        pearson_results, spearman_results, lower_outliers_frac, upper_outliers_frac = results
        pearson_corr, pearson_pval = pearson_results
        spearman_corr, spearman_pval = spearman_results
        
        print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_pval:.4e})")
        print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_pval:.4e})")
        print(f"Lower Outliers Fraction: {lower_outliers_frac:.4f}")
        print(f"Upper Outliers Fraction: {upper_outliers_frac:.4f}")
        print("="*50)
        
        # Save datasets (only save ground truth if it wasn't already loaded from file)
        if not os.path.exists(gt_full_path):
            save_dataset(gt_data, gt_save_path, gt_filename)
        else:
            print(f"Ground truth dataset already exists at: {gt_full_path}")
        
        # Save model dataset
        # Handle None job_type for directory structure
        job_folder = "null" if args.job_type is None else args.job_type
        model_save_path = os.path.join(base_save_path, project_name, dataset_name, model_group, job_folder)
        os.makedirs(model_save_path, exist_ok=True)  # Create directory structure if it doesn't exist
        model_filename = f"{args.model.lower()}_seed{args.seed}.pkl"
        save_dataset(model_data, model_save_path, model_filename)
        
        print(f"\nDatasets saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())