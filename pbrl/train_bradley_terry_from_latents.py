#!/usr/bin/env python3
"""
Script to train a Bradley-Terry reward model using latent representations from a pretrained capacity encoder.

Usage:
python train_bradley_terry_from_latents.py --config_path <path_to_config.pkl> --dataset_name <dataset_name> --save_dir <output_directory>
"""

import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
import wandb
import datetime

# Import the capacity encoder classes
from pbrl.ContrastiveCapacityEncoderpbrl import TransformerForInference, CapacityEncoderV2

def get_finished_capenc_run(project_name, group, job_type):
    """
    Get a finished capacity encoder run for the given project, group, and job type.
    
    Args:
        project_name: Name of the wandb project
        group: Group name for the run
        job_type: Job type for the run
        
    Returns:
        str: Filepath to the capacity encoder model
        
    Raises:
        ValueError: If no finished runs found or more than 1 finished run found
    """
    api = wandb.Api()
    
    # Build filters for wandb
    filters = {
        'group': group,
        'state': 'finished'
    }
    
    if job_type is None:
        filters['jobType'] = {'$eq': None}
    else:
        filters['jobType'] = job_type
    
    # Get runs matching the criteria
    runs = api.runs(project_name, filters=filters)
    finished_runs = list(runs)
    
    if len(finished_runs) == 0:
        raise ValueError(f"No finished runs found for project '{project_name}', group '{group}', job_type '{job_type}'")
    elif len(finished_runs) > 1:
        run_names = [run.name for run in finished_runs]
        raise ValueError(f"Found {len(finished_runs)} finished runs for project '{project_name}', group '{group}', job_type '{job_type}'. Expected exactly 1. Run names: {run_names}")
    
    # Get the single finished run
    run = finished_runs[0]
    
    # Extract the filepath from the run config
    if 'filepath' not in run.config:
        raise ValueError(f"Run {run.name} does not have 'filepath' in config")
    
    filepath = run.config['filepath']
    print(f"Found finished capacity encoder run: {run.name}")
    print(f"Capacity encoder filepath: {filepath}")
    
    return filepath

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")

class BradleyTerryRewardModel(nn.Module):
    """Bradley-Terry reward model that takes latent representations as input."""
    
    def __init__(self, embed_dim, hidden_dim=128):
        super(BradleyTerryRewardModel, self).__init__()
        self.network = nn.Sequential( #two layer network with 128 hidden dim and one relu activation-- this is the same as the TransRewardModel used by PreferenceTransformer (between embedding and reward output)
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, latents):
        """
        Args:
            latents: Tensor of shape (batch_size, seq_len, embed_dim) or (batch_size, embed_dim)
        Returns:
            rewards: Tensor of shape (batch_size, seq_len, 1) or (batch_size, 1)
        """
        return self.network(latents)

class PreferenceDataset(Dataset):
    """Dataset for preference learning with latent representations."""
    
    def __init__(self, traj1_latents, traj2_latents, labels):
        """
        Args:
            traj1_latents: Tensor of shape (n_pairs, seq_len, embed_dim)
            traj2_latents: Tensor of shape (n_pairs, seq_len, embed_dim)
            labels: Tensor of shape (n_pairs, 2) - preference labels
        """
        self.traj1_latents = traj1_latents
        self.traj2_latents = traj2_latents
        self.labels = labels
        
        # Convert labels to preference indices
        # [1, 0] -> 0 (first trajectory preferred)
        # [0, 1] -> 1 (second trajectory preferred)  
        # [0.5, 0.5] -> 0.5 (equal preference)
        self.preferences = labels[:, 1]  # Take second element: 0 for first preferred, 1 for second preferred, 0.5 for equal
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'traj1_latents': self.traj1_latents[idx],
            'traj2_latents': self.traj2_latents[idx], 
            'preference': self.preferences[idx]
        }

def load_capacity_encoder(capacityEncoderFilepath, causal_pool1, causal_pool2, windowRewards, device):
    """Load pretrained capacity encoder model."""
    # Load config
    with open(os.path.join(capacityEncoderFilepath,'cfg.pkl'), 'rb') as f:
        capEncCfg = pickle.load(f) 
    
    print(f"Loading capacity encoder from: {capacityEncoderFilepath}")
    
    # Create and load the model
    capacity_encoder = TransformerForInference(capEncCfg, causal_pool1=causal_pool1,causal_pool2=causal_pool2,window_size=windowRewards,device=device).to(device) 
    capacity_encoder.eval()
    
    return capacity_encoder, capEncCfg

def extract_latents_from_trajectories(capacity_encoder, observations, actions, device, use_src_mask, batch_size=32):
    """
    Extract latents from trajectory data using the capacity encoder.
    
    Args:
        capacity_encoder: Pretrained capacity encoder model
        observations: numpy array of shape (n_trajs, seq_len, obs_dim)
        actions: numpy array of shape (n_trajs, seq_len, action_dim)
        device: torch device
        batch_size: batch size for processing
    
    Returns:
        latents: Tensor of shape (n_trajs, seq_len, embed_dim)
    """
    n_trajs = observations.shape[0]
    seq_len = observations.shape[1]
    
    # Concatenate observations and actions
    traj_data = np.concatenate([observations, actions], axis=-1)  # (n_trajs, seq_len, obs_dim + action_dim)
    traj_tensor = torch.FloatTensor(traj_data).to(device)
    
    # Create padding mask (assume no padding for now)
    padding_mask = torch.zeros(n_trajs, seq_len, dtype=torch.bool).to(device)
    
    all_latents = []
    
    # Process in batches
   
    for i in range(0, n_trajs, batch_size):
        batch_end = min(i + batch_size, n_trajs) 
        batch_trajs = traj_tensor[i:batch_end]
        batch_mask = padding_mask[i:batch_end]
        
        # Get latents from capacity encoder
        batch_latents = capacity_encoder(batch_trajs, src_key_padding_mask=batch_mask, use_src_mask=use_src_mask).detach() 
        all_latents.append(batch_latents) 
    
    return torch.cat(all_latents, dim=0).detach()

def bradley_terry_loss(reward_model, traj1_latents, traj2_latents, preferences, seqLenList):
    """
    Compute Bradley-Terry loss for preference learning at specific timesteps.
    
    Args:
        reward_model: Bradley-Terry reward model
        traj1_latents: Latents for first trajectories (batch_size, seq_len, embed_dim)
        traj2_latents: Latents for second trajectories (batch_size, seq_len, embed_dim)
        preferences: Preference labels (batch_size,) - 0 for first preferred, 1 for second, 0.5 for equal
        seqLenList: List of timestep indices to compute loss for
    
    Returns:
        loss: Bradley-Terry loss summed over specified timesteps
    """
    # Get rewards for all timesteps
    rewards1 = reward_model(traj1_latents)  # (batch_size, seq_len, 1)
    rewards2 = reward_model(traj2_latents)  # (batch_size, seq_len, 1)
    
    # Squeeze to get (batch_size, seq_len) for easier computation
    rewards1 = rewards1.squeeze(-1)  # (batch_size, seq_len)
    rewards2 = rewards2.squeeze(-1)  # (batch_size, seq_len)
    
    # Compute Bradley-Terry loss for all timesteps at once
    logits = rewards1 - rewards2  # (batch_size, seq_len)
    probs = torch.sigmoid(logits)  # (batch_size, seq_len)
    targets = (1 - preferences).unsqueeze(1).expand(-1, rewards1.shape[1])  # (batch_size, seq_len)
    
    # Compute loss for all timesteps
    losses_all_timesteps = nn.functional.binary_cross_entropy(probs, targets, reduction='none')  # (batch_size, seq_len)
    
    # Create mask for timesteps in seqLenList
    seq_len = rewards1.shape[1]
    timestep_mask = torch.zeros(seq_len, dtype=torch.bool, device=rewards1.device)
    
    # Set mask to True for timesteps in seqLenList (convert to 0-indexed)
    for seq_len_val in seqLenList:
        if seq_len_val <= seq_len:
            timestep_idx = seq_len_val - 1
            timestep_mask[timestep_idx] = True
    
    # Apply mask and sum over selected timesteps
    masked_losses = losses_all_timesteps[:, timestep_mask]  # (batch_size, num_selected_timesteps)
    total_loss = masked_losses.mean()
    
    return total_loss

def train_bradley_terry_model(reward_model, dataloader, optimizer, device, seqLenList, num_epochs=100, logger=None):
    """Train the Bradley-Terry reward model."""
    reward_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            traj1_latents = batch['traj1_latents'].to(device)
            traj2_latents = batch['traj2_latents'].to(device)
            preferences = batch['preference'].to(device)
            
            optimizer.zero_grad()
            
            loss = bradley_terry_loss(reward_model, traj1_latents, traj2_latents, preferences, seqLenList)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Log to wandb if logger is provided
        if logger is not None:
            logger.log({'train/bradley_terry_loss': avg_loss}, step=epoch)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return reward_model

def bradley_terry_inference(ckpt_dir, trajectories, env_name, batch_size=256):
    """
    Load a trained Bradley-Terry model and compute rewards for given trajectories.
    
    Args:
        ckpt_dir: Path to Bradley-Terry checkpoint directory
        trajectories: List of (observations, actions) tuples for each trajectory
        env_name: Environment name for validation
        batch_size: Batch size for processing
    
    Returns:
        all_traj_rewards: List of reward arrays, one per trajectory
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Bradley-Terry model config
    model_path = os.path.join(ckpt_dir, 'bradley_terry_reward_model.pt')
    config_path = os.path.join(ckpt_dir, 'training_config.pkl')
    
    with open(config_path, 'rb') as f:
        training_config = pickle.load(f)
    
    # Extract path information to find capacity encoder
    path_parts = ckpt_dir.split(os.sep)
    
    # Find task name and dataset name from path
    bt_idx = -1
    for i, part in enumerate(path_parts):
        if 'BradleyTerryContrastive' in part: 
            bt_idx = i
            break
    
    if bt_idx == -1:
        raise ValueError("Could not find BradleyTerryContrastive in checkpoint path")
    
    task_name = path_parts[bt_idx - 1]
    if task_name != env_name:
        raise ValueError("Incorrect Bradley Terry checkpoint path for env name")
    
    capacityEncoderFilepath=training_config['capacityEncoderFilepath']
    
    # Load capacity encoder with default parameters (should ideally be stored in training config)
    causal_pool1 = training_config['causal_pool1'] 
    causal_pool2 = training_config['causal_pool2']
    windowRewards = training_config['windowRewards']
    use_src_mask = training_config['use_src_mask']
    
    capacity_encoder, _ = load_capacity_encoder(capacityEncoderFilepath, causal_pool1, causal_pool2, windowRewards, device)
    
    # Load Bradley-Terry model
    embed_dim = training_config['embed_dim']
    hidden_dim = training_config['hidden_dim']
    bradley_terry_model = BradleyTerryRewardModel(embed_dim, hidden_dim).to(device)
    bradley_terry_model.load_state_dict(torch.load(model_path, map_location=device))
    bradley_terry_model.eval()
    
    # Process trajectories to get rewards
    all_traj_rewards = []
    
    
    for traj_obs, traj_actions in trajectories:
        # Extract latents for this trajectory
        traj_obs_expanded = np.expand_dims(traj_obs, axis=0)  # (1, seq_len, obs_dim)
        traj_actions_expanded = np.expand_dims(traj_actions, axis=0)  # (1, seq_len, action_dim)
        
        latents = extract_latents_from_trajectories(
            capacity_encoder,
            traj_obs_expanded,
            traj_actions_expanded,
            device,
            use_src_mask=use_src_mask,
            batch_size=1
        )  # (1, seq_len, embed_dim)
        
        # Get rewards from Bradley-Terry model
        traj_rewards = bradley_terry_model(latents).detach()  # (1, seq_len, 1)
        
        # Properly handle squeezing to maintain trajectory structure
        traj_rewards = traj_rewards.squeeze(0).squeeze(-1).cpu().numpy()  # (seq_len,)
        
        # Ensure it's always 1D (handle single timestep case)
        if traj_rewards.ndim == 0:
            traj_rewards = np.array([traj_rewards])
        
        all_traj_rewards.append(traj_rewards)
    
    return all_traj_rewards

def main():
    parser = argparse.ArgumentParser(description='Train Bradley-Terry reward model from capacity encoder latents')
    parser.add_argument('--task_name', required=True, help='Task name') 
    parser.add_argument('--dataset_name', required=True, help='Name of dataset') 
    parser.add_argument('--save_dir', help='Output directory for saved model', default='/mnt/vast-react/projects/rl_pref_constraint/PbRL') 
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for Bradley-Terry model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--job_type', type=str, default='train', help='Wandb job type')
    
    #for capacity encoder inference
    parser.add_argument("--causal_pool1", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--causal_pool2", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--windowRewards",type=lambda x: int(x) if x.lower() != "none" else None, default=None)#parser.add_argument("--windowRewards", type=int, default=None)
    parser.add_argument("--src_mask_decoder", type=lambda x: x.lower() == "true", default=False)
    
    args = parser.parse_args()
    
    #get the capacityEncoderFilepath corresponding to the desired run
    group = args.dataset_name+'_seed{}'.format(args.seed)+'_FakeEval'
    job_type=args.job_type
    task_name=args.task_name
    projectNameSuffix = task_name
    if 'scriptLabel' in args.dataset_name:
        projectNameSuffix += '_scriptLabel'
    if 'mistake' in args.dataset_name:
        projectNameSuffix += '_error'
    projectCapEnc="CapacityEncoder_{}".format(projectNameSuffix)
    capacityEncoderFilepath=get_finished_capenc_run(projectCapEnc,group,job_type) 
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    exp_name=datetime.datetime.now().strftime("%Y.%m.%d_%H%M%S")

    # Create output directory
    output_dir=os.path.join(args.save_dir,task_name,'BradleyTerryContrastive',args.dataset_name,'seed{}'.format(args.seed),exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load capacity encoder
    print("Loading capacity encoder...")
    capacity_encoder, cfg = load_capacity_encoder(capacityEncoderFilepath, args.causal_pool1, args.causal_pool2, args.windowRewards, device)
    if cfg.get('numSetsPerAgent') is not None:
        raise ValueError("Expect that numSetsPerAgent should be None")
    
    # wandb logging
    # Follow the same pattern as ContrastiveCapacityEncoderpbrl
    filepath = Path(capacityEncoderFilepath)
    parts = list(filepath.parts)
    
    if task_name in parts:
        indexfp = parts.index(task_name)
        wandbDir = os.path.join(os.sep.join(parts[:indexfp + 1]), 'wandb')
    else:
        wandbDir = '/mnt/vast-react/projects/rl_pref_constraint/wandb'
       
    
    
    # Create wandb config
    wandb_config = {
        'capacityEncoderFilepath': capacityEncoderFilepath,
        'ckpt_filepath': output_dir,
        'dataset_name': args.dataset_name,
        'job_type': job_type,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'embed_dim': cfg['embedDim'],
        'hidden_dim': args.hidden_dim,
        'seed': args.seed,
        'causal_pool1': args.causal_pool1,
        'causal_pool2': args.causal_pool2,
        'windowRewards': args.windowRewards,
        'use_src_mask': args.src_mask_decoder,
        'device': str(device)
    }
    
    logger = wandb.init(
        project=f"BradleyTerryContrastive_{projectNameSuffix}",
        group=group,
        name=exp_name,
        config=wandb_config,
        job_type=job_type,
        dir=wandbDir
    )
    
    # Load preference dataset
    print("Loading preference dataset...")
    data_dir=os.path.join(args.save_dir,task_name,'Data',args.dataset_name,'preference_dataset.pkl')
    with open(data_dir, 'rb') as f:
        preference_dataset = pickle.load(f)
    
    print(f"Preference dataset contains {len(preference_dataset['observations'])} preference pairs")
    
    # Extract latents for both trajectory sets
    print("Extracting latents for first trajectories...")
    traj1_latents = extract_latents_from_trajectories(
        capacity_encoder, 
        preference_dataset['observations'], 
        preference_dataset['actions'],
        device,
        use_src_mask=args.src_mask_decoder,
        batch_size=args.batch_size
    )
    
    print("Extracting latents for second trajectories...")
    traj2_latents = extract_latents_from_trajectories(
        capacity_encoder,
        preference_dataset['observations_2'],
        preference_dataset['actions_2'], 
        device,
        use_src_mask=args.src_mask_decoder,
        batch_size=args.batch_size
    )
    
    print(f"Latent shape: {traj1_latents.shape}")
    
    # Convert labels to tensor
    labels = torch.FloatTensor(preference_dataset['labels'])
    
    # Create dataset and dataloader
    dataset = PreferenceDataset(traj1_latents, traj2_latents, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create Bradley-Terry reward model
    embed_dim = traj1_latents.shape[-1]
    reward_model = BradleyTerryRewardModel(embed_dim, args.hidden_dim).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(reward_model.parameters(), lr=args.lr)
    
    print(f"Training Bradley-Terry model with {sum(p.numel() for p in reward_model.parameters())} parameters...")
    
    # Train the model
    trained_model = train_bradley_terry_model(
        reward_model, dataloader, optimizer, device, cfg['seqLenList'], args.num_epochs, logger
    )
    
    # Save the trained model
    model_path = os.path.join(output_dir, 'bradley_terry_reward_model.pt')
    torch.save(trained_model.state_dict(), model_path)
    print(f"Saved trained model to: {model_path}")
    
    logger.save(model_path)
    
    # Save training config
    config_path = os.path.join(output_dir, 'training_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(wandb_config, f)
    print(f"Saved training config to: {config_path}")
    
    # Log final metrics to wandb
    logger.finish()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
