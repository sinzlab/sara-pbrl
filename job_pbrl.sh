#!/bin/bash
#SBATCH --job-name=pbrl
#SBATCH -t 8:00:00
#SBATCH --gpus 1  #or change to 10 or 20G slice of GPU if supported by cluster
#SBATCH -p partition
#SBATCH --cpus-per-task=8 
#SBATCH --mem=96G
#SBATCH --ntasks=4 
#SBATCH -a 0-7

echo "Current node: ${SLURM_NODELIST}"
echo "Slurm Array Task ID: ${SLURM_ARRAY_TASK_ID}"
SEEDSTORUN=(231 107 93 1 123 827 67 42)
seed=${SEEDSTORUN[$SLURM_ARRAY_TASK_ID]} 

echo "seed: $seed"

exec > ./slurm_files/slurm-${SLURM_JOB_NAME}-${seed}-${SLURM_JOB_ID}.out \
     2> ./slurm_files/slurm-${SLURM_JOB_NAME}-${seed}-${SLURM_JOB_ID}.err


module load cuda


# Printing out some info.
echo "Allocated node list:  $SLURM_JOB_NODELIST"
echo "Expanded hostnames:    $(scontrol show hostnames $SLURM_JOB_NODELIST)"
echo "This task is running on: $(hostname)"
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi


# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V


 
INSTANCE_NAME="pbrl_instance_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"


PROJ_DIR=FILL IN HERE
SING_PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PROJ_DIR/SARA_PbRL/PreferenceTransformer:$PROJ_DIR/SARA_PbRL/PreferenceTransformer/d4rl:$PROJ_DIR/SARA_PbRL/OfflineRL-Kit:$PROJ_DIR/SARA_PbRL

module load apptainer
apptainer instance start --nv --bind $HOME/.vscode-server:$HOME/.vscode-server,$PROJ_DIR:$PROJ_DIR $PROJ_DIR/SARA_PbRL/pbrlnew.sif $INSTANCE_NAME  
apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python -m pbrl.run_PbRLPipeline --task hopper-medium-replay-v2 --enc_epochs 4000 --run_type run_IQL_SARA --seed $seed --fraction 1.0 --use05 True --script_label False --mistake_rate 0.0 --enc_configpath $PROJ_DIR/SARA_PbRL/pbrl/configs --save_dir $PROJ_DIR/PbRL --jobInfo jobTypeName" 
