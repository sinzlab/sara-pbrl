#!/bin/bash
#SBATCH --job-name=walkpearson
#SBATCH -t 28:00:00
#SBATCH --gpus 1  #or change to 10 or 20G slice of GPU if supported by cluster
#SBATCH -p grete:interactive
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32G
#SBATCH --ntasks=1 


echo "Current node: ${SLURM_NODELIST}"
echo "Slurm Array Task ID: ${SLURM_ARRAY_TASK_ID}"

seed=${SEEDSTORUN[$SLURM_ARRAY_TASK_ID]} 

echo "seed: $seed"
PROJ_DIR=/mnt/vast-react/projects/rl_pref_constraint
exec > $PROJ_DIR/SARA_PbRL/slurm_files/slurm-${SLURM_JOB_NAME}-${seed}-${SLURM_JOB_ID}.out \
     2> $PROJ_DIR/SARA_PbRL/slurm_files/slurm-${SLURM_JOB_NAME}-${seed}-${SLURM_JOB_ID}.err





# Printing out some info.
echo "Allocated node list:  $SLURM_JOB_NODELIST"
echo "Expanded hostnames:    $(scontrol show hostnames $SLURM_JOB_NODELIST)"
echo "This task is running on: $(hostname)"
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"



# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V


 
INSTANCE_NAME="pbrl_instance_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"



SING_PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PROJ_DIR/SARA_PbRL/PreferenceTransformer:$PROJ_DIR/SARA_PbRL/PreferenceTransformer/d4rl:$PROJ_DIR/SARA_PbRL/OfflineRL-Kit:$PROJ_DIR/SARA_PbRL

module load apptainer
apptainer instance start --nv --bind $HOME/.vscode-server:$HOME/.vscode-server,$PROJ_DIR:$PROJ_DIR $PROJ_DIR/SARA_PbRL/pbrlnew.sif $INSTANCE_NAME 
##ALREADY DONE
#apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_hopper-medium-replay-v2 --dataset_name Percent100_05True --model SARA --job_type NoMask_ESFalse_NoSrc_cpTrue_RandomSubseqs --gt_job_type NewNorm --mistake_rate 0.2"

##SUBMITTED 1
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_hopper-medium-replay-v2 --dataset_name Percent100_05True --model PT --job_type None --gt_job_type NewNorm --mistake_rate 0.2"
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_hopper-medium-replay-v2 --dataset_name Percent100_05True --model PT+ADT --job_type ADTHighG --gt_job_type NewNorm --mistake_rate 0.2"

# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_hopper-medium-expert-v2 --dataset_name Percent100_05True --model SARA --job_type NoMask_ESFalse_NoSrc_cpTrue_RandomSubseqs --gt_job_type NewNorm --mistake_rate 0.2"
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_hopper-medium-expert-v2 --dataset_name Percent100_05True --model PT --job_type None --gt_job_type NewNorm --mistake_rate 0.2" 
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_hopper-medium-expert-v2 --dataset_name Percent100_05True --model PT+ADT --job_type ADTHighG --gt_job_type NewNorm --mistake_rate 0.2"

# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_walker2d-medium-expert-v2 --dataset_name Percent100_05True --model SARA --job_type NoMask_ESFalse_NoSrc_cpTrue_RandomSubseqs --gt_job_type NewNorm --mistake_rate 0.2" 
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_walker2d-medium-expert-v2 --dataset_name Percent100_05True --model PT --job_type None --gt_job_type NewNorm --mistake_rate 0.2" 
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_walker2d-medium-expert-v2 --dataset_name Percent100_05True --model PT+ADT --job_type ADTHighG --gt_job_type NewNorm --mistake_rate 0.2"
######

##SUBMITTED 2
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_walker2d-medium-replay-v2 --dataset_name Percent100_05True --model SARA --job_type NoMask_ESFalse_NoSrc_cpTrue_RandomSubseqs --gt_job_type NewNorm --mistake_rate 0.2" 
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_walker2d-medium-replay-v2 --dataset_name Percent100_05True --model PT --job_type None --gt_job_type NewNorm --mistake_rate 0.2" 
# apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_walker2d-medium-replay-v2 --dataset_name Percent100_05True --model PT+ADT --job_type ADTHighG --gt_job_type NewNorm --mistake_rate 0.2"

apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_kitchen-mixed-v0 --dataset_name Percent100_05True --model SARA --job_type NoMask_ESFalse_e16RandomSubseqs_allSeq4k1LCorrLen --gt_job_type NewNormCorrLen --mistake_rate 0.2" 
apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_kitchen-mixed-v0 --dataset_name Percent100_05True --model PT --job_type CorrLen --gt_job_type NewNormCorrLen --mistake_rate 0.2" 
apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_kitchen-mixed-v0 --dataset_name Percent100_05True --model PT+ADT --job_type ADTHighG_CorrLen --gt_job_type NewNormCorrLen --mistake_rate 0.2"

apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_kitchen-partial-v0 --dataset_name Percent100_05True --model SARA --job_type NoMask_ESFalse_e16RandomSubseqs_allSeq4k1LCorrLen --gt_job_type NewNormCorrLen --mistake_rate 0.2" 
apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_kitchen-partial-v0 --dataset_name Percent100_05True --model PT --job_type CorrLen --gt_job_type NewNormCorrLen --mistake_rate 0.2" 
apptainer exec --nv instance://$INSTANCE_NAME bash -c "export PYTHONPATH=$SING_PYTHONPATH && cd $PROJ_DIR/SARA_PbRL && python run_correlations_batch.py --project IQL_kitchen-partial-v0 --dataset_name Percent100_05True --model PT+ADT --job_type ADTHighG_CorrLen --gt_job_type NewNormCorrLen --mistake_rate 0.2"