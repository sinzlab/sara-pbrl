# Similarity as Reward Alignment: Robust and Versatile Preference-based Reinforcement Learning

Code for https://arxiv.org/abs/2506.12529. This repo provides the SARA PbRL model in the pbrl folder. 

To run baselines, we also provide our fork of the PreferenceTransformer (https://github.com/csmile-1006/PreferenceTransformer), which includes human labels for the D4RL locomotion tasks. We created the Preference Transformer with ADT baseline in this repo. We also made minor edits to the repo to follow our preferred WandB logging structure, and to use the pre-created preference datasets (ensuring that all models use the exact same datasets).

Likewise, we provide a fork for the DPPO repo (https://github.com/snu-mllab/DPPO). This repo provides the human labels for the kitchen and pen tasks. In addition to modifying the WandB logging structure, we also modified the computation of policy evaluation rewards to do a running average as discussed in our main paper. As in our fork for PreferenceTransformer, we also modified this repo to use the pre-created preference datasets.

We also include our fork of the OfflineRL-Kit (https://github.com/yihaosun1124/OfflineRL-Kit). We have adapted this to run Implicit Q-Learning for offline datasets with rewards computed from Preference Transformer, Preference Transformer with ADT, and the SARA framework. We have made our best effort to match the reward normalization preprocessing steps and hyperparameters of the Preference Transformer implementation of IQL, and we have kept this consistent for all models. The run_PbRLPipeline.py script can be used to run IQL.

## Installation

First build the singularity container using the provided singularity_pbrl.def file:

<pre><code>```singularity build --fakeroot pbrl.sif singularity_pbrl.def ```</code></pre>

## Creating Preference Datasets 

Make the preference datasets from the human labels or script labels by running make_capencdataset. The flag use05 specifies whether or not to include neutral queries (use05=True) or exclude them. Setting scriptLabel to False creates the human labeled datasets. 

Here is an example:

python -m pbrl.make_capencdataset --use05 True --mistake_rate .2 --fraction 1.0 --scriptLabel True --task hopper-medium-replay-v2

The script creates 4 files: pref_dataset.pkl, pref_eval_dataset.pkl, train_set.pkl, and test_set.pkl. The first two are in the required format for PreferenceTransformer, PreferenceTransformer with ADT, and DPPO. The latter two are in the required format for the SARA framework. The pref_eval_dataset is just a small random subset of pref_dataset. Though we create these train/test splits, we only used the train/test splits for model development and HP tuning. For the results reported in the paper, SARA encoder scripts actually combine train and test so that we use all preference data. Likewise, the baselines use all the preference data in pref_dataset.pkl for the results reported in the paper

The combined train_set.pkl and test_set.pkl used by the SARA encoder has the same data as the pref_dataset.pkl, just different formatting.  As noted above we make these datasets first and then all models/seeds for a given experiment use the same datasets. 

## Running SARA model 

Running the SARA encoder on preference data, SARA reward computation for the full offline dataset, and IQL with the offline dataset can be achieved by running the following:

python -m pbrl.run_PbRLPipeline --run_type run_IQL_SARA --task hopper-medium-replay-v2 --enc_epochs 4000 --seed $seed --fraction 1.0 --use05 True --script_label False --mistake_rate 0.0 --enc_configpath $PROJ_DIR/SARA_PbRL/pbrl/configs --save_dir $PROJ_DIR/PbRL --jobInfo jobTypeName

Refer to the paper appendix for the recommended number of enc_epochs for a given task. 

The provided job_pbrl.sh script submits jobs over all 8 seeds for results reported in the paper. 

## Running baselines 

To run Preference Transformer, Preference Transformer with ADT, and DPPO, we need to first train the preference transformers for the first two baselines and the preference predictor for the latter baseline. Then we can run policy training

### Training Preference Transformer baseline

cd PreferenceTransformer/ && python -m JaxPref.new_preference_reward_main  --env hopper-medium-replay-v2 --transformer.embd_dim 256 --transformer.n_layer 1 --transformer.n_head 4 --logging.output_dir 'PbRL/' --batch_size 256 --skip_flag 0 --n_epochs=10000 --seed $seed --fraction 1.0 --use_human_label True --use05 True --mistake_rate 0.0 --model_type PrefTransformer

Then we can compute PT rewards for the full offline dataset and run IQL using our scripts adapted from the OfflineRL-kit:
python -m pbrl.run_PbRLPipeline --task hopper-medium-replay-v2 --run_type run_IQL_PrefTrans --fraction 1.0 --use05 True --seed $seed --mistake_rate 0.0 --script_label False

### Training Preference Transformer with ADT baseline

Similar setup:

cd PreferenceTransformer/ && python -m JaxPref.new_preference_reward_main  --env hopper-medium-replay-v2 --transformer.embd_dim 256 --transformer.n_layer 1 --transformer.n_head 4 --logging.output_dir 'PbRL/' --batch_size 256 --skip_flag 0 --n_epochs=10000 --seed $seed --fraction 1.0 --use_human_label True --use05 True --mistake_rate 0.0 --model_type PrefTransformerADT

Then we can compute PT rewards for the full offline dataset and run IQL using our scripts adapted from the OfflineRL-kit:
python -m pbrl.run_PbRLPipeline --task hopper-medium-replay-v2 --run_type run_IQL_PrefTransADT --fraction 1.0 --use05 True --seed $seed --mistake_rate 0.0 --script_label False

### Training DPPO baseline:

First train the preference predictor model:
cd DPPO/ && python -m JaxPref.new_preference_reward_main --env hopper-medium-replay-v2 --use_human_label True --seed $seed --fraction 1.0 --use05 True --mistake_rate 0.0 --model_type PrefTransformerDPPO

Then train DPPO:
python DPPO/train.py --env_name hopper-medium-replay-v2 --seed $seed --lambd $lambd --dataset_name Percent100_05True

Adjust lambd based on recommended hyperparameters in DPPO paper

## Running modified hopper  

First 

## Post processing

For a given task and model, we did the WandB logging so that the 8 IQL runs for the model fall under a unique GroupName/JobType under the project IQL_{task}. Though the DPPO baseline is not IQL, we also log its results under project IQL_{task} to enable easy comparison with the other models. The running policy evaluation rewards can be compared by grouping by Group and JobType in WandB. For post-processing we will provide our Jupyter Notebook which creates the plots/tables seen in the main paper. 

