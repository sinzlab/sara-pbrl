from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

import optax
import numpy as np
from flax.training.train_state import TrainState

from .jax_utils import next_rng, value_and_multi_grad, mse_loss, cross_ent_loss, kld_loss
from jax import lax

class PrefTransformerADT(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256

        config.tauMax=.3
        config.gamma=0.003

        config.train_type = "mean"

        # Weighted Sum option
        config.use_weighted_sum = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, trans):
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'adamw': optax.adamw,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            'CosineDecay': optax.warmup_cosine_decay_schedule(
                init_value=self.config.trans_lr,
                peak_value=self.config.trans_lr * 10,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=self.config.trans_lr
            ),
            "OnlyWarmup": optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=self.config.trans_lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(
                        value=self.config.trans_lr
                    )
                ],
                [self.config.warmup_steps]
            ),
            'none': None
        }[self.config.scheduler_type]

        if scheduler_class:
            tx = optimizer_class(scheduler_class)
        else:
            tx = optimizer_class(learning_rate=self.config.trans_lr)

        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.zeros((10, 25, self.action_dim)),
            jnp.ones((10, 25), dtype=jnp.int32)
        )
        self._train_states['trans'] = TrainState.create(
            params=trans_params,
            tx=tx,
            apply_fn=None
        )

        model_keys = ['trans']
        self._model_keys = tuple(model_keys)
        self._total_steps = 0
        
       
    def evaluation(self, batch, curr_epoch):
        tau=min(self.config.gamma*curr_epoch,self.config.tauMax)
        num_keep = jnp.floor((1.0 - tau) * batch['observations'].shape[0]).astype(jnp.int32)
        
        metrics = self._eval_pref_step(
            self._train_states, next_rng(), batch, num_keep
        )
        return metrics

    def get_reward(self, batch):
        return self._get_reward_step(self._train_states, batch)

    @partial(jax.jit, static_argnames=('self'))
    def _get_reward_step(self, train_states, batch):
        obs = batch['observations']
        act = batch['actions']
        timestep = batch['timestep']
        # n_obs = batch['next_observations']
        attn_mask = batch['attn_mask']

        train_params = {key: train_states[key].params for key in self.model_keys}
        trans_pred, attn_weights = self.trans.apply(train_params['trans'], obs, act, timestep, attn_mask=attn_mask, reverse=False)
        return trans_pred["value"], attn_weights[-1]
  
    @partial(jax.jit, static_argnames=('self'))
    def _eval_pref_step(self, train_states, rng, batch, num_keep):

        def loss_fn(train_params, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            timestep_1 = batch['timestep_1']
            timestep_2 = batch['timestep_2']
            labels = batch['labels']
          
            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape

            rng, _ = jax.random.split(rng)
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=False, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=False, attn_mask=None, rngs={"dropout": rng})
            
            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
          
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
         
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
          
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target, reduction='none')


            # Assume batch_size = N
            N = trans_loss.shape[0]  # should be static
            sorted_vals = jnp.sort(trans_loss)

            # max size you might slice (must be static, e.g. batch size)
            max_keep = N

            # Make a full-size slice (static)
            full_slice = jax.lax.dynamic_slice(sorted_vals, (0,), (max_keep,))  # okay since max_keep = N

            # Create a mask to zero out the extras
            mask = jnp.arange(max_keep) < num_keep  # shape (N,)
            lowest_vals = full_slice * mask
            mean_loss = jnp.sum(lowest_vals)/num_keep
            cse_loss = mean_loss

            loss_collection['trans'] = mean_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
            eval_cse_loss=aux_values['cse_loss'],
            eval_trans_loss=aux_values['cse_loss'], #aux_values['trans_loss'],
            eval_num_kept=num_keep
        )

        return metrics
    

    # def keep_below_tau(self,lossVals,thresh):
    #     num_samples = lossVals.shape[0]
    #     num_keep = jnp.floor((1.0 - thresh) * num_samples).astype(jnp.int32)

    #     # Get indices of losses sorted from smallest to largest
    #     sorted_indices = jnp.argsort(lossVals)

    #     # Select indices of samples to keep
    #     keep_indices = lax.slice(sorted_indices,
    #                         start_indices=(0,),
    #                         limit_indices=(num_keep,),
    #                         strides=(1,))

    #     # Keep only the lowest losses
    #     filtered_loss = lossVals[keep_indices]

    #     filtered_loss = jnp.mean(filtered_loss)
    #     return filtered_loss
      
    def train(self, batch, curr_epoch):
        self._total_steps += 1
        tau=min(self.config.gamma*curr_epoch,self.config.tauMax)
        num_keep = jnp.floor((1.0 - tau) * batch['observations'].shape[0]).astype(jnp.int32)
        
        self._train_states, metrics = self._train_pref_step(
            self._train_states, next_rng(), batch, num_keep
        )
        return metrics


    @partial(jax.jit, static_argnames=('self'))
    def _train_pref_step(self, train_states, rng, batch, num_keep):

        def loss_fn(train_params, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            timestep_1 = batch['timestep_1']
            timestep_2 = batch['timestep_2']
            labels = batch['labels']
          
            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape

            rng, _ = jax.random.split(rng)
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
           
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
           
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
           
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target, reduction='none')


            # Assume batch_size = N
            N = trans_loss.shape[0]  # should be static
            sorted_vals = jnp.sort(trans_loss)

            # max size you might slice (must be static, e.g. batch size)
            max_keep = N

            # Make a full-size slice (static)
            full_slice = jax.lax.dynamic_slice(sorted_vals, (0,), (max_keep,))  # okay since max_keep = N

            # Create a mask to zero out the extras
            mask = jnp.arange(max_keep) < num_keep  # shape (N,)
            lowest_vals = full_slice * mask
            mean_loss = jnp.sum(lowest_vals)/num_keep
            cse_loss = mean_loss

            loss_collection['trans'] = mean_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            cse_loss=aux_values['cse_loss'],
            trans_loss=aux_values['cse_loss'],#aux_values['trans_loss'],
            num_kept=num_keep
        )

        return new_train_states, metrics


   
  

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
