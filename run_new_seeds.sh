#!/bin/bash

# # --- 激活您的 Conda 环境 ---
# echo "--- Activating Conda environment: SAIL ---"
# # source ~/miniconda3/etc/profile.d/conda.sh
# conda activate SAIL
# echo "--- Conda environment activated ---"


# echo "--- Starting New Seed (123) for Ablation (kl=0) ---"
# python train_V.py \
# env=walker_walk \
# seed=123 \
# experiment=PEBBLE_RevKL0_250k_MoreData_seed123 \
# num_train_steps=250000 \
# agent.params.actor_lr=0.0005 \
# agent.params.critic_lr=0.0005 \
# num_unsup_steps=900 \
# num_interact=2000 \
# max_feedback=250 \
# reward_batch=10 \
# reward_update=50 \
# feed_type=1 \
# agent.params.revkl_gamma=0 \
# reference_update_frequency=50000 \
# num_improving_steps=25000


# echo "--- Starting New Seed (1) for Ablation (kl=0) ---"
# python train_V.py \
# env=walker_walk \
# seed=1 \
# experiment=PEBBLE_RevKL0_250k_MoreData_seed1 \
# num_train_steps=250000 \
# agent.params.actor_lr=0.0005 \
# agent.params.critic_lr=0.0005 \
# num_unsup_steps=900 \
# num_interact=2000 \
# max_feedback=250 \
# reward_batch=10 \
# reward_update=50 \
# feed_type=1 \
# agent.params.revkl_gamma=0 \
# reference_update_frequency=50000 \
# num_improving_steps=25000


# echo "--- Starting New Seed (123) for PEBBLE Baseline ---"
# python train_PEBBLE.py \
# env=walker_walk \
# seed=123 \
# experiment=PEBBLE_baseline_250k_MoreData_seed123 \
# num_train_steps=250000 \
# agent.params.actor_lr=0.0005 \
# agent.params.critic_lr=0.0005 \
# num_unsup_steps=900 \
# num_interact=2000 \
# max_feedback=250 \
# reward_batch=10 \
# reward_update=50 \
# feed_type=1


# echo "--- Starting New Seed (1) for PEBBLE Baseline ---"
# python train_PEBBLE.py \
# env=walker_walk \
# seed=1 \
# experiment=PEBBLE_baseline_250k_MoreData_seed1 \
# num_train_steps=250000 \
# agent.params.actor_lr=0.0005 \
# agent.params.critic_lr=0.0005 \
# num_unsup_steps=900 \
# num_interact=2000 \
# max_feedback=250 \
# reward_batch=10 \
# reward_update=50 \
# feed_type=1


# echo "--- Starting Experiment: Metaworld with RevKL=1.0e-4 seed123 ---"

# python train_V.py \
# env=metaworld_door-open-v2 \
# seed=123 \
# experiment=metaworld_door-open_Revkl_1.0e-4_250k_seed123 \
# agent.params.actor_lr=0.0003 \
# agent.params.critic_lr=0.0003 \
# num_unsup_steps=900 \
# num_train_steps=250000 \
# agent.params.batch_size=512 \
# num_interact=500 \
# max_feedback=1000 \
# reward_batch=10 \
# reward_update=10 \
# feed_type=1 \
# agent.params.revkl_gamma=1.0e-4 \
# reference_update_frequency=50000 \
# num_improving_steps=20000

# echo "---Starting Experiment : Metaworld PEBBLE Baseline seed123---"

# python train_PEBBLE.py \
# env=metaworld_door-open-v2 \
# seed=123 \
# experiment=PEBBLE_metaworld_baseline_250k_seed123 \
# agent.params.actor_lr=0.0003 \
# agent.params.critic_lr=0.0003 \
# num_unsup_steps=900 \
# num_train_steps=250000 \
# agent.params.batch_size=512 \
# num_interact=500 \
# max_feedback=1000 \
# reward_batch=10 \
# reward_update=10 \
# feed_type=1

echo "--- Starting Experiment : Metaworld with RevKL=0 seed123---"

python train_V.py \
env=metaworld_door-open-v2 \
seed=123 \
experiment=metaworld_door-open_Revkl_0_250k_seed123 \
agent.params.actor_lr=0.0003 \
agent.params.critic_lr=0.0003 \
num_unsup_steps=900 \
num_train_steps=250000 \
agent.params.batch_size=512 \
num_interact=500 \
max_feedback=1000 \
reward_batch=10 \
reward_update=10 \
feed_type=1 \
agent.params.revkl_gamma=0 \
reference_update_frequency=50000 \
num_improving_steps=20000

echo "--- Starting Experiment: Metaworld with RevKL=1.0e-4 seed1 ---"

python train_V.py \
env=metaworld_door-open-v2 \
seed=1 \
experiment=metaworld_door-open_Revkl_1.0e-4_250k_seed1 \
agent.params.actor_lr=0.0003 \
agent.params.critic_lr=0.0003 \
num_unsup_steps=900 \
num_train_steps=250000 \
agent.params.batch_size=512 \
num_interact=500 \
max_feedback=1000 \
reward_batch=10 \
reward_update=10 \
feed_type=1 \
agent.params.revkl_gamma=1.0e-4 \
reference_update_frequency=50000 \
num_improving_steps=20000

echo "---Starting Experiment : Metaworld PEBBLE Baseline seed1---"

python train_PEBBLE.py \
env=metaworld_door-open-v2 \
seed=1 \
experiment=PEBBLE_metaworld_baseline_250k_seed1 \
agent.params.actor_lr=0.0003 \
agent.params.critic_lr=0.0003 \
num_unsup_steps=900 \
num_train_steps=250000 \
agent.params.batch_size=512 \
num_interact=500 \
max_feedback=1000 \
reward_batch=10 \
reward_update=10 \
feed_type=1

echo "--- Starting Experiment : Metaworld with RevKL=0 seed1---"

python train_V.py \
env=metaworld_door-open-v2 \
seed=1 \
experiment=metaworld_door-open_Revkl_0_250k_seed1 \
agent.params.actor_lr=0.0003 \
agent.params.critic_lr=0.0003 \
num_unsup_steps=900 \
num_train_steps=250000 \
agent.params.batch_size=512 \
num_interact=500 \
max_feedback=1000 \
reward_batch=10 \
reward_update=10 \
feed_type=1 \
agent.params.revkl_gamma=0 \
reference_update_frequency=50000 \
num_improving_steps=20000



echo "--- All new seed experiments finished! ---"
