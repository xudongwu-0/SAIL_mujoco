cd "$ROOT"
ROOT="$(cd "$(dirname "$0")" && pwd)"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mrn


# ==========================================

export MUJOCO_ROOT="${MUJOCO_ROOT:-$HOME/.mujoco}"
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_ROOT}/mujoco200

export CONDA_HOME=$CONDA_PREFIX
export CFLAGS="-I$CONDA_HOME/include -I$CONDA_HOME/include/GL $CFLAGS"
export LDFLAGS="-L$CONDA_HOME/lib -Wl,-rpath,$CONDA_HOME/lib $LDFLAGS"
export CPATH=$CONDA_HOME/include
export LIBRARY_PATH=$CONDA_HOME/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_ROOT/mujoco200/bin:/usr/lib/nvidia:$CONDA_HOME/lib


rm -rf ~/.mujoco/mujoco-py/build

echo " Starting Training..."


for seed in 10 11 12
do
  echo "--------------------------------------------------"
  echo ">>> [PEBBLE] Running metaworld_door-open-v2 | Seed: $seed"
  echo "--------------------------------------------------"

  CUDA_VISIBLE_DEVICES=2 python train_PEBBLE.py \
    env=metaworld_door-open-v2 \
    seed=$seed \
    experiment=PEBBLE_metaworld_baseline_400k_seed$seed \
    agent.params.actor_lr=0.0003 \
    agent.params.critic_lr=0.0003 \
    num_unsup_steps=900 \
    num_train_steps=400000 \
    agent.params.batch_size=512 \
    num_interact=500 \
    max_feedback=900 \
    reward_batch=10 \
    reward_update=10 \
    feed_type=1
  
  echo "--------------------------------------------------"
  echo ">>> [TRPO] Running metaworld_door-open-v2 | Seed: $seed"
  echo "--------------------------------------------------"
  CUDA_VISIBLE_DEVICES=2 python train_V_npg.py \
    env=metaworld_door-open-v2 \
    seed=$seed \
    experiment=fullTRPO_door_400k_seed$seed \
    agent.params.actor_lr=0.0003 \
    agent.params.critic_lr=0.0003 \
    agent.params.batch_size=512 \
    num_unsup_steps=900 \
    num_train_steps=400000 \
    num_interact=500 \
    max_feedback=1000 \
    reward_batch=10 \
    reward_update=10 \
    feed_type=1 \
    max_kl=0.1 \
    reference_update_frequency=50000 \
    num_improving_steps=5000

  echo "--------------------------------------------------"
  echo ">>> [TRPO abla] Running metaworld_door-open-v2 | Seed: $seed"
  echo "--------------------------------------------------"
  CUDA_VISIBLE_DEVICES=2 python train_V_npg.py \
    env=metaworld_door-open-v2 \
    seed=$seed \
    experiment=TRPO_ab_door_400k_seed$seed \
    agent.params.actor_lr=0.0003 \
    agent.params.critic_lr=0.0003 \
    agent.params.batch_size=512 \
    num_unsup_steps=900 \
    num_train_steps=400000 \
    num_interact=500 \
    max_feedback=1000 \
    reward_batch=10 \
    reward_update=10 \
    feed_type=1 \
    reference_update_frequency=50000 \
    num_improving_steps=5000

  echo "--------------------------------------------------"
  echo ">>> [SGD] Running metaworld_door-open-v2 | Seed: $seed"
  echo "--------------------------------------------------"
  CUDA_VISIBLE_DEVICES=1 python train_V.py \
    env=metaworld_door-open-v2 \
    seed=$seed \
    experiment=TRPO_ab_door_400k_seed$seed \
    agent.params.actor_lr=0.0003 \
    agent.params.critic_lr=0.0003 \
    agent.params.batch_size=512 \
    num_unsup_steps=900 \
    num_train_steps=400000 \
    num_interact=500 \
    max_feedback=1000 \
    reward_batch=10 \
    reward_update=10 \
    feed_type=1 \
    reference_update_frequency=50000 \
    num_improving_steps=5000
done



echo "=================================================="
echo ">>> ALL EXPERIMENTS COMPLETED!"
echo "=================================================="
