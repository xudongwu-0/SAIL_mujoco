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


for SEED in 4 5 6
do
  echo "--------------------------------------------------"
  echo ">>> [PEBBLE] cheetah_run | Seed: $SEED"
  echo "--------------------------------------------------"

  CUDA_VISIBLE_DEVICES=1 python train_PEBBLE.py \
    env=cheetah_run \
    seed=$SEED \
    experiment=PEBBLE_cheetah_run_400k_seed${SEED} \
    num_train_steps=400000 \
    num_unsup_steps=900 \
    num_interact=2000 \
    max_feedback=250 \
    reward_batch=10 \
    reward_update=50 \
    feed_type=1 \
    agent.params.batch_size=512 \
    agent.params.actor_lr=0.0005 \
    agent.params.critic_lr=0.0005

  

  echo "=================================================="
  echo ">>> [TRPO ablation] cheetah_run | Seed: $SEED"
  echo "=================================================="
  CUDA_VISIBLE_DEVICES=1 python train_V_npg.py \
    env=cheetah_run \
    seed=$SEED \
    experiment=TRPO_abl_walker_stand_400k_seed${SEED} \
    agent.params.actor_lr=0.0005 \
    agent.params.critic_lr=0.0005 \
    agent.params.batch_size=512 \
    num_unsup_steps=900 \
    num_train_steps=400000 \
    num_interact=2000 \
    max_feedback=250 \
    reward_batch=10 \
    reward_update=50 \
    feed_type=1 \
    agent.params.revkl_gamma=0 \
    reference_update_frequency=50000 \
    num_improving_steps=5000


  echo "=================================================="
  echo ">>> [TRPO] cheetah_run | Seed: $SEED"
  echo "=================================================="
  CUDA_VISIBLE_DEVICES=1 python train_V_npg.py \
    env=cheetah_run \
    seed=$SEED \
    experiment=TRPO_cheetah_run_400k_seed${SEED} \
    agent.params.actor_lr=0.0005 \
    agent.params.critic_lr=0.0005 \
    agent.params.batch_size=512 \
    num_unsup_steps=900 \
    num_train_steps=400000 \
    num_interact=2000 \
    max_feedback=250 \
    reward_batch=10 \
    reward_update=50 \
    feed_type=1 \
    max_kl=0.1 \
    agent.params.revkl_gamma=1e-4 \
    reference_update_frequency=50000 \
    num_improving_steps=5000

  echo "=================================================="
  echo ">>> [SGD] cheetah_run | Seed: $SEED"
  echo "=================================================="
  CUDA_VISIBLE_DEVICES=1 python train_V.py \
    env=cheetah_run \
    seed=$SEED \
    experiment=SGD_baseline_cheetah_run_400k_seed${SEED} \
    agent.params.actor_lr=0.0005 \
    agent.params.critic_lr=0.0005 \
    agent.params.batch_size=512 \
    num_unsup_steps=900 \
    num_train_steps=400000 \
    num_interact=2000 \
    max_feedback=260 \
    reward_batch=10 \
    reward_update=50 \
    feed_type=1 \
    agent.params.revkl_gamma=0 \
    reference_update_frequency=50000 \
    num_improving_steps=5000
done



echo "=================================================="
echo ">>> ALL EXPERIMENTS COMPLETED!"
echo "=================================================="
