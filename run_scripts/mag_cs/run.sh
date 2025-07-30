#!/bin/bash

source ~/.bashrc

# module load anaconda/2020.07-Python3.8-gcc8
# export PROJECT=xx
# export CONDA_ENVS=/scratch/$PROJECT/xx/conda_envs
# source activate $CONDA_ENVS/sagad
# module load cuda/11.3

# cd wd to the project root
cd /media/nvme1/pycharm_mirror/nsreg_release

source activate sagad
MAIN="#<change to project root dir>/exp/${1}.py"
WORK_DIR= # set to the project root dir

echo "running ${MAIN}"

export PYTHONPATH=$WORK_DIR

# export OMP_NUM_THREADS=52

SEED=42

if [ $# -eq 4 ]
then
    SEED=$3
fi

if [ $# -eq 4 ]
then
    BATCH_ID=$4
fi


python -u $MAIN \
--proj_dir $WORK_DIR \
--config_dir mag_cs \
--meta_config_fn $2 \
--seed $SEED \
