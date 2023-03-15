#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24gb
#SBATCH --time=2:00:00

set -u
#module load singularity/2/2.6.1
cd $SLURM_SUBMIT_DIR
echo "howdy"
echo "singularity exec --nv ~/singularity/pytorch.img python3.6 ./code/test_trained_resnet.py $PY_ARGS"
singularity exec --nv ~/singularity/pytorch.img python3.6 ./code/test_trained_resnet.py $PY_ARGS
