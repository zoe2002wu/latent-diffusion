#!/bin/bash
#SBATCH --job-name=regular
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-15:00
#SBATCH --mem=256G
#SBATCH --output=regular_job.out
#SBATCH --error=regular_job.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
# Activate your virtual environment (if you have one)
source activate ldm

srun python fid_computation.py --num_samples 5064 --batch_size 5 --ddim_steps 25 --penalty_param 2.0 --scale 0.3 --ddim_eta 0.0 