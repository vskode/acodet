#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o acodet_dir/jobs/slurm.%N.%j_ilaria_effnet.out # STDOUT
#SBATCH -e acodet_dir/jobs/slurm.%N.%j_ilaria_effnet.err # STDERR
#SBATCH --gres=gpu:1
cd acodet_dir/acodet
source env_acodetpip/bin/activate
python run.py
