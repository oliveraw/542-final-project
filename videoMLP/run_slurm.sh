#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=videoMLP
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m 
#SBATCH --time=07:59:55
#SBATCH --account=eecs542s001f23_class
#SBATCH --partition=gpu
#SBATCH --gpu_cmode=shared
#SBATCH --gpus=1

# The application(s) to execute along with its input arguments and options:
python3 -m videoMLP.run
