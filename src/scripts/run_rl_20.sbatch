#!/bin/bash
#SBATCH -Jrun_rl_20             # Job name
#SBATCH --account=gts-dmolzahn6-fy20phase3       # charge account
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=0                        # Memory per core
#SBATCH -t4000                                   # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -o/storage/scratch1/8/rharris94/rl_scripts/run_rl_20.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=rharris94@gatech.edu        # E-mail address for notifications
cd /storage/scratch1/8/rharris94/rl_dopf_admm/                           # Change to working directory

module load julia/1.8.0              # Load module dependencies
julia src/original_run_20.jl
