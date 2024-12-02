#!/bin/bash
#SBATCH -c 8 # Number of threads
#SBATCH -t 0-00:30:00 # Amount of time needed DD-HH:MM:SS
#SBATCH -p sapphire # Partition to submit to
#SBATCH --mem-per-cpu=100 #Memory per cpu
module load intel/21.2.0-fasrc01
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK MYPROGRAM > output.txt 2> errors.txt