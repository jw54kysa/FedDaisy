#!/bin/bash
#SBATCH --job-name=vanillaDC
#SBATCH --partition=clara
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o log/%x.out-%j
#SBATCH -e log/%x.error-%j
#SBATCH --mail-type=BEGIN,END

# use singularity python
export PYTHONNOUSERSITE=1

numclients=150
batch_size=64
numdat=64
numrounds=10000
schedrounds=2500

# sbatch seed
seed=1

info="vanillafeddc"
# vanilla daisy chaining
srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch_vanilla_daisy_chaining.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --seed $seed | tee CompExp_Cifar10_${info}_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_r${numrounds}_s${seed}.log
