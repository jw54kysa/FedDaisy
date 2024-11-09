#!/bin/bash
#SBATCH -J gen-ds-info
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
##SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=clara
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%A-%x-%a.out
#SBATCH -e logs/%A-%x-%a.error
#SBATCH --mail-type=FAIL,END

export PYTHONNOUSERSITE=1

numclients=150
batch_size=64
numdat=64
numrounds=1000
schedrounds=500
cuda_device=1

seed=1

#srun singularity exec --nv FEDDC.sif python3.9 -u test_cuda.py

daisy=1
avg=10

srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log
