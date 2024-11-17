#!/bin/bash
#SBATCH --job-name=feddc
#SBATCH --partition=paula
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --gpus=a30:4
#SBATCH --mem=128G                    # Adjust memory as needed
#SBATCH --time=24:00:00             # Set a realistic time limit
#SBATCH -o log/%x.out-%j
#SBATCH -e log/%x.error-%j
#SBATCH --mail-type=START,END

# use singularity python
export PYTHONNOUSERSITE=1

numclients=150
batch_size=64
numdat=64
numrounds=10000
schedrounds=2500

# sbatch seed
seed=1 #$SLURM_ARRAY_TASK_ID

#for seed in 1 2 #3
#do
daisy=1
avg=10
srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_r${numrounds}_s${seed}.log

# vanilla training
srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed --run-ablation vanilla_training | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log


#daisy=10001
#avg=1
#srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log
#
#
#daisy=10001
#avg=10
#
#srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch.py  --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log
#done
