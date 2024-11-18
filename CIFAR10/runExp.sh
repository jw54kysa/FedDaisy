#!/bin/bash
#SBATCH --job-name=feddc
#SBATCH --partition=paula
#SBATCH -N 3
#SBATCH --ntasks=3
#SBATCH --gpus=a30:6
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

info=""
daisy=1
avg=10
srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${info}_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_r${numrounds}_s${seed}.log

info="central"
# vanilla central training
srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed --run-ablation vanilla_training | tee CompExp_Cifar10_${info}_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log

info="vanillafeddc"
# vanilla daisy chaining
srun singularity exec --nv FEDDC.sif python3.9 -u feddc_CIFAR10_pytorch_vanilla_daisy_chaining.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${info}_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_r${numrounds}_s${seed}.log
