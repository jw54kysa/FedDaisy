#!/bin/bash --
#SBATCH --job-name=runMyExp
#SBATCH --partition=paula
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gpus=a30:4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH -o log/%x.out-%j
#SBATCH -e log/%x.error-%j
#SBATCH --mail-type=BEGIN,END

# use singularity python
export PYTHONNOUSERSITE=1

numclients=150
batch_size=64
numdat=64
numrounds=1000
schedrounds=200

seed=1

daisy=1
avg=10

iid='randlen'
min=8
max=64
per='rand'

srun singularity exec --nv FEDDC.sif \
python3.9 -u feddc_CIFAR10_pytorch.py \
    --optimizer SGD \
    --train-batch-size $batch_size \
    --lr 0.1 \
    --lr-schedule-ep ${schedrounds} \
    --lr-change-rate 0.5 \
    --num-clients $numclients \
    --num-rounds $numrounds \
    --num-samples-per-client $numdat \
    --report-rounds 250 \
    --daisy-rounds $daisy \
    --iid-data $iid \
    --min-samples $min \
    --max-samples $max \
    --permutation $per \
    --aggregate-rounds $avg \
    --seed $seed \
    | tee CompExp_Cifar10_iid${iid}_${per}${min}${max}_nc${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log

daisy=10
avg=100

srun singularity exec --nv FEDDC.sif \
python3.9 -u feddc_CIFAR10_pytorch.py \
    --optimizer SGD \
    --train-batch-size $batch_size \
    --lr 0.1 \
    --lr-schedule-ep ${schedrounds} \
    --lr-change-rate 0.5 \
    --num-clients $numclients \
    --num-rounds $numrounds \
    --num-samples-per-client $numdat \
    --report-rounds 250 \
    --daisy-rounds $daisy \
    --iid-data $iid \
    --min-samples $min \
    --max-samples $max \
    --permutation $per \
    --aggregate-rounds $avg \
    --seed $seed \
    | tee CompExp_Cifar10_iid${iid}_${per}${min}${max}_nc${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log

daisy=1
avg=10
per='prob'

srun singularity exec --nv FEDDC.sif \
python3.9 -u feddc_CIFAR10_pytorch.py \
    --optimizer SGD \
    --train-batch-size $batch_size \
    --lr 0.1 \
    --lr-schedule-ep ${schedrounds} \
    --lr-change-rate 0.5 \
    --num-clients $numclients \
    --num-rounds $numrounds \
    --num-samples-per-client $numdat \
    --report-rounds 250 \
    --daisy-rounds $daisy \
    --iid-data $iid \
    --min-samples $min \
    --max-samples $max \
    --permutation $per \
    --aggregate-rounds $avg \
    --seed $seed \
    | tee CompExp_Cifar10_iid${iid}_${per}${min}${max}_nc${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log

daisy=10
avg=100

srun singularity exec --nv FEDDC.sif \
python3.9 -u feddc_CIFAR10_pytorch.py \
    --optimizer SGD \
    --train-batch-size $batch_size \
    --lr 0.1 \
    --lr-schedule-ep ${schedrounds} \
    --lr-change-rate 0.5 \
    --num-clients $numclients \
    --num-rounds $numrounds \
    --num-samples-per-client $numdat \
    --report-rounds 250 \
    --daisy-rounds $daisy \
    --iid-data $iid \
    --min-samples $min \
    --max-samples $max \
    --permutation $per \
    --aggregate-rounds $avg \
    --seed $seed \
    | tee CompExp_Cifar10_iid${iid}_${per}${min}${max}_nc${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log
