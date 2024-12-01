#!/bin/bash --
#SBATCH --job-name=runProbPermExp
#SBATCH --partition=paula
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus=a30:4
#SBATCH --cpus-per-task=4
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
numrounds=10000
#schedrounds=2500

seed=1

daisy=1
avg=10

iid='randsize'
min=8
max=64
per='rand' #'prob'
#    --with-amp \

srun singularity exec --nv FEDDC.sif \
python3.9 -u feddc_CIFAR10_pytorch_test_prob_perm.py \
    --optimizer SGD \
    --train-batch-size $batch_size \
    --lr 0.1 \
    --num-clients $numclients \
    --num-rounds $numrounds \
    --num-samples-per-client $numdat \
    --report-rounds 250 \
    --daisy-rounds $daisy \
    --iid-data "$iid" \
    --min-samples $min \
    --max-samples $max \
    --permutation "$per" \
    --aggregate-rounds $avg \
    --seed $seed \
    | tee CompExp_Cifar10_iid${iid}_${per}${min}${max}_nc${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_r${numrounds}_s${seed}.log
