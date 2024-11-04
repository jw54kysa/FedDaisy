#! /bin/bash --
#SBATCH -J gpu-064
#SBATCH --ntasks=1
#SBATCH --gpus=v100
#SBATCH --time=8-00:00:00
#SBATCH -o $HOME/Documents/Projects/cudamuca/jobfiles/log/GPU/gpu-potts-064.o-%a
#SBATCH -e $HOME/Documents/Projects/cudamuca/jobfiles/log/GPU/gpu-potts-064.e-%a

# change cluster
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
pip install --upgrade pip
pip install -r requirements.txt

numclients=150
batch_size=64
numdat=64
numrounds=10000
schedrounds=2500
cuda_device=1

for seed in 1 2 #3
do
     daisy=1
     avg=10
     CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python -u feddc_CIFAR10_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log

#     daisy=1
#     CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_CIFAR10_pytorch.py --run-local-epochs --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log


#     daisy=10001
#     avg=1
#     CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_CIFAR10_pytorch.py --run-local-epochs --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log
#
#
#     daisy=10001
#     avg=10
#
#     CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_CIFAR10_pytorch.py --run-local-epochs --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg --seed $seed | tee CompExp_Cifar10_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log
done
