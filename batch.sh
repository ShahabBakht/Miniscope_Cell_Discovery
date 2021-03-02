#!/bin/bash
#SBATCH --cpus-per-task=8                       # Ask for 8 CPUs
#SBATCH --gres=gpu:v100:1                           # Ask for 2 GPU
#SBATCH --mem=12                              # Ask for 48 GB of RAM
#SBATCH --time=50:00:00                         # The job will run for 24 hours
#SBATCH -o /network/tmp1/bakhtias/slurm-%j.out  # Write the log on tmp1 (change it to your tmp1 directory)

# 1. load modules
module load anaconda/3
module load pytorch/1.7
#module load cuda/11

# 2. Load your environment
source $CONDA_ACTIVATE
conda activate DeepMouse_new

echo $PATH

python -V

# # 3. Copy your dataset on the compute node (change the source diectory to the location of your dataset)
# cp /network/tmp1/bakhtias/UCF101_full/UCF101_full.zip $SLURM_TMPDIR
# unzip -q $SLURM_TMPDIR/UCF101_full.zip -d $SLURM_TMPDIR

#cp /network/tmp1/bakhtias/Shahab/UCF101catcam.zip $SLURM_TMPDIR
#unzip -q $SLURM_TMPDIR/UCF101catcam.zip -d $SLURM_TMPDIR

# 4. Launch your job

# cd ./process_data/src/
# python write_csv.py

# cd ../../dpc

python main.py
#python main.py --gpu 0,1 --net simmousenet --dataset ucf101 --batch_size 30 --img_dim 64 --epochs 100 --prefix 'simmousenet_seed20_h1p4_largeRF_wmaxpool' --resume '/network/tmp1/bakhtias/Results/log_simmousenet_seed20_h1p4_largeRF_wmaxpool/ucf101-64_rsenet_dpc-rnn_bs30_lr0.001_seq8_pred3_len5_ds3_train-all/model/epoch84.pth.tar' --lr 1e-3 --save_checkpoint_freq 25

# 5. Copy whatever you want to save on $SCRATCH (change the destination directory to your Results folder)
# cp -r $SLURM_TMPDIR/log_simmousenet_seed20_h1p4_realLGN /network/tmp1/bakhtias/Results
