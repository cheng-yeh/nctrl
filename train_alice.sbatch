#!/bin/bash
#SBATCH -J train_alice                          # Job name
#SBATCH --account=gts-rs275                     # charge account
#SBATCH -N 1 --ntasks-per-node=1                # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=10G                       # Memory per core
#SBATCH -t 60                                   # Duration of the job (Ex: 15 mins)
#SBATCH -q inferno                              # QOS Name
#SBATCH -o Report-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=cchen847@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3/2022.05                   # Load module dependencies
srun conda create -n nctrl python=3.10 -y
srun conda activate nctrl
srun pip install torch==1.13.1 torchvision torchaudio
srun pip install -r requirements.txt
srun python train_alice.py -c configs/alice_eeg/alice_eeg_nctrl.yaml
