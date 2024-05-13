import torch
import random
import argparse
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import os
import pwd
import yaml
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.simulation import NLICADataset
import models.simulation as sim_models
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"

# sfreq for broderick2019
sfreq = 128.0

# H for high, M for middle, and L for low mode
mode_dict = {'H': {'stride': -1, 'lfreq': 0.1, 'hfreq': 1.0, 'sfreq': 2.0, 'ch': 42, 'directory': 'bestruns20_stride-1', 'data_path': 'lfreq0.1_hfreq1.0_sfreq2.0_ch42_len10'},
             'M': {'stride': 1, 'lfreq': 1.0, 'hfreq': 10.0, 'sfreq': 16.0, 'ch': 42, 'directory': 'bestruns20_stride1', 'data_path': 'lfreq1.0_hfreq10.0_sfreq16.0_ch42_len10'},
             'L': {'stride': 8, 'lfreq': 10.0, 'hfreq': None, 'sfreq': 128.0, 'ch': 42, 'directory': 'bestruns20_stride8', 'data_path': 'lfreq10.0_hfreqNone_sfreq128.0_ch42_len10'}}

def main(args):
    # seed everything
    config = yaml.safe_load(open(args.config, 'r'))

    # Evaluate on multiple levels of models for THE-EEG
    for mode in args.mode:
        print("Processing mode", mode)
        # Modify subject and run index
        data_path = config['dataset']['data_path']
        data_path = data_path.replace("SubjectX", "Subject"+args.subject)
        data_path = data_path.replace("RunX", "Run"+args.run)
        data_path = data_path.replace("XXX", mode_dict[mode]['data_path'])
        
        if not os.path.exists(data_path):
            print(f"{data_path} does not exists!")
            return
        
        print(f"Start processing subject {args.subject}'s run {args.run} with path {data_path}")
        pl.seed_everything(args.seed)
        data = NLICADataset(data_path=data_path)
    
        eval_loader = DataLoader(data,
                                 shuffle=False,
                                 batch_size=1,
                                 num_workers=config['dataloader']['num_workers'],
                                 pin_memory=config['dataloader']['pin_memory'])

        # Configure checkpoint path
        dataset = 'broderick2019_eeg'
        data_folder = f"subject{args.subject}_bestrun20_stride{mode_dict[mode]['stride']}_lfreq{mode_dict[mode]['lfreq']}_hfreq{mode_dict[mode]['hfreq']}_sfreq{mode_dict[mode]['sfreq']}_ch{mode_dict[mode]['ch']}_len10"
        param = 'z128_c1_lags2_len8_Nlayer3'
        version = 0
        checkpoint_path = f'outputs/{dataset}/{data_folder}/{param}/tdrl/lightning_logs/version_{version}/checkpoints/'

        
        # List all checkpoints in the directory and sort them
        ckpts = sorted(os.listdir(checkpoint_path))
        last_ckpt_path = os.path.join(checkpoint_path, ckpts[-1])
        print(f"Path for checkpoint: {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path)

        # Extract the model state_dict from the checkpoint
        model_state_dict = checkpoint['state_dict']

        model_class = getattr(sim_models, config['model'])
        model = model_class(**config['model_kwargs'])
        model.load_state_dict(model_state_dict)
    
        # Set the model to evaluation mode
        model.eval()
        

        record = {'X': [], 'X_recon':[], 'Z_est': []}

        # Perform evaluation
        with torch.no_grad():
            for batch in eval_loader:
                # (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
                x, _, _ = batch
            
                '''
                x_recon, mus, logvars, z_est = model.net(x)

                record['X'].extend(list(x.numpy()))
                record['X_recon'].extend(list(x_recon.numpy()))
                record['Z_est'].extend(list(z_est.numpy()))
                '''

                record['X'].extend(list(x))
                record['X_recon'].extend(list(x))
                record['Z_est'].extend(list(x))

        record['X'] = np.array(record['X'])
        record['X_recon'] = np.array(record['X_recon'])
        record['Z_est'] = np.array(record['Z_est'])

        print(record['X'].shape)
        print(record['X_recon'].shape)
        print(record['Z_est'].shape)

        # Check if the directory exists, if not, create it
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        # Save the dictionary to a .npz file
        np.savez(data_path + 'x_z.npz', **record)
        print("Completing writing data to ", data_path + 'x_z.npz')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c',
        '--config',
        type=str,
        required=True
    )
    argparser.add_argument(
        '-s',
        '--subject',
        type=str,
        default=1
    )
    argparser.add_argument(
        '-r',
        '--run',
        type=str,
        default=1
    )
    argparser.add_argument(
        '-m',
        '--mode',
        type=str,
        required=True
    )
    argparser.add_argument(
        '-e',
        '--seed',
        type=int,
        default=770
    )

    args = argparser.parse_args()
    main(args)
