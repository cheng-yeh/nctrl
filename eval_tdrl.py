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

def main(args):
    # seed everything
    config = yaml.safe_load(open(args.config, 'r'))
    pl.seed_everything(args.seed)
    data = NLICADataset(data_path=config['dataset']['data_path'])

    eval_loader = DataLoader(data,
                             shuffle=False,
                             batch_size=1,
                             num_workers=config['dataloader']['num_workers'],
                             pin_memory=config['dataloader']['pin_memory'])
    
    # Load the checkpoint
    dataset = 'broderick2019_eeg'
    data_folder = 'lfreq10_hfreqNone_len10'
    param = 'z128_c1_lags2_len8_Nlayer3'
    version = 8
    ckpt = 'epoch=159-step=17033.ckpt'
    checkpoint_path = f'outputs/{dataset}/{data_folder}/{param}/tdrl/lightning_logs/version_{version}/checkpoints/{ckpt}'
    checkpoint = torch.load(checkpoint_path)

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
            x_recon, mus, logvars, z_est = model.net(x)

            record['X'].extend(list(x.numpy()))
            record['X_recon'].extend(list(x_recon.numpy()))
            record['Z_est'].extend(list(z_est.numpy()))

    record['X'] = np.array(record['X'])
    record['X_recon'] = np.array(record['X_recon'])
    record['Z_est'] = np.array(record['Z_est'])

    print(record['X'].shape)
    print(record['X_recon'].shape)
    print(record['Z_est'].shape)

    # Save the dictionary to a .npz file
    np.savez(config['dataset']['data_path'] + 'x_z.npz', **record)

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
        '--seed',
        type=int,
        default=770
    )
    args = argparser.parse_args()
    main(args)
