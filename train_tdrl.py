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
#torch.set_float32_matmul_precision('high')
torch.set_float32_matmul_precision('medium')
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"

def main(args):
    # seed everything
    config = yaml.safe_load(open(args.config, 'r'))
    config['dataset']['data_path'] = config['dataset']['data_path'].replace("subject1", "subject"+args.subject)
    config['trainer']['default_root_dir'] = config['trainer']['default_root_dir'].replace("subject1", "subject"+args.subject)
    if not os.path.exists(config['dataset']['data_path']):
        print(f"{config['dataset']['data_path']} does not exists!")
        return
    print(f"Start processing subject {args.subject} with path {config['dataset']['data_path']}")
    print(f"Output files will be saved to {config['trainer']['default_root_dir']}")
    pl.seed_everything(args.seed)
    data = NLICADataset(data_path=config['dataset']['data_path'])
    n_validation = config['dataset']['n_validation']
    train_data, valid_data = random_split(
        data, [len(data) - n_validation, n_validation])

    train_loader = DataLoader(train_data,
                              shuffle=False,
                              batch_size=config['dataloader']['train_batch_size'],
                              num_workers=config['dataloader']['num_workers'],
                              pin_memory=config['dataloader']['pin_memory'])
    valid_loader = DataLoader(valid_data,
                              shuffle=False,
                              batch_size=config['dataloader']['valid_batch_size'],
                              num_workers=config['dataloader']['num_workers'],
                              pin_memory=config['dataloader']['pin_memory'])
    model_class = getattr(sim_models, config['model'])
    model = model_class(**config['model_kwargs'])
    #model.A = data.A
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=config['trainer']['default_root_dir']
    )
    checkpoint_callback = ModelCheckpoint(monitor='val/loss',
                                          save_top_k=3,
                                          mode='min')
    early_stop_callback = EarlyStopping(monitor="val/loss",
                                        stopping_threshold=0.01,
                                        patience=10_000,
                                        verbose=False,
                                        mode="min")
    logger_list = [tb_logger]
    trainer = pl.Trainer(
        logger=logger_list,
        callbacks=[checkpoint_callback,early_stop_callback],
        **config['trainer'],)
    log_dir = Path(trainer.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(config, open(log_dir/'config.yaml', 'w'))
    trainer.fit(model, train_loader, valid_loader)
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
        '-e',
        '--seed',
        type=int,
        default=770
    )
    args = argparser.parse_args()
    main(args)
