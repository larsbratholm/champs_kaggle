import torch
from torch.utils.data import DataLoader

import logging

from cormorant.models import EdgeCormorant
from cormorant.tests import cormorant_tests

from cormorant.train import TrainCormorant
from cormorant.train import init_argparse, init_file_paths, init_logger, init_cuda
from cormorant.train import init_optimizer, init_scheduler
from cormorant.data.utils_kaggle import init_nmr_kaggle_dataset
from cormorant.cg_lib import global_cg_dict

from cormorant.data.collate import collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')


def main():

    # Initialize arguments -- Just
    args = init_argparse()

    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initializing CG coefficients
    global_cg_dict(maxl=max(args.maxl+args.max_sh), dtype=dtype, device=device)

    # Initialize dataloder
    additional_atom_features = ['zeros']

    edge_features = ['jj_all', 'jj_1', 'jj_2', 'jj_3', '1JHC', '1JHN', '2JHH', '2JHC', '2JHN', \
            '3JHH', '3JHC', '3JHN', 'jj_fc', 'jj_sd', 'jj_pso', 'jj_dso']
    args, datasets, num_species, charge_scale = init_nmr_kaggle_dataset(args, args.datadir, \
            file_name='targets_train_expanded.npz', additional_atom_features=additional_atom_features)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=lambda x: collate_fn(x, edge_features))
                   for split, dataset in datasets.items()}

    # Initialize model
    model = EdgeCormorant(args.num_cg_levels, args.maxl, args.max_sh, args.num_channels, num_species,
                          args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                          args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                          charge_scale, args.gaussian_mask,
                          args.top, args.input, args.num_mpnn_levels, args.num_top_levels,
                          activation=args.top_activation,
                          additional_atom_features=additional_atom_features, num_scalars_in=16,
                          device=device, dtype=dtype)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define a loss function. Just use L2 loss for now.
    loss_fn = torch.nn.functional.mse_loss

    # Apply the covariance and permutation invariance tests.
    cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = TrainCormorant(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint(load_training_state=False)

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()


if __name__ == '__main__':
    main()
