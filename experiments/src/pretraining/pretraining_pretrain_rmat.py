import os
from collections import Counter
from typing import Optional, List, Tuple, Any, Callable, Type

import gin
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Metric
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from experiments.src import TrainingModule
from experiments.src.gin import get_default_experiment_name
from experiments.src.training.training_utils import GinModel, get_default_callbacks, get_custom_callbacks, \
    get_default_loggers, get_neptune_logger, apply_neptune_logger, get_optimizer, get_loss_fn, get_metric_cls
from src.huggingmolecules import RMatFeaturizer
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin
from src.huggingmolecules.featurization.featurization_mat_utils import pad_sequence
from src.huggingmolecules.featurization.featurization_rmat import RMatMoleculeEncoding, RMatBatchEncoding
from src.huggingmolecules.models.models_api import PretrainedModelBase


def get_atom_context(atom):
    context_list = []
    for neighbour, bond in zip(atom.GetNeighbors(), atom.GetBonds()):
        context_list.append(f'{neighbour.GetSymbol()}-{bond.GetBondTypeAsDouble()}')

    atom_context = f'{atom.GetSymbol()}'
    for k, v in Counter(context_list).items():
        atom_context += f'_{k}-{v}'

    return atom_context


class RMatPretriningDataset(Dataset[RMatMoleculeEncoding]):
    def __init__(self, featurizer: RMatFeaturizer, dataset_path: str, split: str, context_dict_path: str = None):
        self.featurizer = featurizer
        df = pd.read_csv(dataset_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.context_dict = {'other': 0, 'C_C-1.0-1': 1, 'O_C-1.0-1': 2}
        # with open(context_dict_path, 'rb') as fp:
        #     self.context_dict = pickle.load(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple[RMatMoleculeEncoding, Any]:
        smiles = self.df.iloc[index]['smiles']
        encodings = self.featurizer._encode_smiles(smiles, y=None)
        context = ['other'] + [get_atom_context(atom) for atom in encodings.mol.GetAtoms()]
        context = [(x if x in self.context_dict else 'other') for x in context]
        context = [self.context_dict[x] for x in context]
        return encodings, context

    def collate(self, items: List[Tuple[RMatMoleculeEncoding, Any]]) -> Tuple[RMatBatchEncoding, Any]:
        encodings, contexts = zip(*items)
        encodings_batch = self.featurizer._collate_encodings(encodings)
        contexts = pad_sequence([torch.tensor(c).long() for c in contexts])
        return encodings_batch, contexts


def create_mask(shape: Tuple[int, int], ones_rate: float, device):
    return torch.empty(*shape, device=device).uniform_() <= ones_rate


def create_atom_mask(
        batch_mask: torch.Tensor, drop_probability: float, device
) -> torch.Tensor:
    """
    shape: BATCH x MAX_ATOMS_NUM
    output: BATCH x MAX_ATOMS_NUM

    Example:
    [ 1, 1, 1, 1, 1, 0, 0, 0]
    [ 1, 1, 1, 1, 1, 1, 0, 0]
    [ 1, 1, 1, 1, 0, 0, 0, 0]
    Output:
    [ 0, 0, 1, 0, 0, 0, 0, 0 ]
    [ 0, 1, 0, 1, 0, 0, 0, 0 ]
    [ 0, 1, 0, 0, 0, 0, 0, 0 ]
    """
    initial_mask = create_mask(batch_mask.shape, drop_probability, device)
    atom_mask = (
            batch_mask & initial_mask
    )  # keep ones only in "droppable" existing atoms

    atom_mask[:, 0] = 0  # dummy node
    return atom_mask


class PretrainTrainingModule(TrainingModule):
    def __init__(self, model: PretrainedModelBase, *, loss_fn: Callable[[Tensor, Tensor], Tensor],
                 optimizer: torch.optim.Optimizer, metric_cls: Type[Metric], mask_ratio: float = 0.1):
        super().__init__(model, loss_fn=loss_fn, optimizer=optimizer, metric_cls=metric_cls)
        self.mask_ratio = mask_ratio

    def forward(self, batch):
        return self.model.forward(batch)

    def _mask_batch(self, batch: RMatBatchEncoding, context: Tensor) -> Tuple[RMatBatchEncoding, Tensor]:
        correct_mask = False
        while not correct_mask:
            atom_mask = create_atom_mask(
                batch.batch_mask,
                drop_probability=self.mask_ratio,
                device=batch.node_features.device,
            )
            correct_mask = atom_mask.sum() > 0
        # Mask edges
        adj = batch.relative_matrix[:, :2].sum(axis=1)  # sprawdzic czy to ok

        to_mask = adj.bool() & atom_mask.unsqueeze(2)
        to_mask_transpose = torch.transpose(to_mask, -1, -2)

        batch.bond_features = batch.bond_features.permute(1, 0, 2, 3)
        batch.bond_features[0, to_mask] = 1
        batch.bond_features[1:, to_mask] = 0
        batch.bond_features[0, to_mask_transpose] = 1
        batch.bond_features[1:, to_mask_transpose] = 0
        batch.bond_features = batch.bond_features.permute(1, 0, 2, 3)

        # Mask atoms
        all_atom_mask = batch.bond_features[:, 0].sum(dim=-1) > 0
        batch.node_features = batch.node_features.permute(2, 0, 1)
        batch.node_features[0, all_atom_mask] = 1
        batch.node_features[1:, all_atom_mask] = 0
        batch.node_features = batch.node_features.permute(1, 2, 0)

        return batch, atom_mask

    def _step(self, mode: str, batch, batch_idx: int) -> torch.Tensor:
        batch, context = batch
        batch, atom_mask = self._mask_batch(batch, context)

        output = self.forward(batch)
        output = output[atom_mask]
        context = context[atom_mask]
        loss = self.loss_fn(output, context)
        preds = output

        self.weighted_loss[mode](loss, len(batch))
        self.metric[mode](preds.cpu(), context.cpu())

        self.log(f'{mode}_loss', self.weighted_loss[mode], on_epoch=True, on_step=False)
        self.log(f'{mode}_{self.metric_name}', self.metric[mode], on_epoch=True, on_step=False)

        return loss


@gin.configurable('pretrain', blacklist=['model', 'featurizer'])
def pretrain_rmat(*,
                  model: Optional[PretrainedModelBase] = None,
                  featurizer: Optional[PretrainedFeaturizerMixin] = None,
                  dataset_path: str,
                  root_path: str,
                  num_epochs: int,
                  gpus: List[int],
                  resume: bool = False,
                  save_checkpoints: bool = True,
                  use_neptune: bool = False,
                  custom_callbacks: Optional[List[str]] = None,
                  batch_size: int,
                  num_workers: int = 0):
    if not model:
        gin_model = GinModel()
        model = gin_model.produce_model()
        featurizer = gin_model.produce_featurizer()

    study_name = get_default_experiment_name()
    save_path = os.path.join(root_path, study_name)

    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError(f'Please clear {save_path} folder before running or pass train.resume=True')

    callbacks = get_default_callbacks() + get_custom_callbacks(custom_callbacks)
    loggers = get_default_loggers(save_path)
    if save_checkpoints:
        callbacks += [ModelCheckpoint(dirpath=save_path, save_last=True)]
    if use_neptune:
        neptune_logger = get_neptune_logger(model, experiment_name=study_name, description=save_path)
        apply_neptune_logger(neptune_logger, callbacks, loggers)

    trainer = Trainer(default_root_dir=save_path,
                      max_epochs=num_epochs,
                      callbacks=callbacks,
                      checkpoint_callback=save_checkpoints,
                      logger=loggers,
                      log_every_n_steps=1,
                      resume_from_checkpoint=resume_path if resume else None,
                      gpus=gpus)

    pl_module = PretrainTrainingModule(model,
                                       optimizer=get_optimizer(model),
                                       loss_fn=get_loss_fn(),
                                       metric_cls=get_metric_cls())

    train_dataset = RMatPretriningDataset(featurizer, dataset_path, 'train')
    valid_dataset = RMatPretriningDataset(featurizer, dataset_path, 'valid')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=valid_dataset.collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=valid_dataset.collate)

    trainer.fit(pl_module,
                train_dataloader=train_loader,
                val_dataloaders=valid_loader)

    return trainer
