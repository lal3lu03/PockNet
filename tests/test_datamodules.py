from pathlib import Path

import pytest
import torch

from src.data.binding_site_datamodule import BindingSiteDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_binding_site_datamodule(batch_size: int) -> None:
    """Tests `BindingSiteDataModule` to verify that it can load data correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = BindingSiteDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32  # BindingSite uses float labels
