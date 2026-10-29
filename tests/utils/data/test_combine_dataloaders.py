from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    read_systems,
    read_targets,
    unpack_batch,
)


RESOURCES_PATH = Path(__file__).parents[2] / "resources"

np.random.seed(0)


def test_without_shuffling():
    """Tests combining dataloaders without shuffling."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "qm9_reduced_100.xyz",
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})
    collate_fn = CollateFn(target_info_dict)
    dataloader_qm9 = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
    # will yield 10 batches of 10

    systems = read_systems(RESOURCES_PATH / "carbon_reduced_100.xyz")[:10]

    conf = {
        "mtt::free_energy": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "carbon_reduced_100.xyz",
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    targets = {"mtt::free_energy": targets["mtt::free_energy"][:10]}
    dataset = Dataset.from_dict(
        {"system": systems, "mtt::free_energy": targets["mtt::free_energy"]}
    )
    collate_fn = CollateFn(target_info_dict)
    dataloader_alchemical = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    # will yield 5 batches of 2

    combined_dataloader = CombinedDataLoader(
        [dataloader_qm9, dataloader_alchemical], shuffle=False
    )

    assert len(combined_dataloader) == 15
    for i_batch, batch in enumerate(combined_dataloader):
        batch = unpack_batch(batch)
        if i_batch < 10:
            assert batch[1]["mtt::U0"].block().values.shape == (10, 1)
        else:
            assert batch[1]["mtt::free_energy"].block().values.shape == (2, 1)


def test_with_shuffling():
    """Tests combining dataloaders with shuffling."""
    # WARNING: this test might fail if the random seed is changed,
    # with a probability of 1/(15 5) = 1/3003

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "qm9_reduced_100.xyz",
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    qm9_reference = [tensor.block().values.item() for tensor in targets["mtt::U0"]]
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})
    collate_fn = CollateFn(target_info_dict)
    dataloader_qm9 = DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn, shuffle=True
    )
    # will yield 10 batches of 10

    systems = read_systems(RESOURCES_PATH / "carbon_reduced_100.xyz")[:10]

    conf = {
        "mtt::free_energy": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "carbon_reduced_100.xyz",
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    targets = {"mtt::free_energy": targets["mtt::free_energy"][:10]}
    alchemical_reference = [
        tensor.block().values.item() for tensor in targets["mtt::free_energy"]
    ]
    dataset = Dataset.from_dict(
        {"system": systems, "mtt::free_energy": targets["mtt::free_energy"]}
    )
    collate_fn = CollateFn(target_info_dict)
    dataloader_alchemical = DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )
    # will yield 5 batches of 2

    combined_dataloader = CombinedDataLoader(
        [dataloader_qm9, dataloader_alchemical], shuffle=True
    )

    assert len(combined_dataloader) == 15

    qm9_batch_count = 0
    alchemical_batch_count = 0
    original_ordering = ["qm9"] * 10 + ["alchemical"] * 5
    actual_ordering = []
    qm9_values = []
    alchemical_values = []

    for batch in combined_dataloader:
        batch = unpack_batch(batch)
        if "mtt::U0" in batch[1]:
            qm9_batch_count += 1
            assert batch[1]["mtt::U0"].block().values.shape == (10, 1)
            actual_ordering.append("qm9")
            block = batch[1]["mtt::U0"].block()
            assert block.samples.column("system").tolist() == list(range(10))
            qm9_values.extend(block.values[:, 0].tolist())
        else:
            alchemical_batch_count += 1
            assert batch[1]["mtt::free_energy"].block().values.shape == (2, 1)
            actual_ordering.append("alchemical")
            block = batch[1]["mtt::free_energy"].block()
            assert block.samples.column("system").tolist() == list(range(2))
            alchemical_values.extend(block.values[:, 0].tolist())

    assert qm9_batch_count == 10
    assert alchemical_batch_count == 5
    assert actual_ordering != original_ordering

    # Batch labels restart at zero; source coverage is checked through the data.
    assert sorted(qm9_values) == sorted(qm9_reference)
    assert sorted(alchemical_values) == sorted(alchemical_reference)
