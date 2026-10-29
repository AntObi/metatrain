import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data import (
    CollateFn,
    Dataset,
    DatasetInfo,
    DiskDataset,
    unpack_batch,
)
from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_dataset_info,
    densify_atomic_basis_target,
    get_per_atom_sample_labels,
    get_prepare_atomic_basis_targets_transform,
    prepare_atomic_basis_targets,
    sparsify_atomic_basis_target,
)
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.data.writers import DiskDatasetWriter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_layout_coupled_per_atom():
    """
    Layout TensorMap (zero-sample blocks) defining the coupled-spherical-harmonic basis
    for H (type 1) and C (type 6).
    """
    empty = Labels(
        names=["system", "atom"],
        values=torch.empty((0, 2), dtype=torch.int32),
    )
    l0_comp = [Labels(names=["o3_mu"], values=torch.tensor([[0]], dtype=torch.int32))]
    l1_comp = [
        Labels(
            names=["o3_mu"],
            values=torch.tensor([[-1], [0], [1]], dtype=torch.int32),
        )
    ]

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor(
                [[0, 1, 1], [0, 1, 6], [1, 1, 1], [1, 1, 6]], dtype=torch.int32
            ),
        ),
        blocks=[
            TensorBlock(  # l=0, H: n=[0,1] - seeds the union at positions 0,1
                values=torch.empty((0, 1, 2), dtype=torch.float64),
                samples=empty,
                components=l0_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [1]], dtype=torch.int32)
                ),
            ),
            TensorBlock(  # l=0, C: n=[2,3] - ends up at positions 2,3 in union
                values=torch.empty((0, 1, 2), dtype=torch.float64),
                samples=empty,
                components=l0_comp,
                properties=Labels(
                    names=["n"],
                    values=torch.tensor([[1], [2]], dtype=torch.int32),
                ),
            ),
            TensorBlock(  # l=1, H: n=[0] - seeds the union at position 0
                values=torch.empty((0, 3, 1), dtype=torch.float64),
                samples=empty,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0]], dtype=torch.int32)
                ),
            ),
            TensorBlock(  # l=1, C: n=[1] - disjoint; ends up at position 1 in union
                values=torch.empty((0, 3, 2), dtype=torch.float64),
                samples=empty,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [1]], dtype=torch.int32)
                ),
            ),
        ],
    )


def _make_layout_per_atom_pair():
    """
    Layout TensorMap (zero-sample blocks) defining an atomic basis spherical target
    of rank 1 that is per atom pair.
    """
    empty = Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=torch.empty((0, 6), dtype=torch.int32),
    )

    l0_comp = [Labels(names=["o3_mu"], values=torch.tensor([[0]], dtype=torch.int32))]
    l1_comp = [
        Labels(
            names=["o3_mu"],
            values=torch.tensor([[-1], [0], [1]], dtype=torch.int32),
        )
    ]

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "first_atom_type", "second_atom_type"],
            values=torch.tensor(
                [[0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 6, 1], [1, 1, 6, 1]],
                dtype=torch.int32,
            ),
        ),
        blocks=[
            TensorBlock(  # l=0, HH: n=[0,1,2,3] - seeds the union at positions 0,1,2,3
                values=torch.empty((0, 1, 3), dtype=torch.float64),
                samples=empty,
                components=l0_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [3], [2]], dtype=torch.int32)
                ),
            ),
            TensorBlock(  # l=1, HH: n=[0] - seeds the union at position 0
                values=torch.empty((0, 3, 1), dtype=torch.float64),
                samples=empty,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0]], dtype=torch.int32)
                ),
            ),
            TensorBlock(  # l=0, CH: n=[4,5,6,7] - disjoint; ends in union pos. 4-7
                values=torch.empty((0, 1, 4), dtype=torch.float64),
                samples=empty,
                components=l0_comp,
                properties=Labels(
                    names=["n"],
                    values=torch.tensor([[1], [0], [2], [3]], dtype=torch.int32),
                ),
            ),
            TensorBlock(  # l=1, CH: n=[8,9] - disjoint; ends in union pos. 8-9
                values=torch.empty((0, 3, 2), dtype=torch.float64),
                samples=empty,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[1], [0]], dtype=torch.int32)
                ),
            ),
        ],
    )


def _make_systems():
    """
    Two systems:
      - System 0: 2 atoms  [H (atom 0), C (atom 1)]
      - System 1: 3 atoms  [H (atom 0), H (atom 1), C (atom 2)]
    """
    cell = torch.zeros((3, 3), dtype=torch.float64)
    pbc = torch.zeros(3, dtype=torch.bool)
    return [
        System(
            types=torch.tensor([1, 6]),
            positions=torch.zeros((2, 3), dtype=torch.float64),
            cell=cell,
            pbc=pbc,
        ),
        System(
            types=torch.tensor([1, 1, 6]),
            positions=torch.zeros((3, 3), dtype=torch.float64),
            cell=cell,
            pbc=pbc,
        ),
    ]


def _make_sparse_tensor(system_ids=(0, 1)):
    """
    Construct an atomic basis spherical target in the coupled basis, per-atom, for the
    two systems from ``_make_systems()``.

    Samples within each block are ordered following ``_make_systems()`` (system 0's
    atoms, then system 1's atoms) – this is the order that the padding step in
    ``prepare_atomic_basis_targets`` produces, so starting from this order makes
    round-trip comparisons straightforward.

    :param system_ids: the "system" sample label value to use for system 0 and system
        1 respectively. Defaults to ``(0, 1)``; pass non-contiguous values (e.g. ``(7,
        2)``) to check that a target's own "system" ids are round-tripped rather than
        assumed to be batch-local positions.
    """
    s0, s1 = system_ids
    h_samples = Labels(
        names=["system", "atom"],
        values=torch.tensor([[s0, 0], [s1, 0], [s1, 1]], dtype=torch.int32),
    )
    c_samples = Labels(
        names=["system", "atom"],
        values=torch.tensor([[s0, 1], [s1, 2]], dtype=torch.int32),
    )
    l0_comp = [Labels(names=["o3_mu"], values=torch.tensor([[0]], dtype=torch.int32))]
    l1_comp = [
        Labels(
            names=["o3_mu"],
            values=torch.tensor([[-1], [0], [1]], dtype=torch.int32),
        )
    ]

    # Use easily distinguishable values to make assertion failures informative. H and C
    # have disjoint property indices, matching the layout above.
    h_l0 = torch.arange(6, dtype=torch.float64).reshape(3, 1, 2)
    c_l0 = torch.arange(100, 104, dtype=torch.float64).reshape(2, 1, 2)
    h_l1 = torch.arange(9, dtype=torch.float64).reshape(3, 3, 1) + 200.0
    c_l1 = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2) + 300.0

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor(
                [[0, 1, 1], [0, 1, 6], [1, 1, 1], [1, 1, 6]], dtype=torch.int32
            ),
        ),
        blocks=[
            TensorBlock(
                values=h_l0,
                samples=h_samples,
                components=l0_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [1]], dtype=torch.int32)
                ),
            ),
            TensorBlock(
                values=c_l0,
                samples=c_samples,
                components=l0_comp,
                properties=Labels(
                    names=["n"],
                    values=torch.tensor([[1], [2]], dtype=torch.int32),
                ),
            ),
            TensorBlock(
                values=h_l1,
                samples=h_samples,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0]], dtype=torch.int32)
                ),
            ),
            TensorBlock(
                values=c_l1,
                samples=c_samples,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [1]], dtype=torch.int32)
                ),
            ),
        ],
    )


def _make_sparse_tensor_atompair(layout):
    """ """
    hh_samples = Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.int32),
    )
    ch_samples = Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=torch.tensor(
            [[0, 1, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0], [1, 2, 1, 0, 0, 0]],
            dtype=torch.int32,
        ),
    )

    l0_comp = [Labels(names=["o3_mu"], values=torch.tensor([[0]], dtype=torch.int32))]
    l1_comp = [
        Labels(
            names=["o3_mu"],
            values=torch.tensor([[-1], [0], [1]], dtype=torch.int32),
        )
    ]

    n_hh = hh_samples.values.shape[0]
    n_ch = ch_samples.values.shape[0]

    # Use easily distinguishable values to make assertion failures informative. H and C
    # have disjoint property indices, matching the layout above.
    hh_l0 = torch.arange(n_hh * 3, dtype=torch.float64).reshape(n_hh, 1, 3)
    hh_l1 = torch.arange(n_hh * 3, dtype=torch.float64).reshape(n_hh, 3, 1) + 100.0
    ch_l0 = torch.arange(n_ch * 4, dtype=torch.float64).reshape(n_ch, 1, 4) + 200.0
    ch_l1 = torch.arange(n_ch * 6, dtype=torch.float64).reshape(n_ch, 3, 2) + 300.0

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "first_atom_type", "second_atom_type"],
            values=torch.tensor(
                [[0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 6, 1], [1, 1, 6, 1]],
                dtype=torch.int32,
            ),
        ),
        blocks=[
            TensorBlock(
                values=hh_l0,
                samples=hh_samples,
                components=l0_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [3], [2]], dtype=torch.int32)
                ),
            ),
            TensorBlock(
                values=hh_l1,
                samples=hh_samples,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0]], dtype=torch.int32)
                ),
            ),
            TensorBlock(
                values=ch_l0,
                samples=ch_samples,
                components=l0_comp,
                properties=Labels(
                    names=["n"],
                    values=torch.tensor([[1], [0], [2], [3]], dtype=torch.int32),
                ),
            ),
            TensorBlock(
                values=ch_l1,
                samples=ch_samples,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[1], [0]], dtype=torch.int32)
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# JIT compiling tests
# ---------------------------------------------------------------------------


def test_jit_densify_atomic_basis_target():
    torch.jit.script(densify_atomic_basis_target)


def test_jit_sparsify_atomic_basis_target():
    torch.jit.script(sparsify_atomic_basis_target)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=["atom", "atom_pair"])
def sample_kind(request):
    return request.param


@pytest.mark.parametrize(
    "dense_layout", [True, False], ids=["dense_layout", "sparse_layout"]
)
def test_densify_sparsify_round_trip_(sample_kind, dense_layout):
    """
    Densifying and then sparsifying (in atom type) an atomic basis spherical
    target should exactly recover the original TensorMap.
    """
    systems = _make_systems()

    if sample_kind == "atom":
        layout = _make_layout_coupled_per_atom()
        sparse = _make_sparse_tensor()
    else:
        layout = _make_layout_per_atom_pair()
        sparse = _make_sparse_tensor_atompair(layout)

    layout_for_densify = (
        layout if not dense_layout else densify_atomic_basis_target(layout, layout)
    )
    dense = densify_atomic_basis_target(sparse, layout_for_densify)
    recovered = sparsify_atomic_basis_target(systems, dense, layout)

    mts.allclose_raise(recovered, sparse, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dense_layout", [True, False], ids=["dense_layout", "sparse_layout"]
)
def test_densify_nan_padding_structure(dense_layout):
    """
    Tests that an intermediate densified TensorMap must has the correct NaN-padded
    structure: property slots belonging to an atom type carry the original values, while
    property slots that *don't* belong to that type are NaN.

    After densification, samples are sorted by (system, atom).  For our two systems [H,
    C] and [H, H, C] this gives:

      index 0: (sys=0, atom=0) → H
      index 1: (sys=0, atom=1) → C
      index 2: (sys=1, atom=0) → H
      index 3: (sys=1, atom=1) → H
      index 4: (sys=1, atom=2) → C

    Scalar block (l=0), union n=[0,1,2]:
      - H atoms (layout n=[0,1]): real values at positions 0,1 ; NaN at position 2
      - C atoms (layout n=[1,2]): real values at positions 1,2 ; NaN at position 0

    Covariant block (l=1), union n=[0,1]:
      - H atoms (layout n=[0]): real value at position 0 ; NaN at position 1
      - C atoms (layout n=[0,1]): real value at positions 0,1 ; no NaNs
    """
    layout = _make_layout_coupled_per_atom()
    sparse = _make_sparse_tensor()

    if dense_layout:
        layout = densify_atomic_basis_target(layout, layout)

    dense = densify_atomic_basis_target(sparse, layout)

    # Keys no longer contain atom_type after densification
    assert "atom_type" not in dense.keys.names
    assert dense.keys.names == ["o3_lambda", "o3_sigma"]

    # Sample index sets (see docstring for ordering rationale)
    h_idx = torch.tensor([0, 2, 3])
    c_idx = torch.tensor([1, 4])

    # --- scalar block (l=0): union properties n=[0,1,2,3] ---
    l0 = dense.block({"o3_lambda": 0, "o3_sigma": 1})
    assert l0.values.shape == (5, 1, 3)

    # H atoms: real at union positions 0,1 (n=0,1); NaN at positions 2 (n=2)
    h_l0_orig = torch.arange(6, dtype=torch.float64).reshape(3, 1, 2)
    torch.testing.assert_close(l0.values[h_idx][..., :2], h_l0_orig)
    assert torch.all(torch.isnan(l0.values[h_idx][..., 2:]))

    # C atoms: NaN at union position 0 (n=0); real at positions 1,2 (n=1,2)
    c_l0_orig = torch.arange(100, 104, dtype=torch.float64).reshape(2, 1, 2)
    assert torch.all(torch.isnan(l0.values[c_idx][..., 0]))
    torch.testing.assert_close(l0.values[c_idx][..., 1:], c_l0_orig)

    # --- covariant block (l=1): union properties n=[0,1] ---
    l1 = dense.block({"o3_lambda": 1, "o3_sigma": 1})
    assert l1.values.shape == (5, 3, 2)

    # H atoms: real at union position 0 (n=0); NaN at position 1 (n=1)
    h_l1_orig = torch.arange(9, dtype=torch.float64).reshape(3, 3, 1) + 200.0
    torch.testing.assert_close(l1.values[h_idx][..., :1], h_l1_orig)
    assert torch.all(torch.isnan(l1.values[h_idx][..., 1:]))

    # C atoms: real at positions 0,1 (n=0,1), no NaNs
    c_l1_orig = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2) + 300.0
    assert not torch.any(torch.isnan(l1.values[c_idx][..., :]))
    torch.testing.assert_close(l1.values[c_idx][..., :], c_l1_orig)


def test_prepare_atomic_basis_targets_nontrivial_system_ids():
    """
    ``prepare_atomic_basis_targets`` (densify + pad) must label samples with a
    target's own "system" ids, not with the position of each system within the batch.

    Uses non-contiguous, non-zero-based ids (system 0 -> 7, system 1 -> 2) so that a
    regression reintroducing an implicit reindexing to 0, ..., n-1 would be caught by
    the exact round-trip check below, rather than just a shape/crash check.
    """
    system_ids = torch.tensor([7, 2])
    layout = _make_layout_coupled_per_atom()
    sparse = _make_sparse_tensor(system_ids=tuple(system_ids.tolist()))
    systems = _make_systems()

    prepared = prepare_atomic_basis_targets(systems, system_ids, sparse, layout)

    # The padded samples must carry the target's own system ids, in the same order
    # as `systems`/`system_ids` (not sorted, not renumbered to 0, ..., n-1).
    expected_samples = get_per_atom_sample_labels(systems, system_ids)
    for block in prepared.blocks():
        assert block.samples == expected_samples

    recovered = sparsify_atomic_basis_target(systems, prepared, layout)
    mts.allclose_raise(recovered, sparse, atol=0.0, rtol=0.0)


def test_get_prepare_atomic_basis_targets_transform_batch_from_larger_dataset():
    """
    Direct transform callers may use non-adjacent source-numbered labels instead
    of going through collation. Auxiliary labels must match the target labels;
    auxiliary values need not match either (and are never used for alignment).
    A preparation/reverse round trip must recover the original tensor exactly.
    """
    system_ids = torch.tensor([1, 3])
    layout = _make_layout_coupled_per_atom()
    sparse = _make_sparse_tensor(system_ids=tuple(system_ids.tolist()))
    systems = _make_systems()

    target_info = TargetInfo(layout=layout)
    assert target_info.is_atomic_basis

    system_index_extra = TensorMap(
        keys=Labels(names=["_"], values=torch.tensor([[0]])),
        blocks=[
            TensorBlock(
                values=torch.tensor([[99.0], [88.0]], dtype=torch.float64),
                samples=Labels(names=["system"], values=system_ids.reshape(-1, 1)),
                components=[],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            )
        ],
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=[1, 6], targets={"target": target_info}
    )
    transform, reverse_transform = get_prepare_atomic_basis_targets_transform(
        dataset_info, densify_atomic_basis_dataset_info(dataset_info)
    )

    _, prepared, _ = transform(
        systems,
        {"target": sparse},
        {"mtt::aux::system_index": system_index_extra},
    )
    expected_samples = get_per_atom_sample_labels(systems, system_ids)
    for block in prepared["target"].blocks():
        assert block.samples == expected_samples

    _, recovered, _ = reverse_transform(systems, prepared, {})
    mts.allclose_raise(recovered["target"], sparse, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("as_extra", [False, True])
@pytest.mark.parametrize("on_disk", [False, True])
def test_atomic_basis_collate_repeated_occurrences(tmp_path, as_extra, on_disk):
    """Densification, augmentation and sparsification keep three aligned occurrences."""
    systems = _make_systems()
    sparse = _make_sparse_tensor()
    layout = _make_layout_coupled_per_atom()
    info = TargetInfo(layout=layout)
    pre_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={} if as_extra else {"target": info},
        extra_data={"target": info} if as_extra else {},
    )
    dense_info = TargetInfo(layout=densify_atomic_basis_target(layout, layout))
    post_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={} if as_extra else {"target": dense_info},
        extra_data={"target": dense_info} if as_extra else {},
    )
    transform, reverse = get_prepare_atomic_basis_targets_transform(pre_info, post_info)
    per_system = [
        mts.slice(sparse, "samples", Labels("system", torch.tensor([[i]])))
        for i in range(2)
    ]
    if on_disk:
        writer = DiskDatasetWriter(tmp_path / "basis.zip")
        for system, tensor in zip(systems, per_system, strict=True):
            writer.write([system], {"target": tensor})
        writer.finish()
        dataset = DiskDataset(tmp_path / "basis.zip")
    else:
        aux = [
            TensorMap(
                Labels.single(),
                [
                    TensorBlock(
                        values=torch.tensor([[float(i)]], dtype=torch.float64),
                        samples=Labels("system", torch.tensor([[i]])),
                        components=[],
                        properties=Labels.single(),
                    )
                ],
            )
            for i in range(2)
        ]
        dataset = Dataset.from_dict(
            {"system": systems, "target": per_system, "mtt::aux::system_index": aux}
        )
    records = [dataset[i] for i in [1, 0, 1]]
    target_keys = [] if as_extra else ["target"]
    _, raw_targets, raw_extra = unpack_batch(CollateFn(target_keys)(records))
    prepared_systems, prepared_targets, prepared_extra = unpack_batch(
        CollateFn(target_keys, callables=[transform])(records)
    )
    expected_samples = get_per_atom_sample_labels(prepared_systems, torch.arange(3))
    prepared = prepared_extra["target"] if as_extra else prepared_targets["target"]
    for block in prepared.blocks():
        assert block.samples == expected_samples
    assert prepared_extra["mtt::aux::system_index"].block().values[:, 0].tolist() == [
        1,
        0,
        1,
    ]
    _, recovered_targets, recovered_extra = reverse(
        prepared_systems, prepared_targets.copy(), prepared_extra.copy()
    )
    recovered = recovered_extra["target"] if as_extra else recovered_targets["target"]
    raw = raw_extra["target"] if as_extra else raw_targets["target"]
    mts.equal_raise(recovered, raw)

    # Give each occurrence a controlled transformation (including an inversion).
    matrices = [
        torch.eye(3, dtype=torch.float64),
        -torch.eye(3, dtype=torch.float64),
        torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)),
    ]
    augmenter = O3Augmenter(post_info.targets, post_info.extra_data)
    rotated_systems, rotated_targets, rotated_extra = augmenter.apply_augmentations(
        prepared_systems, prepared_targets, matrices, prepared_extra
    )
    _, rotated_targets, rotated_extra = reverse(
        rotated_systems, rotated_targets, rotated_extra
    )
    rotated = rotated_extra["target"] if as_extra else rotated_targets["target"]
    for occurrence, source in enumerate([1, 0, 1]):
        one_aug = O3Augmenter(pre_info.targets, pre_info.extra_data)
        _, one_targets, one_extra = one_aug.apply_augmentations(
            [systems[source]],
            {} if as_extra else {"target": per_system[source]},
            [matrices[occurrence]],
            {"target": per_system[source]} if as_extra else {},
        )
        expected = one_extra["target"] if as_extra else one_targets["target"]
        for key, block in rotated.items():
            selection = block.samples.column("system") == occurrence
            expected_block = expected.block(key)
            torch.testing.assert_close(block.values[selection], expected_block.values)
    # In particular, the vector channels of occurrences 0 and 2 must differ.
    vector = rotated.block({"o3_lambda": 1, "o3_sigma": 1, "atom_type": 1})
    assert not torch.equal(
        vector.values[vector.samples.column("system") == 0],
        vector.values[vector.samples.column("system") == 2],
    )
