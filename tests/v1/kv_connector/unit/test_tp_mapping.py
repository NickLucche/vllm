# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TP mapping and transfer plan utilities.

These tests verify that TP mapping produces correct outputs
(source ranks, split handles, desc IDs).
No GPU or NIXL required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

# ======================================================================
# Test fixtures / helpers
# ======================================================================


def _compute_mapping(
    tp_rank: int = 0,
    tp_size: int = 1,
    remote_tp_size: int = 1,
    is_mla: bool = False,
    num_kv_heads: int = 8,
    group_spec_types: tuple[type, ...] = (FullAttentionSpec,),
    dcp_size: int = 1,
    remote_dcp_size: int = 1,
) -> TPMapping:
    transfer_topology = SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        is_mla=is_mla,
        total_num_kv_heads=num_kv_heads,
        dcp_size=dcp_size,
        dcp_rank=tp_rank % dcp_size,
    )
    return compute_tp_mapping(
        transfer_topology=transfer_topology,
        remote_tp_size=remote_tp_size,
        group_spec_types=group_spec_types,
        remote_dcp_size=remote_dcp_size,
    )


# ======================================================================
# TP mapping structure tests
# ======================================================================


class TestTPMappingStructure:
    def test_source_ranks_homogeneous(self):
        m = _compute_mapping(tp_size=2, tp_rank=1, remote_tp_size=2)
        assert m.all_source_ranks == (1,)

    def test_source_ranks_d_gt_p(self):
        m = _compute_mapping(tp_size=4, tp_rank=2, remote_tp_size=2)
        assert m.all_source_ranks == (1,)

    def test_source_ranks_p_gt_d(self):
        m = _compute_mapping(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert m.all_source_ranks == (0, 1)


# ======================================================================
# Split handle tests
# ======================================================================


def _make_mock_worker_for_splits(group_spec_types):
    """Build a mock NixlConnectorWorker with _group_spec_types for split tests.

    No per-region replicate flags are configured (``block_len_per_layer`` empty
    and ``num_regions == 0``), so ``_fa_desc_replicated`` takes its early-return
    path and treats every FA descriptor as SPLIT, matching the legacy behavior
    these tests assert.
    """
    worker = object.__new__(NixlConnectorWorker)
    worker._group_spec_types = group_spec_types
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=False)
    worker.block_len_per_layer = []
    worker.num_regions = 0
    worker._region_is_mla = []
    return worker


class TestBuildSrcSplitHandles:
    @pytest.mark.parametrize("remote_tp_size", [2, 4])
    def test_build_src_split_handles(self, remote_tp_size):
        tp_rank = 0
        tp_size = 1

        plan = _compute_mapping(
            tp_rank=tp_rank,
            tp_size=tp_size,
            remote_tp_size=remote_tp_size,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec,))
        src_blocks_data = np.array(
            [(0x2000 + i * 1024, 1024, 0) for i in range(8)],
            dtype=np.uint64,
        )
        num_descs = len(src_blocks_data)
        splits = list(
            worker._build_local_splits_from_plan(
                plan,
                src_blocks_data,
                num_descs,
            )
        )

        assert len(splits) == remote_tp_size
        for handle in splits:
            assert len(handle) == len(src_blocks_data)
            for _, length, _ in handle:
                assert length == 1024 // remote_tp_size


class TestMambaPlanSplitHandles:
    """Verify split handles for Mamba with FA/SSM distinction."""

    def test_fa_and_ssm_different_split_factors(self):
        """Section 0 split by num_attn_reads, section 1 by abs_tp."""
        fa_readers = (0,)
        ssm_readers = (0, 1)
        plan = TPMapping(
            source_ranks_per_group=(fa_readers, ssm_readers),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 0},
            rank_offset_factor=0,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        # 2 FA descs + 1 SSM desc
        src_blocks_data = np.array(
            [
                (1000, 200, 0),  # FA desc 0
                (2000, 200, 0),  # FA desc 1
                (3000, 400, 0),  # SSM desc 0
            ],
            dtype=np.uint64,
        )

        splits = list(worker._build_local_splits_from_plan(plan, src_blocks_data, 2))

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 (FA source, p_idx=0):
        # FA: chunk=200//1=200, slot=0 → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=0 → (3000, 200, 0)
        assert splits[0] == [(1000, 200, 0), (2000, 200, 0), (3000, 200, 0)]

        # Rank 1 (not FA source, p_idx=1):
        # FA: chunk=200//1=200, slot=0 (skip_fa) → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=1 → (3200, 200, 0)
        assert splits[1] == [(1000, 200, 0), (2000, 200, 0), (3200, 200, 0)]

    def test_hetero_block_size_splits(self):
        """With a block-size ratio, single-source FA sub-block descs pass
        through whole; SSM descs are unexpanded and split per source."""
        plan = TPMapping(
            source_ranks_per_group=((0,), (0, 1)),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 0},
            rank_offset_factor=0,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        # 2 FA blocks x ratio 2 sub-blocks + 1 SSM desc (never expanded).
        src_blocks_data = np.array(
            [
                (1000, 100, 0),
                (1100, 100, 0),
                (2000, 100, 0),
                (2100, 100, 0),
                (3000, 400, 0),
            ],
            dtype=np.uint64,
        )

        splits = list(worker._build_local_splits_from_plan(plan, src_blocks_data, 4, 2))

        assert len(splits) == 2
        fa_passthrough = [
            (1000, 100, 0),
            (1100, 100, 0),
            (2000, 100, 0),
            (2100, 100, 0),
        ]
        assert splits[0] == fa_passthrough + [(3000, 200, 0)]
        assert splits[1] == fa_passthrough + [(3200, 200, 0)]

    def test_hetero_block_size_head_sharded_asserts(self):
        """Head-sharded FA reads (multiple FA sources) are incompatible with
        a block-size mismatch and must fail loudly."""
        plan = TPMapping(
            source_ranks_per_group=((0, 1), (0, 1)),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 1},
            rank_offset_factor=0,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        src_blocks_data = np.array(
            [(1000, 100, 0), (1100, 100, 0), (3000, 400, 0)],
            dtype=np.uint64,
        )

        with pytest.raises(AssertionError, match="Head-sharded"):
            list(worker._build_local_splits_from_plan(plan, src_blocks_data, 2, 2))


# ======================================================================
# Decode context parallelism
# ======================================================================


def _owned_positions(
    dcp_size: int, dcp_rank: int, cached: int, count: int
) -> list[int]:
    """Global block positions a DCP rank owns, past its prefix-cache hit."""
    return [(cached + i) * dcp_size + dcp_rank for i in range(count)]


@pytest.mark.parametrize(
    "local_dcp,remote_dcp", [(1, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2)]
)
@pytest.mark.parametrize("cached", [0, 3])
def test_dcp_slices_deliver_each_block_exactly_once(local_dcp, remote_dcp, cached):
    """Across all its source ranks, a decode rank must receive every block of
    its own token slice exactly once -- no gaps, no duplicate transfers."""
    tp_size = max(local_dcp, 1) * 2
    remote_tp_size = max(remote_dcp, 1) * 2
    n_blocks = 24

    for tp_rank in range(tp_size):
        dcp_rank = tp_rank % local_dcp
        local_positions = _owned_positions(local_dcp, dcp_rank, cached, n_blocks)
        m = _compute_mapping(
            tp_rank=tp_rank,
            tp_size=tp_size,
            remote_tp_size=remote_tp_size,
            is_mla=True,
            num_kv_heads=1,
            dcp_size=local_dcp,
            remote_dcp_size=remote_dcp,
        )
        received: dict[int, int] = {}
        for rank in m.all_source_ranks:
            block_slice = m.block_slices[rank]
            start_local, start_remote = block_slice.starts(local_positions[0])
            remote_positions = _owned_positions(
                remote_dcp, rank % remote_dcp, 0, n_blocks
            )
            local_idx = range(start_local, n_blocks, block_slice.local_stride)
            remote_idx = range(start_remote, n_blocks, block_slice.remote_stride)
            for li, ri in zip(local_idx, remote_idx):
                # The two sides must be pointing at the same place in the sequence.
                assert local_positions[li] == remote_positions[ri]
                received[local_positions[li]] = received.get(local_positions[li], 0) + 1

        reachable = [p for p in local_positions if p < n_blocks * remote_dcp]
        assert sorted(received) == reachable
        assert set(received.values()) == {1}


def test_dcp_slices_skip_locally_cached_blocks():
    """A prefix-cache hit shifts where the local slice starts, so the transfer
    must begin further into the remote's block list."""
    m = _compute_mapping(
        tp_size=1,
        remote_tp_size=1,
        is_mla=True,
        num_kv_heads=1,
        dcp_size=1,
        remote_dcp_size=1,
    )
    block_slice = m.block_slices[0]

    assert block_slice.starts(0) == (0, 0)
    # 5 blocks already cached locally: read the remote list from index 5.
    assert block_slice.starts(5) == (0, 5)


@pytest.mark.parametrize(
    "tp_size,remote_tp_size,local_dcp,remote_dcp",
    [(4, 2, 4, 2), (4, 4, 4, 4), (4, 1, 4, 1), (4, 4, 2, 4), (2, 4, 2, 4)],
)
def test_consumer_count_matches_the_ranks_that_actually_read(
    tp_size, remote_tp_size, local_dcp, remote_dcp
):
    """The producer frees blocks after this many notifications, so an
    over-count strands them until the lease expires and an under-count frees
    them while a reader is still pulling."""
    mappings = [
        _compute_mapping(
            tp_rank=tp_rank,
            tp_size=tp_size,
            remote_tp_size=remote_tp_size,
            is_mla=True,
            num_kv_heads=1,
            dcp_size=local_dcp,
            remote_dcp_size=remote_dcp,
        )
        for tp_rank in range(tp_size)
    ]

    for remote_rank in range(remote_tp_size):
        readers = sum(remote_rank in m.all_source_ranks for m in mappings)
        if readers == 0:
            # Nobody reads this rank; it is notified directly instead.
            continue
        for m in mappings:
            if remote_rank in m.all_source_ranks:
                assert m.local_consumers == readers
