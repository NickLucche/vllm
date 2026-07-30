# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping computation for NIXL KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

# ======================================================================
# Data structures
# ======================================================================


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds


def _is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


def _is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


@dataclass(frozen=True)
class TPMapping:
    """Complete local-to-remote TP mapping for one remote engine.

    Generated once per remote engine during handshake.
    """

    # Remote TP ranks that this local rank reads from, per group.
    # Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # FA head offset factor for hetero-TP (D_TP > P_TP).
    rank_offset_factor: int

    # Per source rank, which logical blocks the two sides have in common.
    block_slices: dict[int, BlockSlice] = field(default_factory=dict)

    # Local workers reading from each source rank. The producer frees a
    # request's blocks only once that many of us have notified it.
    local_consumers: int = 1


@dataclass(frozen=True)
class BlockSlice:
    """Logical blocks shared by a local/remote DCP rank pair.

    A rank with DCP size ``d`` and DCP rank ``p`` owns the global block
    positions ``{p, p + d, p + 2d, ...}``. Two ranks therefore share the
    positions solving ``x = p_local (mod d_local)`` and
    ``x = p_remote (mod d_remote)``, which by CRT is a single arithmetic
    progression of stride ``lcm(d_local, d_remote)``. Both sides reduce to
    ``ids[start::stride]``, with everything but ``start`` fixed at handshake.
    """

    base: int
    """Smallest shared global block position."""

    period: int
    """lcm(local_dcp_size, remote_dcp_size), in global positions."""

    local_dcp_size: int
    remote_dcp_size: int
    remote_dcp_rank: int

    @property
    def local_stride(self) -> int:
        return self.period // self.local_dcp_size

    @property
    def remote_stride(self) -> int:
        return self.period // self.remote_dcp_size

    def starts(self, local_first_position: int) -> tuple[int, int]:
        """First matching index on each side.

        Args:
            local_first_position: global block position of local index 0,
                i.e. past any locally prefix-cached blocks.

        Returns:
            ``(start_local, start_remote)``.
        """
        first = self.base
        if first < local_first_position:
            gap = local_first_position - first
            first += -(-gap // self.period) * self.period
        return (
            (first - local_first_position) // self.local_dcp_size,
            (first - self.remote_dcp_rank) // self.remote_dcp_size,
        )


# ======================================================================
# TP mapping computation
# ======================================================================


def _make_block_slice(
    local_dcp_size: int,
    local_dcp_rank: int,
    remote_dcp_size: int,
    remote_dcp_rank: int,
) -> BlockSlice:
    """Solve the two congruences for a pair known to overlap.

    Callers must filter with :func:`_overlapping_remote_dcp_ranks` first.
    """
    common = gcd(local_dcp_size, remote_dcp_size)
    assert (local_dcp_rank - remote_dcp_rank) % common == 0, (
        f"DCP ranks {local_dcp_rank}/{remote_dcp_rank} do not overlap"
    )
    period = local_dcp_size // common * remote_dcp_size
    base = local_dcp_rank
    while base % remote_dcp_size != remote_dcp_rank % remote_dcp_size:
        base += local_dcp_size
    return BlockSlice(
        base=base % period,
        period=period,
        local_dcp_size=local_dcp_size,
        remote_dcp_size=remote_dcp_size,
        remote_dcp_rank=remote_dcp_rank,
    )


def _overlapping_remote_dcp_ranks(
    local_dcp_size: int, local_dcp_rank: int, remote_dcp_size: int
) -> list[int]:
    """Remote DCP ranks whose token slice overlaps ours."""
    common = gcd(local_dcp_size, remote_dcp_size)
    return [p for p in range(remote_dcp_size) if (local_dcp_rank - p) % common == 0]


def _source_head_shards(
    *,
    tp_rank: int,
    tp_size: int,
    remote_tp_size: int,
    total_num_kv_heads: int,
    is_mla: bool,
    has_ssm: bool,
) -> tuple[list[int], list[int]]:
    """Remote head shards this local head shard reads, for attention and SSM.

    All sizes are in head-shard space, i.e. already divided by DCP size.
    """
    if is_mla or tp_size >= remote_tp_size:
        # D (local TP) > P (remote TP): multiple local ranks read different chunks from
        # *one* remote rank, corresponding to different kv heads.
        # For MLA, we only need one remote since cache is duplicated. When P TP=k*TP k,
        # this will spread mla ranks to read from remote k*tp_rank.
        attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P (remote TP) > D (local TP): one local rank
        # reads from multiple remote ranks.
        # GQA dedup: when K < remote_tp_size, several remote ranks
        # hold the same KV head.  np.unique keeps only the first
        # rank per unique head so we don't issue redundant reads.
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        attn_ranks = (start + np.sort(unique_idx)).tolist()

    if not has_ssm:
        return attn_ranks, []
    if tp_size < remote_tp_size:
        abs_tp = remote_tp_size // tp_size
        return attn_ranks, list(range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp))
    return attn_ranks, list(attn_ranks)


def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    remote_dcp_size: int = 1,
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.

    With DCP the TP index carries two coordinates,
    ``tp_rank = head_shard * dcp_size + dcp_rank``: only ``head_shard``
    participates in KV head sharding, while ``dcp_rank`` selects a token
    slice. The mapping is therefore computed over head shards and projected
    back onto remote TP ranks at the end -- once per remote DCP rank whose slice
    overlaps ours, which is more than one when the two DCP sizes differ. With
    ``dcp_size == 1`` on both sides every step below is an identity and the
    mapping is bit-for-bit unchanged.
    """
    local_dcp_size = transfer_topology.dcp_size
    local_dcp_rank = transfer_topology.dcp_rank
    tp_rank = transfer_topology.tp_rank // local_dcp_size
    tp_size = transfer_topology.tp_size // local_dcp_size
    remote_tp_size = remote_tp_size // remote_dcp_size
    total_num_kv_heads = transfer_topology.total_num_kv_heads
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)

    attn_ranks, ssm_ranks = _source_head_shards(
        tp_rank=tp_rank,
        tp_size=tp_size,
        remote_tp_size=remote_tp_size,
        total_num_kv_heads=total_num_kv_heads,
        is_mla=transfer_topology.is_mla,
        has_ssm=has_ssm,
    )
    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Rank offset factor (head-shard space, before projection) ---
    if transfer_topology.is_mla or tp_size <= remote_tp_size:
        # We don't index into remote for reading, no offset needed.
        rank_offset_factor = 0
    elif tp_size > total_num_kv_heads:
        local_head = tp_rank * total_num_kv_heads // tp_size
        p_start = attn_ranks[0] * total_num_kv_heads // remote_tp_size
        rank_offset_factor = local_head - p_start
    else:
        # D TP > P TP: we index into remote to read different heads depending on rank.
        rank_offset_factor = tp_rank % (tp_size // remote_tp_size)

    # --- Project head shards onto the remote TP ranks covering our slice ---
    # One entry per (head shard, overlapping remote DCP rank). The set is the
    # same for every shard, so the block slice only depends on the DCP rank.
    remote_dcp_ranks = _overlapping_remote_dcp_ranks(
        local_dcp_size, local_dcp_rank, remote_dcp_size
    )
    slice_by_dcp_rank = {
        remote_dcp_rank: _make_block_slice(
            local_dcp_size, local_dcp_rank, remote_dcp_size, remote_dcp_rank
        )
        for remote_dcp_rank in remote_dcp_ranks
    }

    def project(shards: list[int]) -> list[int]:
        return [
            shard * remote_dcp_size + remote_dcp_rank
            for shard in shards
            for remote_dcp_rank in remote_dcp_ranks
        ]

    head_to_slot: dict[int, int] = {}
    for i, shard in enumerate(attn_ranks):
        head_to_slot[shard * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        shard * remote_dcp_size + remote_dcp_rank: head_to_slot.get(
            shard * total_num_kv_heads // remote_tp_size, 0
        )
        for shard in all_ranks
        for remote_dcp_rank in remote_dcp_ranks
    }
    block_slices = {
        shard * remote_dcp_size + remote_dcp_rank: slice_by_dcp_rank[remote_dcp_rank]
        for shard in all_ranks
        for remote_dcp_rank in remote_dcp_ranks
    }

    attn_ranks = project(attn_ranks)
    ssm_ranks = project(ssm_ranks)
    all_ranks = project(all_ranks)

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # Every source rank has the same number of local readers: head shards fan
    # in by tp_size ratio, DCP ranks by local_dcp_size // gcd. Only valid for
    # ranks in all_source_ranks; ranks we skip have none.
    local_consumers = max(1, tp_size // remote_tp_size) * (
        local_dcp_size // gcd(local_dcp_size, remote_dcp_size)
    )

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
        block_slices=block_slices,
        local_consumers=local_consumers,
    )
