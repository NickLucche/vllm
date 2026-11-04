# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only warmup for xgrammar's token-bitmask Triton kernel.

V1 applies grammar bitmasks through ``xgr.apply_token_bitmask_inplace``, which
on CUDA launches xgrammar's own Triton kernel. vLLM never calls that kernel
directly, so this launcher only warms it: it reproduces xgrammar's launch
(``apply_token_bitmask_inplace_triton``) and fills the kernel's Triton cache,
which xgrammar's runtime launch then hits.
"""

from typing import Any

import torch
from xgrammar.kernels.apply_token_bitmask_inplace_triton import (
    apply_token_bitmask_inplace_kernel,
)

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import pad_vocab_size
from vllm.model_executor.warmup.jit_warmup import WarmupChoices, _when
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)


def _apply_token_bitmask_inplace_warmup_inputs(
    *, vllm_config: VllmConfig
) -> dict[str, Any]:
    # compute_logits returns the model dtype unless the model upcasts, and a
    # [:, :vocab_size] view of the padded LM-head logits when the vocab is
    # padded. The bitmask is sized by the scheduler's structured-output
    # backend (model or tokenizer vocab), which the worker cannot see, so
    # also warm generic and multiple-of-16 row widths. Row counts cover
    # Triton's == 1, generic and multiple-of-16 classes, up to the sampled
    # rows of a full batch with speculative tokens. Pooling models never sample.
    model_config = vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    max_num_rows = vllm_config.scheduler_config.max_num_seqs * (
        1 + vllm_config.num_speculative_tokens
    )
    upcast_logits = WarmupChoices(False, True)
    padded_vocab = WarmupChoices(False, True)
    other_bitmask_width = WarmupChoices(None, 16, 17)
    num_rows = WarmupChoices(1, 2, 16)
    has_indices = WarmupChoices(False, True)
    _when(num_rows <= max_num_rows and model_config.runner_type != "pooling")
    return dict(
        logits=TritonWarmupTensor(
            torch.float32 if upcast_logits else model_config.dtype,
            shape=(num_rows, vocab_size),
            strides=(pad_vocab_size(vocab_size) if padded_vocab else vocab_size, 1),
        ),
        bitmask=TritonWarmupTensor(
            torch.int32,
            shape=(
                num_rows,
                (vocab_size + 31) // 32
                if other_bitmask_width is None
                else other_bitmask_width,
            ),
        ),
        indices=(
            TritonWarmupTensor(torch.int32, shape=(num_rows,)) if has_indices else None
        ),
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=apply_token_bitmask_inplace_kernel,
    warmup_inputs=_apply_token_bitmask_inplace_warmup_inputs,
)
def _apply_token_bitmask_inplace(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: torch.Tensor | None,
) -> DispatchSpec:
    # xgrammar 0.2.8 apply_token_bitmask_inplace_triton, called without
    # vocab_size and with int32 device indices, as apply_grammar_bitmask does.
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    block_size = 4096
    warp_size = (
        64
        if torch.version.hip is not None
        and "gfx1" not in torch.cuda.get_device_properties(0).gcnArchName
        else 32
    )
    num_rows = len(indices) if indices is not None else logits.shape[0]
    return (num_sms,), dict(
        logits_ptr=logits,
        bitmask_ptr=bitmask,
        indices_ptr=indices,
        num_rows=num_rows,
        vocab_size=min(logits.shape[-1], bitmask.shape[-1] * 32),
        logits_strides=logits.stride()[0],
        bitmask_strides=bitmask.stride()[0],
        NUM_SMS=num_sms,
        BLOCK_SIZE=block_size,
        num_warps=block_size // warp_size // (16 // logits.element_size()),
        num_stages=3,
    )
