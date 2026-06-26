# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor


def _make_request(request_id: str = "test-req-42") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


class TestAssignRequestId:
    def test_randomization_enabled_by_default(self):
        req = _make_request("my-req")
        InputProcessor.assign_request_id(req)

        assert req.external_req_id == "my-req"
        assert req.request_id.startswith("my-req-")
        assert len(req.request_id) > len("my-req-")

    def test_raises_if_external_req_id_already_set(self):
        req = _make_request("my-req")
        req.external_req_id = "already-set"

        with pytest.raises(ValueError, match="external_req_id"):
            InputProcessor.assign_request_id(req)

    def test_randomized_ids_are_unique(self):
        """Randomization is always on, even for identical external ids, so
        internal ids never collide (the reason we cannot rely on an external
        actor to keep request_id unique)."""
        ids = set()
        for _ in range(100):
            req = _make_request("same-id")
            InputProcessor.assign_request_id(req)
            ids.add(req.request_id)

        assert len(ids) == 100
