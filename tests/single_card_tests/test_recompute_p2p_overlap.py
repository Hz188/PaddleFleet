# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recomputing the next backward chunk inside a p2p window.

A selective-recompute span replays its forward from inputs saved during the
original forward and never reads the incoming gradient, so running it early must
be invisible to the gradients -- that is what these tests pin down. The scheduler
names the chunk it wants by ``(virtual_pp_rank, micro_id)``, so the keying is the
other thing worth testing: forward order and backward order differ under
interleaving, and taking the wrong group would silently do almost no work.
"""

import unittest

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    RecomputeStore,
)

from paddlefleet.recompute_utils import install_recompute_p2p_overlap
from paddlefleet.tensor_parallel import RecomputeWithoutOutput


class _RcConfig:
    recompute_granularity = "selective"
    pipeline_model_parallel_size = 8

    def __init__(self, enabled):
        self.p2p_overlap_recompute = enabled


class _Fake:
    """Span stand-in that records when it was recomputed."""

    def __init__(self, tag, log):
        self.tag = tag
        self._log = log

    def run_recompute_now(self):
        self._log.append(self.tag)


def _build(seed=0):
    paddle.seed(seed)
    fc1 = nn.Linear(32, 64)
    fc2 = nn.Linear(64, 32)
    x = paddle.randn([8, 32])
    x.stop_gradient = False
    return fc1, fc2, x


def _run_span(fc1, fc2, x, key=(0, 0)):
    """One forward inside a named chunk, as the scheduler would drive it."""
    RecomputeStore.begin_chunk(key)
    span = RecomputeWithoutOutput()
    hidden = span.recompute(fc1, x, preserve_rng_state=False)
    out = fc2(hidden)
    span.discard_output_and_register_recompute(out)
    RecomputeStore.end_chunk()
    return span, paddle.sum(out)


class TestRecomputeStore(unittest.TestCase):
    def setUp(self):
        RecomputeStore.clear()
        RecomputeStore.enabled = False

    tearDown = setUp

    def test_off_registers_nothing(self):
        fc1, fc2, x = _build()
        _run_span(fc1, fc2, x)
        self.assertEqual(RecomputeStore.pending((0, 0)), 0)
        print("[rc store] disabled registers nothing OK")

    def test_install_requires_selective(self):
        cfg = _RcConfig(True)
        cfg.recompute_granularity = "full"
        with self.assertRaises(ValueError):
            install_recompute_p2p_overlap(cfg)
        self.assertFalse(RecomputeStore.enabled)

        install_recompute_p2p_overlap(_RcConfig(True))
        self.assertTrue(RecomputeStore.enabled)
        install_recompute_p2p_overlap(_RcConfig(False))
        self.assertFalse(RecomputeStore.enabled)

        # no pipeline parallel means no p2p window to fill
        cfg = _RcConfig(True)
        cfg.pipeline_model_parallel_size = 1
        with self.assertRaises(ValueError):
            install_recompute_p2p_overlap(cfg)
        print("[rc store] install validates granularity and pp OK")

    def test_put_outside_a_named_chunk_is_ignored(self):
        """Nothing may enter the store that the scheduler cannot name."""
        install_recompute_p2p_overlap(_RcConfig(True))
        fc1, fc2, x = _build()
        span = RecomputeWithoutOutput()
        hidden = span.recompute(fc1, x, preserve_rng_state=False)
        out = fc2(hidden)
        span.discard_output_and_register_recompute(out)  # no begin_chunk
        self.assertEqual(RecomputeStore.groups, {})
        paddle.sum(out).backward()  # its own hook must still run it
        self.assertIsNotNone(x.grad)
        print("[rc store] put outside a named chunk ignored OK")

    def test_only_the_named_chunk_runs(self):
        """Taking the wrong group is the failure mode that measures as a no-op."""
        install_recompute_p2p_overlap(_RcConfig(True))
        log = []
        for key in ((0, 0), (1, 0), (0, 1)):
            RecomputeStore.begin_chunk(key)
            for i in range(3):
                RecomputeStore.put(_Fake((key, i), log))
            RecomputeStore.end_chunk()

        self.assertEqual(RecomputeStore.pending((1, 0)), 3)
        self.assertEqual(RecomputeStore.run((1, 0)), 3)
        self.assertEqual({t[0] for t in log}, {(1, 0)})
        self.assertEqual(RecomputeStore.pending((1, 0)), 0)
        self.assertEqual(RecomputeStore.pending((0, 1)), 3, "untouched")

        self.assertEqual(RecomputeStore.run((7, 7)), 0, "unknown key")
        self.assertEqual(RecomputeStore.run((1, 0)), 0, "already taken")
        print("[rc store] only the named chunk runs OK")

    def test_drop_keeps_pending_honest(self):
        install_recompute_p2p_overlap(_RcConfig(True))
        log = []
        RecomputeStore.begin_chunk((2, 5))
        spans = [_Fake(i, log) for i in range(2)]
        for sp in spans:
            RecomputeStore.put(sp)
        RecomputeStore.end_chunk()

        RecomputeStore.drop(spans[0])
        self.assertEqual(RecomputeStore.pending((2, 5)), 1)
        RecomputeStore.drop(spans[1])
        self.assertEqual(RecomputeStore.pending((2, 5)), 0)
        self.assertEqual(
            RecomputeStore.groups, {}, "an emptied group must not linger"
        )
        self.assertEqual(RecomputeStore.run((2, 5)), 0)
        self.assertEqual(log, [], "a dropped span must never be recomputed")
        print("[rc store] drop keeps pending honest OK")


class TestHoistedSpanMatchesInline(unittest.TestCase):
    """Hoisting must change nothing except when the recompute runs."""

    def setUp(self):
        RecomputeStore.clear()
        RecomputeStore.enabled = False

    tearDown = setUp

    def _grads(self, hoist):
        fc1, fc2, x = _build(seed=1234)
        span, loss = _run_span(fc1, fc2, x, key=(3, 9))
        if hoist:
            self.assertEqual(RecomputeStore.pending((3, 9)), 1)
            self.assertEqual(RecomputeStore.run((3, 9)), 1)
            self.assertEqual(RecomputeStore.pending((3, 9)), 0)
        loss.backward()
        return [
            np.array(t.grad.astype("float32"))
            for t in (x, fc1.weight, fc1.bias, fc2.weight, fc2.bias)
        ]

    def test_gradients_are_identical(self):
        inline = self._grads(hoist=False)
        install_recompute_p2p_overlap(_RcConfig(True))
        hoisted = self._grads(hoist=True)
        for a, b in zip(inline, hoisted):
            np.testing.assert_array_equal(a, b)
        print("[rc hoist] hoisted gradients identical to inline OK")

    def test_hook_after_hoist_is_a_noop(self):
        """The grad hook cannot be unregistered, so it will still fire."""
        install_recompute_p2p_overlap(_RcConfig(True))
        fc1, fc2, x = _build(seed=7)
        span, loss = _run_span(fc1, fc2, x)
        RecomputeStore.run((0, 0))
        self.assertIsNone(span.ctx, "hoisting consumes the span")
        span._recompute(None)  # what the hook will do; must not raise
        loss.backward()
        self.assertIsNotNone(x.grad)
        print("[rc hoist] hook firing after hoist is a no-op OK")

    def test_untaken_span_still_runs_via_hook(self):
        install_recompute_p2p_overlap(_RcConfig(True))
        fc1, fc2, x = _build(seed=99)
        span, loss = _run_span(fc1, fc2, x)
        self.assertEqual(RecomputeStore.pending((0, 0)), 1)
        loss.backward()  # scheduler never took it; the hook must
        self.assertIsNotNone(x.grad)
        self.assertEqual(
            RecomputeStore.pending((0, 0)),
            0,
            "a span that ran on its own must drop itself, or pending() lies "
            "and the scheduler opens an async window with no work to do",
        )
        print("[rc hoist] untaken span runs via hook and self-drops OK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
