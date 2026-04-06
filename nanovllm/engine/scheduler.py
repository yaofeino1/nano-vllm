"""
Scheduler with Decode-first mixed batching and Chunked Prefill.

Scheduling policy (per step):
1. Decode-first: all RUNNING sequences that need a decode step are collected
   first, up to max_num_seqs.  They consume 1 token of budget each.
2. Chunked Prefill: remaining token budget is given to WAITING sequences.
   Each waiting sequence contributes at most `chunked_prefill_size` tokens.
3. If chunked_prefill_size == 0, fall back to legacy Prefill-first full-prompt
   scheduling (original nano-vllm behaviour).

Return value of schedule():
  (prefill_seqs, decode_seqs)
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config) -> None:
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.chunked_prefill_size = config.chunked_prefill_size
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        # running holds all sequences in the decode phase (prefill_done == True)
        # plus any chunked-prefill sequences mid-way through their prompt.
        self.running: deque[Sequence] = deque()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], List[Sequence]]:
        """Return (prefill_seqs, decode_seqs) for this step."""
        if self.chunked_prefill_size > 0:
            return self._schedule_chunked()
        else:
            return self._schedule_legacy()

    def postprocess(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
        token_ids: List[int],
    ) -> None:
        """Update sequences after a forward pass.

        token_ids is aligned with prefill_seqs + decode_seqs (in that order).
        - A mid-chunk prefill sequence: ignore token, move back to waiting.
        - A sequence that finished its last prefill chunk: treat token as first
          decode output; sequence now moves into the decode phase.
        - A decode sequence: append token normally.
        """
        all_seqs = prefill_seqs + decode_seqs
        assert len(token_ids) == len(all_seqs), (
            f"token_ids length {len(token_ids)} != "
            f"seqs length {len(all_seqs)}"
        )
        prefill_set = set(id(s) for s in prefill_seqs)

        for seq, token_id in zip(all_seqs, token_ids):
            if id(seq) in prefill_set and not seq.prefill_done:
                # Mid-chunk: prefill not finished.
                # Return to waiting queue for the next chunk.
                # chunk_offset was already advanced by _schedule_chunked.
                self.running.remove(seq)
                seq.status = SequenceStatus.WAITING
                self.waiting.appendleft(seq)
                continue

            # Sequence produced a real output token
            seq.append_token(token_id)
            # Manage KV blocks AFTER appending: allocate new block if needed,
            # or insert completed block into the radix tree.
            self.block_manager.may_append(seq)
            finished = (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            )
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                try:
                    self.running.remove(seq)
                except ValueError:
                    pass  # already removed (e.g. preempted earlier)

    # ------------------------------------------------------------------
    # Chunked Prefill scheduler
    # ------------------------------------------------------------------

    def _schedule_chunked(self) -> Tuple[List[Sequence], List[Sequence]]:
        """Decode-first mixed scheduling with chunked prefill."""
        decode_seqs: List[Sequence] = []
        prefill_seqs: List[Sequence] = []

        # --- Step 1: Decode first ---
        # Walk all running sequences; handle preemption if no block space.
        token_budget = self.max_num_batched_tokens
        next_running: deque[Sequence] = deque()

        for seq in list(self.running):
            if seq.prefill_done:
                # Decode sequence
                if len(decode_seqs) >= self.max_num_seqs or token_budget <= 0:
                    next_running.append(seq)
                    continue
                if not self.block_manager.can_append(seq):
                    self._preempt(seq)
                    continue
                # NOTE: do NOT call may_append here; the new token hasn't been
                # appended yet.  may_append is called in postprocess after
                # append_token().
                decode_seqs.append(seq)
                next_running.append(seq)
                token_budget -= 1
            else:
                # Mid-chunk prefill sequence already admitted:
                # it will be re-scheduled in Step 2 if budget allows.
                # For now just keep it in next_running if it was moved back
                # to waiting; otherwise it shouldn't be here.
                # (postprocess moves mid-chunk seqs back to waiting, so
                # running only has prefill_done seqs + newly admitted.)
                next_running.append(seq)

        self.running = next_running

        # --- Step 2: Chunked Prefill from waiting queue ---
        admitted: List[Sequence] = []
        remaining_waiting: deque[Sequence] = deque()

        for seq in list(self.waiting):
            if len(decode_seqs) + len(admitted) >= self.max_num_seqs:
                remaining_waiting.append(seq)
                continue
            if token_budget <= 0:
                remaining_waiting.append(seq)
                continue

            # Tokens still to compute (excluding already-cached prefix)
            uncached_start = max(seq.chunk_offset, seq.num_cached_tokens)
            tokens_to_compute = seq.num_prompt_tokens - uncached_start

            chunk_tokens = max(1, min(tokens_to_compute, self.chunked_prefill_size, token_budget))

            # Check physical block availability
            if not self.block_manager.can_allocate(seq):
                remaining_waiting.append(seq)
                continue

            # First chunk: allocate all blocks upfront
            if seq.chunk_offset == 0:
                self.block_manager.allocate(seq)

            seq.advance_chunk(chunk_tokens)
            token_budget -= chunk_tokens
            seq.status = SequenceStatus.RUNNING
            admitted.append(seq)

        self.waiting = remaining_waiting
        for seq in admitted:
            self.running.append(seq)
            prefill_seqs.append(seq)

        return prefill_seqs, decode_seqs

    # ------------------------------------------------------------------
    # Legacy scheduler (chunked_prefill_size == 0)
    # ------------------------------------------------------------------

    def _schedule_legacy(self) -> Tuple[List[Sequence], List[Sequence]]:
        """Original Prefill-first full-prompt scheduling."""
        prefill_seqs: List[Sequence] = []
        decode_seqs: List[Sequence] = []

        # Prefill: admit as many waiting sequences as budget allows
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            tokens_needed = len(seq) - seq.num_cached_tokens
            if (
                num_batched_tokens + tokens_needed > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += tokens_needed
            seq.status = SequenceStatus.RUNNING
            seq.chunk_start = seq.num_cached_tokens    # compute from cache end
            seq.chunk_offset = seq.num_prompt_tokens   # mark full prefill done
            self.waiting.popleft()
            self.running.append(seq)
            prefill_seqs.append(seq)

        if prefill_seqs:
            return prefill_seqs, []

        # Decode: all running sequences
        next_running: deque[Sequence] = deque()
        for seq in list(self.running):
            if num_seqs >= self.max_num_seqs:
                next_running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if next_running:
                    victim = next_running.pop()
                    self._preempt(victim)
                elif self.running:
                    victim = self.running.pop()
                    self._preempt(victim)
                else:
                    self._preempt(seq)
                    seq = None
                    break
            if seq is not None:
                # NOTE: may_append is called in postprocess after append_token().
                decode_seqs.append(seq)
                next_running.append(seq)
                num_seqs += 1

        self.running = next_running
        return [], decode_seqs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preempt(self, seq: Sequence) -> None:
        """Move seq back to waiting, releasing its KV blocks."""
        seq.status = SequenceStatus.WAITING
        seq.chunk_offset = 0   # restart from the beginning on re-admission
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
