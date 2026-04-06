from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        # --- Chunked Prefill support ---
        # chunk_offset: how many prompt tokens have already been processed in
        # previous prefill steps.  0 means the sequence hasn't started prefill.
        # When chunk_offset == num_prompt_tokens the sequence moves to decode.
        self.chunk_offset: int = 0
        # chunk_start: the value of chunk_offset BEFORE the current chunk was
        # advanced.  Used by model_runner to know the exact compute window:
        #   [chunk_start, chunk_offset)  (after caching: [max(chunk_start, num_cached_tokens), chunk_offset))
        self.chunk_start: int = 0
        # _radix_leaf is set by BlockManager.allocate and updated by may_append
        # Imported here lazily to avoid circular import; BlockManager sets it.
        self._radix_leaf = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    # --- Chunked Prefill helpers ---

    @property
    def prefill_done(self) -> bool:
        """True once all prompt tokens have been fed through the model."""
        return self.chunk_offset >= self.num_prompt_tokens

    def next_chunk(self, chunk_size: int) -> list[int]:
        """Return the next chunk of prompt tokens to process.

        Returns at most `chunk_size` tokens starting from chunk_offset,
        skipping any already-cached tokens at the front.
        Cached tokens (num_cached_tokens) are handled transparently:
        they don't need re-computation but must still advance chunk_offset.
        """
        # Skip cached region
        effective_start = max(self.chunk_offset, self.num_cached_tokens)
        effective_end = min(self.chunk_offset + chunk_size, self.num_prompt_tokens)
        return self.token_ids[effective_start:effective_end]

    def advance_chunk(self, chunk_size: int) -> None:
        """Advance chunk_offset by chunk_size tokens.

        Records the previous chunk_offset as chunk_start so that model_runner
        knows exactly which window [chunk_start, chunk_offset) to compute.
        """
        self.chunk_start = self.chunk_offset
        self.chunk_offset = min(self.chunk_offset + chunk_size, self.num_prompt_tokens)

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.block_table, self.chunk_offset, self.chunk_start,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.block_table, self.chunk_offset, self.chunk_start, last) = state
        if self.num_completion_tokens == 0:
            self.token_ids = last
        else:
            self.last_token = last
        self._radix_leaf = None
