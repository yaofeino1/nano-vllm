"""
Block manager backed by a Radix Tree prefix cache.

Key differences from the original chain-hash implementation:
- Uses RadixTree for cross-request KV block sharing (same prefix = shared nodes).
- Physical block allocation/deallocation is separate from the tree structure:
  the tree holds block_ids; the free list holds unallocated physical blocks.
- ref_count on RadixNodes protects in-use nodes from LRU eviction.
- `allocate` walks the tree to find cached prefix blocks, then allocates
  fresh blocks only for the uncached tail.
- `deallocate` inserts newly computed blocks into the tree FIRST (so they
  stay visible), then decrements ref_count on the old leaf (making it
  potentially evictable).
- `may_append` allocates one new physical block when the last block is full,
  and inserts the completed block into the tree.
"""

from __future__ import annotations

from collections import deque
from typing import List

from nanovllm.engine.radix_tree import RadixNode, RadixTree
from nanovllm.engine.sequence import Sequence


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        # Physical block pool: free_block_ids is the free list
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # Radix tree for prefix cache
        self.tree = RadixTree(block_size)
        # Map physical block_id -> the RadixNode that currently references it.
        # Used to clean up stale references when blocks are evicted.
        self._block_to_node: dict[int, RadixNode] = {}

    # ------------------------------------------------------------------
    # Capacity queries
    # ------------------------------------------------------------------

    def _free_count(self) -> int:
        return len(self.free_block_ids)

    def _available_count(self) -> int:
        """Free blocks + evictable blocks in the tree."""
        return self._free_count() + self.tree.evictable_blocks

    def can_allocate(self, seq: Sequence) -> bool:
        """True if we can service all blocks for seq (worst case: no cache hit)."""
        return self._available_count() >= seq.num_blocks

    def can_append(self, seq: Sequence) -> bool:
        """True if we can allocate one more block for the next decode step.

        After appending the next token, len(seq) becomes len(seq)+1.
        A new block is needed when that new length starts a new block:
            (len(seq) + 1) % block_size == 1  i.e.  len(seq) % block_size == 0
        """
        needs_new_block = (len(seq) % self.block_size == 0)
        if not needs_new_block:
            return True
        return self._available_count() >= 1

    # ------------------------------------------------------------------
    # Physical block allocation helpers
    # ------------------------------------------------------------------

    def _alloc_physical_block(self) -> int:
        """Return a free physical block id, evicting from tree if necessary."""
        if not self.free_block_ids:
            self._evict_one_block()
        return self.free_block_ids.popleft()

    def _evict_one_block(self) -> None:
        """Evict one LRU block from the radix tree and return it to free list."""
        freed_ids = self.tree.evict(1)
        for bid in freed_ids:
            self._block_to_node.pop(bid, None)
            self.free_block_ids.append(bid)

    def _free_physical_block(self, block_id: int) -> None:
        self.free_block_ids.append(block_id)

    # ------------------------------------------------------------------
    # Core allocation / deallocation
    # ------------------------------------------------------------------

    def allocate(self, seq: Sequence) -> None:
        """Allocate physical blocks for seq, reusing any cached prefix.

        Sets seq.block_table (length == seq.num_blocks) and
        seq.num_cached_tokens.  Increments ref_count along the matched path.
        Only call this once per sequence (on first admission).
        """
        assert not seq.block_table, "seq already has a block_table"

        token_ids: List[int] = seq.token_ids

        # 1. Find the longest cached prefix in the radix tree
        matched_tokens, matched_block_ids, leaf_node = self.tree.match_prefix(token_ids)

        num_matched_blocks = len(matched_block_ids)
        num_cached_tokens = num_matched_blocks * self.block_size

        # 2. Protect the matched path from eviction
        self.tree.inc_ref(leaf_node)
        for bid in matched_block_ids:
            self._block_to_node[bid] = leaf_node

        # 3. Allocate fresh physical blocks for the uncached tail
        total_blocks_needed = seq.num_blocks
        new_block_ids: List[int] = []
        for _ in range(total_blocks_needed - num_matched_blocks):
            bid = self._alloc_physical_block()
            new_block_ids.append(bid)

        # 4. Build the full block table and update sequence state
        seq.block_table = matched_block_ids + new_block_ids
        seq.num_cached_tokens = num_cached_tokens
        seq._radix_leaf = leaf_node   # remember for dec_ref on deallocate

    def deallocate(self, seq: Sequence) -> None:
        """Release blocks held by seq.

        Steps (order matters for correctness):
        1. Insert newly computed full blocks into the tree (they stay cached).
        2. Increment ref on the new tree path (protects newly inserted nodes).
        3. Decrement ref on the old matched-prefix path (makes it evictable).
        4. Free physical blocks for the last partial block (not in tree).
        5. Reset sequence state.
        """
        if not seq.block_table:
            return

        token_ids: List[int] = seq.token_ids
        block_table: List[int] = seq.block_table
        num_cached_blocks = seq.num_cached_tokens // self.block_size
        old_leaf: RadixNode = seq._radix_leaf

        # 1. Insert newly computed full blocks into the tree
        new_leaf = self.tree.insert_prefix(
            token_ids=token_ids,
            block_ids=block_table,
            parent_node=old_leaf,
            start_block=num_cached_blocks,
        )

        # 2. Protect the newly inserted path BEFORE releasing old ref,
        #    so the blocks are never temporarily unprotected.
        self.tree.inc_ref(new_leaf)
        num_full_blocks = len(token_ids) // self.block_size
        for blk_idx in range(num_cached_blocks, num_full_blocks):
            bid = block_table[blk_idx]
            self._block_to_node[bid] = new_leaf

        # 3. Release the ref held for the old matched prefix path
        self.tree.dec_ref(old_leaf)

        # 4. Immediately release the new ref we just added
        #    (the blocks are now in the tree and evictable when not matched)
        self.tree.dec_ref(new_leaf)

        # 5. Free the last partial block (was never inserted into the tree)
        if len(block_table) > num_full_blocks:
            partial_bid = block_table[-1]
            self._block_to_node.pop(partial_bid, None)
            self._free_physical_block(partial_bid)

        # 6. Reset sequence state
        seq.num_cached_tokens = 0
        seq.block_table = []
        seq._radix_leaf = self.tree.root

    def may_append(self, seq: Sequence) -> None:
        """Manage block allocation during decode.

        Called after `seq.append_token()` has already been applied.

        - position_in_block == 1: first token of a new block → allocate block.
        - position_in_block == 0: just completed a block → insert into tree.
        - Otherwise: mid-block, nothing to do.
        """
        token_count = len(seq)
        position_in_block = token_count % self.block_size

        if position_in_block == 1:
            # Started a new block: allocate a physical block for it
            new_bid = self._alloc_physical_block()
            seq.block_table.append(new_bid)

        elif position_in_block == 0:
            # Completed a block: insert into the radix tree for future reuse
            completed_blk_idx = len(seq.block_table) - 1
            bid = seq.block_table[completed_blk_idx]

            old_leaf: RadixNode = seq._radix_leaf
            new_leaf = self.tree.insert_prefix(
                token_ids=seq.token_ids,
                block_ids=seq.block_table,
                parent_node=old_leaf,
                start_block=completed_blk_idx,
            )
            self._block_to_node[bid] = new_leaf

            # Transfer the ref: protect new path, release old
            self.tree.inc_ref(new_leaf)
            self.tree.dec_ref(old_leaf)
            seq._radix_leaf = new_leaf
        # else: mid-block, nothing to do
