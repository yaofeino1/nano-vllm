"""
Radix Tree based prefix cache.

Design:
- Tree nodes are aligned to block_size (256 tokens by default).
- Each node stores a list of block_ids corresponding to its token segment.
- ref_count > 0 means the node is in use (protected from eviction).
- Eviction uses LRU: leaf nodes with ref_count == 0 and smallest
  last_access_time are evicted first.
- Cross-request sharing: nodes with matching token content are reused
  across different requests without copying KV blocks.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple


class RadixNode:
    """A node in the Radix Tree.

    Each node represents a contiguous segment of tokens aligned to block_size.
    `token_ids` stores the token ids for this node's segment.
    `block_ids` stores the corresponding physical KV cache block ids.
    """

    def __init__(self) -> None:
        # token_ids and block_ids have the same length (in block units)
        self.token_ids: List[int] = []       # tokens covered by this node
        self.block_ids: List[int] = []       # physical block ids for KV cache
        self.children: Dict[int, RadixNode] = {}  # keyed by first token of child
        self.parent: Optional[RadixNode] = None
        self.ref_count: int = 0
        self.last_access_time: float = time.monotonic()

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def touch(self) -> None:
        """Update LRU timestamp."""
        self.last_access_time = time.monotonic()


class RadixTree:
    """Radix Tree for block-aligned prefix caching.

    Supports:
    - match_prefix: find the longest cached prefix for a request
    - insert_prefix: insert newly computed KV blocks into the tree
    - evict: free least-recently-used blocks to reclaim `num_blocks` physical blocks
    - inc_ref / dec_ref: protect nodes from eviction while in use
    """

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.root = RadixNode()
        self.root.ref_count = 1   # root is always alive
        # total evictable blocks (ref_count == 0, not root)
        self._evictable_blocks: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_prefix(
        self, token_ids: List[int]
    ) -> Tuple[List[int], List[int], RadixNode]:
        """Find the longest cached prefix.

        Returns:
            matched_token_ids: the token ids that were matched
            matched_block_ids: the physical block ids for the matched prefix
            last_node: the deepest node reached (used for ref-counting)
        """
        matched_tokens: List[int] = []
        matched_blocks: List[int] = []
        node = self.root
        offset = 0  # how many tokens of token_ids we've consumed

        while offset + self.block_size <= len(token_ids):
            # The key for children lookup is the first token of the next block
            first_token = token_ids[offset]
            child = node.children.get(first_token)
            if child is None:
                break

            # Verify the child's token_ids match the input
            segment = token_ids[offset: offset + len(child.token_ids)]
            if list(segment) != child.token_ids:
                # Hash collision or mismatch — stop here
                break

            matched_tokens.extend(child.token_ids)
            matched_blocks.extend(child.block_ids)
            child.touch()
            node = child
            offset += len(child.token_ids)

        return matched_tokens, matched_blocks, node

    def insert_prefix(
        self,
        token_ids: List[int],
        block_ids: List[int],
        parent_node: RadixNode,
        start_block: int,
    ) -> RadixNode:
        """Insert newly computed KV blocks into the tree.

        Args:
            token_ids: full token ids of the request
            block_ids: all physical block ids (full block table of the request)
            parent_node: the node returned by match_prefix (insertion point)
            start_block: index of the first block to insert (blocks before this
                         were already matched)

        Only fully-filled blocks (block_size tokens each) are inserted.
        The last partial block is never cached.

        Returns:
            The leaf node that was inserted (or parent_node if nothing inserted).
        """
        num_full_blocks = len(token_ids) // self.block_size
        node = parent_node

        for blk_idx in range(start_block, num_full_blocks):
            seg_start = blk_idx * self.block_size
            seg_end = seg_start + self.block_size
            seg_tokens = token_ids[seg_start:seg_end]
            first_token = seg_tokens[0]

            if first_token in node.children:
                # Already cached by another request — just walk down
                child = node.children[first_token]
                child.touch()
                node = child
                continue

            # Create a new node for this block segment
            new_node = RadixNode()
            new_node.token_ids = list(seg_tokens)
            new_node.block_ids = [block_ids[blk_idx]]
            new_node.parent = node
            node.children[first_token] = new_node
            self._evictable_blocks += 1   # new node has ref_count == 0
            node = new_node

        return node

    def inc_ref(self, node: RadixNode) -> None:
        """Increment ref_count from node up to (but not including) root."""
        cur = node
        while cur is not self.root and cur.parent is not None:
            if cur.ref_count == 0:
                self._evictable_blocks -= cur.num_blocks
            cur.ref_count += 1
            cur = cur.parent

    def dec_ref(self, node: RadixNode) -> None:
        """Decrement ref_count from node up to (but not including) root."""
        cur = node
        while cur is not self.root and cur.parent is not None:
            cur.ref_count -= 1
            assert cur.ref_count >= 0, "ref_count went negative"
            if cur.ref_count == 0:
                self._evictable_blocks += cur.num_blocks
            cur = cur.parent

    def evict(self, num_blocks_needed: int) -> List[int]:
        """Evict LRU leaf nodes until `num_blocks_needed` blocks are freed.

        Returns the list of physical block_ids that have been freed.
        Raises RuntimeError if not enough evictable blocks exist.
        """
        if num_blocks_needed > self._evictable_blocks:
            raise RuntimeError(
                f"Cannot evict {num_blocks_needed} blocks; "
                f"only {self._evictable_blocks} evictable."
            )

        freed_block_ids: List[int] = []
        freed_count = 0

        while freed_count < num_blocks_needed:
            leaf = self._find_lru_leaf()
            if leaf is None:
                raise RuntimeError("No evictable leaf found — tree inconsistency.")
            freed_block_ids.extend(leaf.block_ids)
            freed_count += leaf.num_blocks
            self._evictable_blocks -= leaf.num_blocks
            self._remove_node(leaf)

        return freed_block_ids

    @property
    def evictable_blocks(self) -> int:
        return self._evictable_blocks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_lru_leaf(self) -> Optional[RadixNode]:
        """DFS to find the evictable leaf with the smallest last_access_time."""
        best: Optional[RadixNode] = None
        stack = list(self.root.children.values())

        while stack:
            node = stack.pop()
            if node.ref_count > 0:
                # Protected — can still walk children
                stack.extend(node.children.values())
                continue
            if node.is_leaf():
                if best is None or node.last_access_time < best.last_access_time:
                    best = node
            else:
                stack.extend(node.children.values())

        return best

    def _remove_node(self, node: RadixNode) -> None:
        """Remove a leaf node from the tree."""
        assert node.is_leaf(), "Only leaf nodes can be removed."
        assert node.ref_count == 0, "Cannot remove a node with active refs."
        parent = node.parent
        assert parent is not None
        # Remove from parent's children dict
        first_token = node.token_ids[0]
        parent.children.pop(first_token, None)
