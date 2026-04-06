import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, [], False)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: List[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_mixed_batch(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a single input_ids / positions tensor for a mixed batch.

        Prefill sequences come first (only the un-cached, un-processed tokens),
        then decode sequences (one token each).

        Returns input_ids and positions tensors on GPU.
        """
        input_ids_list: List[int] = []
        positions_list: List[int] = []
        cu_seqlens_q: List[int] = [0]
        cu_seqlens_k: List[int] = [0]
        max_seqlen_q: int = 0
        max_seqlen_k: int = 0
        slot_mapping: List[int] = []
        has_prefix_cache = False

        # --- Prefill sequences ---
        for seq in prefill_seqs:
            # seqlen_k: how many KV entries are available in cache for this seq.
            # After this chunk is written (via slot_mapping/store_kvcache),
            # the KV cache holds tokens [0, chunk_offset).
            # The cached prefix [0, num_cached_tokens) came from RadixTree;
            # the previous chunks [num_cached_tokens, chunk_start) were written
            # by earlier scheduling steps.  So the total KV length is chunk_offset.
            seqlen_k = seq.chunk_offset
            # Tokens to actually compute: chunk_start..chunk_offset
            # chunk_start was recorded by advance_chunk() before this step.
            # For cache hits, skip the cached prefix within this window.
            compute_start = max(seq.chunk_start, seq.num_cached_tokens)
            compute_end = seq.chunk_offset      # set by scheduler
            seqlen_q = compute_end - compute_start

            input_ids_list.extend(seq.token_ids[compute_start:compute_end])
            positions_list.extend(range(compute_start, compute_end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            if seq.num_cached_tokens > 0:
                has_prefix_cache = True

            # Slot mapping: physical positions for the tokens being written
            # in THIS chunk only (compute_start..compute_end).
            # Each token at absolute position p maps to block[p // block_size]
            # at offset (p % block_size).
            for pos in range(compute_start, compute_end):
                blk_idx = pos // self.block_size
                blk_offset = pos % self.block_size
                slot_mapping.append(seq.block_table[blk_idx] * self.block_size + blk_offset)

        # --- Decode sequences ---
        decode_slot_mapping: List[int] = []
        decode_context_lens: List[int] = []
        for seq in decode_seqs:
            input_ids_list.append(seq.last_token)
            positions_list.append(len(seq) - 1)
            decode_context_lens.append(len(seq))
            decode_slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            # For decode in a mixed batch, append separate decode metadata
            cu_seqlens_q.append(cu_seqlens_q[-1] + 1)
            cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))
            max_seqlen_q = max(max_seqlen_q, 1)
            max_seqlen_k = max(max_seqlen_k, len(seq))

        slot_mapping.extend(decode_slot_mapping)

        # Determine whether prefix cache is active (need block_tables for attn)
        all_seqs = prefill_seqs + decode_seqs
        block_tables = None
        if has_prefix_cache or decode_seqs:
            block_tables = self.prepare_block_tables(all_seqs)

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Context lens for decode portion: only decode seqs need this
        context_lens_t = None
        if decode_seqs:
            context_lens_t = torch.tensor(decode_context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Always use the varlen (prefill) path for mixed or pure prefill batches.
        # flash_attn_varlen_func handles q_len=1 decode entries correctly.
        # Only pure decode batches use flash_attn_with_kvcache.
        is_prefill = True  # mixed batch or pure prefill: always varlen path
        set_context(
            is_prefill,
            cu_seqlens_q_t,
            cu_seqlens_k_t,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping_t,
            context_lens_t,
            block_tables,
        )
        return input_ids, positions

    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy: pure prefill batch (no decode sequences)."""
        return self.prepare_mixed_batch(seqs, [])

    def prepare_decode(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy: pure decode batch (no prefill sequences)."""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: List[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
        use_cuda_graph: bool = True,
    ) -> List[int]:
        """Run one forward pass for a mixed batch of prefill + decode seqs.

        Returns a list of sampled token_ids with length
        len(prefill_seqs) + len(decode_seqs).
        Callers must filter which tokens are meaningful (see LLMEngine.step).
        """
        all_seqs = prefill_seqs + decode_seqs
        is_decode_only = len(prefill_seqs) == 0

        if is_decode_only:
            input_ids, positions = self.prepare_decode(decode_seqs)
        else:
            # Prefill-only or mixed batch: unified varlen path.
            # flash_attn_varlen_func handles both pure prefill and mixed
            # batches (decode seqs appear as q_len=1 varlen entries).
            input_ids, positions = self.prepare_mixed_batch(prefill_seqs, decode_seqs)

        temperatures = self.prepare_sample(all_seqs) if self.rank == 0 else None

        # CUDA Graph can only be used for pure decode batches
        use_graph = (
            use_cuda_graph
            and is_decode_only
            and not self.enforce_eager
            and input_ids.size(0) <= 512
        )
        logits = self.run_model(input_ids, positions, not use_graph)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else []
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
