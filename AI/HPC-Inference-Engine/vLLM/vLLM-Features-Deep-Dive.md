# vLLM - Tìm Hiểu Các Tính Năng Chính

## Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [vLLM V1 Architecture](#vllm-v1-architecture)
3. [Paged Attention](#1-paged-attention)
4. [Continuous Batching](#2-continuous-batching)
5. [Distributed Inference](#3-distributed-inference)
6. [Chunked Prefill Scheduling](#4-chunked-prefill-scheduling)
7. [Automatic Prefix Caching](#5-automatic-prefix-caching)
8. [Model Quantization với LLM Compressor](#6-model-quantization-với-llm-compressor)
9. [Context Parallelism](#7-context-parallelism)
10. [Disaggregated Prefill and Decode](#8-disaggregated-prefill-and-decode)
11. [Kết Luận và So Sánh](#kết-luận-và-so-sánh)

---

## Giới Thiệu

**vLLM** (Virtual Large Language Model) là một thư viện mã nguồn mở được thiết kế để tối ưu hóa việc serving và inference của các Large Language Models (LLMs). vLLM được phát triển bởi nhóm nghiên cứu tại UC Berkeley và đạt được hiệu suất cao hơn đáng kể so với các framework truyền thống như HuggingFace Transformers hay FasterTransformer.

### Những Vấn Đề vLLM Giải Quyết

1. **Memory Inefficiency**: KV cache trong attention mechanism chiếm lượng lớn GPU memory và thường bị fragmentation
2. **Low GPU Utilization**: Các framework truyền thống không tối ưu được GPU throughput
3. **Batching Inefficiency**: Static batching yêu cầu chờ toàn bộ batch hoàn thành, dẫn đến latency cao
4. **Distributed Complexity**: Việc deploy models lớn trên nhiều GPUs phức tạp
5. **Redundant Computation**: Các requests có prefix giống nhau vẫn phải tính toán lại từ đầu
6. **Long Context Bottleneck**: Context length rất lớn (100K-1M tokens) gây ra memory và latency bottleneck

### Kiến Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────────────┐
│                         vLLM Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              API Layer (OpenAI Compatible)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    LLM Engine                            │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │         Scheduler (Continuous Batching +          │  │   │
│  │  │         Chunked Prefill + Prefix Cache)           │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │            Block Manager                          │  │   │
│  │  │      (Paged Attention Memory Management)          │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Model Executor                          │   │
│  │        (Distributed: TP, PP, DP, EP Support)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              GPU Workers (CUDA Kernels)                  │   │
│  │           - Paged Attention Kernels                      │   │
│  │           - Custom Attention Operations                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## vLLM V1 Architecture

### Tổng Quan V1 vs V0

vLLM V1 (released January 2025) là một **major redesign** của core architecture, tập trung vào scheduler, memory manager, và distributed architecture. V1 không thay đổi APIs, models, kernels, mà chỉ cải thiện internal engine.

#### Các Mục Tiêu Chính của V1

1. **Simple, modular, easy-to-hack codebase**: Dễ dàng mở rộng và customize
2. **Near-zero CPU overhead**: Giảm thiểu bottleneck từ Python code
3. **Unified architecture**: Kết hợp các optimizations vào một kiến trúc thống nhất
4. **Zero-config defaults**: Tự động enable các features/optimizations

#### Performance Improvements

```
vLLM V1 vs V0 Benchmark:

Metric                    | V0         | V1         | Improvement
--------------------------|------------|------------|-------------
Throughput (tok/s)        | 1000       | 1700       | 1.7x
Prefix Cache Overhead     | 5-15%      | <1%        | Near-zero
CPU Overhead              | High       | Near-zero  | Significant
Multi-modal Processing    | Sequential | Parallel   | 2-3x faster
```

### V1 Scheduler: Unified Token-Based Design

#### Core Design: Loại Bỏ Prefill/Decode Distinction

V1 scheduler **không phân biệt** prefill và decode phases. Thay vào đó, tất cả tokens được xử lý như nhau thông qua một **dictionary mapping**:

```python
# V1 Scheduling Decision
scheduling_decision = {
    request_id_1: num_tokens_to_process,  # e.g., 512 (prefill chunk)
    request_id_2: 1,                       # e.g., 1 (decode token)
    request_id_3: 256,                     # e.g., 256 (prefill chunk)
    ...
}

# V0 Scheduling (OLD - separate prefill/decode)
# - Either process prefills OR decodes in a step
# - Cannot mix efficiently

# V1 Scheduling (NEW - unified)
# - Mix prefills and decodes in same step
# - Single token budget for all requests
```

#### Two-Queue Prioritization Model

```python
class V1Scheduler:
    """
    V1 Scheduler với two-queue model

    Queues:
    - waiting: Requests chờ prefill (FIFO hoặc priority)
    - running: Requests đang decode (prioritized)

    Key Innovation:
    - Decode requests ALWAYS prioritized (lower latency)
    - Prefill và decode có thể mix trong cùng batch
    """

    def __init__(self, config):
        self.waiting = []      # Queue for new requests
        self.running = []      # Queue for running requests
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.policy = config.scheduling_policy  # "fcfs" or "priority"

    def schedule(self) -> Dict[str, int]:
        """
        Main scheduling algorithm

        Returns:
            Dict mapping request_id → num_tokens to process
        """
        schedule_decision = {}
        token_budget = self.max_num_batched_tokens

        # Step 1: ALWAYS prioritize decode requests (running queue)
        # This ensures low inter-token latency
        for request in self.running:
            if request.is_finished():
                continue

            # Decode: always 1 token
            if token_budget >= 1:
                schedule_decision[request.id] = 1
                token_budget -= 1

        # Step 2: Add prefill requests from waiting queue
        # Fill remaining budget with prefill chunks
        for request in self.waiting:
            if token_budget <= 0:
                break

            remaining_prefill = request.get_remaining_prefill_len()

            # Chunked prefill: cap at chunk_size
            chunk_size = min(remaining_prefill, self.chunk_size, token_budget)

            if chunk_size > 0:
                schedule_decision[request.id] = chunk_size
                token_budget -= chunk_size

                # Move to running if prefill complete
                if chunk_size >= remaining_prefill:
                    self.waiting.remove(request)
                    self.running.append(request)

        return schedule_decision
```

#### Flattened Sequence Representation

V1 sử dụng **flattened sequences** thay vì padded batches:

```
Traditional Batching (Padded):
┌──────────────────────────────────────┐
│ Seq1: [tok1, tok2, tok3, PAD, PAD]  │
│ Seq2: [tok1, tok2, PAD, PAD, PAD]   │
│ Seq3: [tok1, tok2, tok3, tok4, tok5] │
└──────────────────────────────────────┘
Shape: (3, 5) - wasted computation on PAD tokens

V1 Flattened (No Padding):
┌──────────────────────────────────────────────────────────┐
│ [tok1_s1, tok2_s1, tok3_s1, tok1_s2, tok2_s2, tok1_s3...] │
│ Position indices: [0, 1, 2, 0, 1, 0, 1, 2, 3, 4]         │
│ Sequence IDs:     [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]         │
└──────────────────────────────────────────────────────────┘
Shape: (10,) - no wasted computation

Benefits:
- Zero padding overhead
- Flexible batch composition
- Efficient continuous batching
```

### V1 Block Manager: Optimized KV Cache

#### Block Structure

```python
class KVCacheBlock:
    """
    Physical block in KV cache

    Memory Layout:
    block_memory = 2 × block_size × num_kv_heads × head_size × dtype_bytes

    Example (LLaMA-2 70B, block_size=16, FP16):
    - num_kv_heads = 8 (GQA)
    - head_size = 128
    - block_memory = 2 × 16 × 8 × 128 × 2 = 64 KB per block

    For 80GB GPU with 50% for KV cache:
    - Available: 40 GB
    - Num blocks: 40 GB / 64 KB ≈ 625,000 blocks
    - Max tokens: 625,000 × 16 = 10M tokens
    """

    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.last_accessed = 0
        self.hash = None  # For prefix caching
```

#### Block Allocation với Constant-Time Operations

```python
class V1BlockManager:
    """
    V1 Block Manager với optimized data structures

    Key Improvements over V0:
    1. Doubly-linked list for O(1) block allocation/deallocation
    2. Hash-based prefix caching với near-zero overhead
    3. No swapping needed (preemption instead)
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Doubly-linked list for O(1) operations
        self.free_block_queue = DoublyLinkedList()
        for i in range(num_blocks):
            self.free_block_queue.append(i)

        # Request → blocks mapping
        self.req_to_blocks: Dict[str, List[int]] = {}

        # Hash table for prefix caching (V1 improvement)
        self.hash_to_block: Dict[int, int] = {}
        self.block_to_hash: Dict[int, int] = {}

    def allocate_slots(self, request_id: str, num_tokens: int) -> List[int]:
        """
        Allocate KV cache slots for new tokens

        Time Complexity: O(num_blocks_needed)
        - V0: O(n) traversal
        - V1: O(1) per block via linked list
        """
        num_blocks_needed = math.ceil(num_tokens / self.block_size)

        # Check availability
        if len(self.free_block_queue) < num_blocks_needed:
            # Trigger preemption (no swapping in V1)
            self._preempt_lowest_priority()

        # Allocate from free pool
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_block_queue.pop_front()  # O(1)
            allocated.append(block_id)

        # Update mapping
        if request_id not in self.req_to_blocks:
            self.req_to_blocks[request_id] = []
        self.req_to_blocks[request_id].extend(allocated)

        return allocated

    def free_blocks(self, request_id: str):
        """Free blocks when request completes"""
        blocks = self.req_to_blocks.pop(request_id, [])
        for block_id in blocks:
            # Check if block is used by prefix cache
            if block_id not in self.hash_to_block.values():
                self.free_block_queue.append(block_id)  # O(1)
```

### V1 Multiprocessing Architecture

#### Process Layout

```
V1 Multiprocessing Architecture:

Single GPU (TP=1):
┌─────────────────────────────────────────┐
│              Driver Process              │
│  ┌─────────────────────────────────┐    │
│  │      AsyncLLM (API Layer)        │    │
│  └─────────────────────────────────┘    │
│                   ↓ IPC                  │
│  ┌─────────────────────────────────┐    │
│  │    EngineCore (Scheduler +       │    │
│  │    Block Manager)                │    │
│  └─────────────────────────────────┘    │
│                   ↓                      │
│  ┌─────────────────────────────────┐    │
│  │         Worker Process           │    │
│  │    (GPU 0, Model Executor)       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘

Multi-GPU (TP=4):
┌─────────────────────────────────────────┐
│              Driver Process              │
│  ┌─────────────────────────────────┐    │
│  │      AsyncLLM (API Layer)        │    │
│  └─────────────────────────────────┘    │
│                   ↓ IPC                  │
│  ┌─────────────────────────────────┐    │
│  │    EngineCore (Scheduler +       │    │
│  │    Block Manager)                │    │
│  └─────────────────────────────────┘    │
│         ↓ RPC Broadcast Queue           │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
│  │Worker│ │Worker│ │Worker│ │Worker│   │
│  │ GPU0 │ │ GPU1 │ │ GPU2 │ │ GPU3 │   │
│  └──────┘ └──────┘ └──────┘ └──────┘   │
└─────────────────────────────────────────┘

Key V1 Improvement:
- Scheduler và Worker 0 trong SEPARATE processes
- Symmetric architecture (mỗi worker giống nhau)
- Only incremental state updates transmitted
```

#### Incremental State Updates

```python
class V1IncrementalState:
    """
    V1 truyền ONLY incremental changes (diffs) mỗi step

    V0 Problem:
    - Send full request state every step
    - High IPC overhead for large batches

    V1 Solution:
    - Cache state on workers
    - Only send diffs: new tokens, finished requests
    """

    def compute_diff(self, prev_state, curr_state):
        """
        Compute minimal diff between states

        Diff contains:
        - New requests (added to running)
        - Finished requests (to remove)
        - New tokens for each request (1 for decode)
        - Updated block mappings
        """
        diff = {
            'new_requests': [],
            'finished_requests': [],
            'new_tokens': {},  # request_id → [new_token_ids]
            'new_blocks': {},  # request_id → [new_block_ids]
        }

        # Find new requests
        for req_id in curr_state.requests:
            if req_id not in prev_state.requests:
                diff['new_requests'].append(curr_state.requests[req_id])

        # Find finished requests
        for req_id in prev_state.requests:
            if req_id not in curr_state.requests:
                diff['finished_requests'].append(req_id)

        # Find new tokens (for decode)
        for req_id, request in curr_state.requests.items():
            if req_id in prev_state.requests:
                prev_len = len(prev_state.requests[req_id].tokens)
                curr_len = len(request.tokens)
                if curr_len > prev_len:
                    diff['new_tokens'][req_id] = request.tokens[prev_len:]

        return diff
```

#### Persistent Batch Technique

```python
class PersistentBatch:
    """
    V1 caches input tensors và chỉ apply incremental changes

    V0: Rebuild tensors from scratch mỗi step
    V1: Maintain persistent tensors, update in-place

    Benefits:
    - Avoid Python object creation overhead
    - Use NumPy operations instead of Python loops
    - Enable CUDA Graph capture
    """

    def __init__(self, max_batch_size: int, max_seq_len: int):
        # Persistent tensors (pre-allocated)
        self.input_ids = np.zeros(max_batch_size * max_seq_len, dtype=np.int32)
        self.positions = np.zeros(max_batch_size * max_seq_len, dtype=np.int32)
        self.slot_mapping = np.zeros(max_batch_size * max_seq_len, dtype=np.int32)

        # Current state
        self.num_tokens = 0
        self.request_indices = {}  # request_id → (start_idx, end_idx)

    def update_with_diff(self, diff: dict):
        """
        Apply incremental diff to persistent batch

        Uses NumPy vectorized operations for speed
        """
        # Remove finished requests
        for req_id in diff['finished_requests']:
            start, end = self.request_indices.pop(req_id)
            # Shift remaining data
            remaining = self.num_tokens - end
            self.input_ids[start:start+remaining] = self.input_ids[end:self.num_tokens]
            self.num_tokens -= (end - start)
            # Update indices for shifted requests
            for other_id, (s, e) in self.request_indices.items():
                if s > start:
                    self.request_indices[other_id] = (s - (end - start), e - (end - start))

        # Add new tokens (NumPy vectorized)
        for req_id, new_tokens in diff['new_tokens'].items():
            start, end = self.request_indices[req_id]
            new_end = end + len(new_tokens)
            self.input_ids[end:new_end] = new_tokens
            self.request_indices[req_id] = (start, new_end)
            self.num_tokens += len(new_tokens)
```

### V1 Prefix Caching: Near-Zero Overhead

#### Hash-Based Approach (V1) vs Tree-Based (SGLang)

```
vLLM V1: Hash Table-Based Prefix Caching
─────────────────────────────────────────
Data Structure: Hash table (dict)
Key: hash(token_ids in block + prefix_hash)
Value: physical_block_id

Advantages:
✅ O(1) lookup, insert, delete
✅ Near-zero overhead when cache miss
✅ Simple implementation
✅ Works well with block-based memory

Disadvantages:
❌ Cannot find partial prefix matches
❌ Fixed granularity (block-level)

─────────────────────────────────────────

SGLang: Radix Tree-Based (RadixAttention)
─────────────────────────────────────────
Data Structure: Radix tree (prefix tree with compressed edges)
Key: Token sequence as path from root
Value: KV cache tensors at nodes

Advantages:
✅ Token-level granularity
✅ Find longest prefix match efficiently
✅ Better for variable-length common prefixes

Disadvantages:
❌ Higher memory overhead (tree structure)
❌ More complex implementation
❌ Traversal overhead for each lookup
```

#### V1 Prefix Caching Implementation

```python
class V1PrefixCaching:
    """
    V1 Hash-based prefix caching

    Key Observation:
    Each KV block can be uniquely identified by:
    1. Tokens within the block
    2. Tokens in the prefix BEFORE the block

    Hash Function:
    block_hash = hash(prefix_hash, tokens_in_block)

    This ensures:
    - Same prefix + same tokens = same hash
    - Different prefix = different hash (even with same tokens)
    """

    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.hash_to_block: Dict[int, int] = {}
        self.block_to_hash: Dict[int, int] = {}
        self.block_last_accessed: Dict[int, float] = {}

    def compute_block_hash(self, prefix_hash: int, block_tokens: List[int]) -> int:
        """
        Compute hash for a block

        Hash includes:
        - Previous blocks' hash (chain)
        - Current block's tokens
        """
        # Use tuple hashing for stability
        return hash((prefix_hash, tuple(block_tokens)))

    def lookup_prefix(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """
        Find cached blocks for token sequence

        Returns:
            (num_cached_tokens, cached_block_ids)
        """
        cached_blocks = []
        prefix_hash = 0
        block_size = self.block_manager.block_size

        for i in range(0, len(token_ids), block_size):
            block_tokens = token_ids[i:i + block_size]

            # Only check complete blocks
            if len(block_tokens) < block_size:
                break

            block_hash = self.compute_block_hash(prefix_hash, block_tokens)

            if block_hash in self.hash_to_block:
                # Cache hit!
                block_id = self.hash_to_block[block_hash]
                cached_blocks.append(block_id)
                self.block_last_accessed[block_id] = time.time()
                prefix_hash = block_hash  # Chain for next block
            else:
                # Cache miss - stop here
                break

        return len(cached_blocks) * block_size, cached_blocks

    def insert_blocks(self, token_ids: List[int], block_ids: List[int]):
        """
        Insert computed blocks into cache
        """
        prefix_hash = 0
        block_size = self.block_manager.block_size

        for i, block_id in enumerate(block_ids):
            start = i * block_size
            block_tokens = token_ids[start:start + block_size]

            if len(block_tokens) == block_size:
                block_hash = self.compute_block_hash(prefix_hash, block_tokens)
                self.hash_to_block[block_hash] = block_id
                self.block_to_hash[block_id] = block_hash
                self.block_last_accessed[block_id] = time.time()
                prefix_hash = block_hash

    def evict(self, num_blocks_needed: int):
        """
        LRU eviction policy

        V1 Optimization:
        - Constant-time eviction via sorted structure
        - Prioritize evicting blocks at end of longest prefix
        """
        # Get eviction candidates (ref_count == 0)
        candidates = [
            (block_id, self.block_last_accessed[block_id])
            for block_id, hash_val in self.block_to_hash.items()
            if self.block_manager.get_ref_count(block_id) == 0
        ]

        # Sort by last accessed (LRU)
        candidates.sort(key=lambda x: x[1])

        evicted = []
        for block_id, _ in candidates[:num_blocks_needed]:
            # Remove from cache
            block_hash = self.block_to_hash.pop(block_id)
            del self.hash_to_block[block_hash]
            del self.block_last_accessed[block_id]

            # Return to free pool
            self.block_manager.free_block(block_id)
            evicted.append(block_id)

        return evicted
```

### V1 Forward Pass Pipeline

```
V1 Forward Pass (5 Stages):
═══════════════════════════════════════════════════════════════

Stage 1: State Updates
────────────────────────
• Prune finished requests from input_batch
• Update request metadata
• Apply incremental diffs

Stage 2: Input Preparation
────────────────────────
• Copy tensors CPU → GPU
• Compute position indices
• Build slot_mapping for paged attention
• Flatten sequences into "super sequence"

Stage 3: Model Execution
────────────────────────
• Run transformer layers
• Paged attention kernels access KV via slot_mapping
• All sequences computed in single forward pass

Stage 4: Token Extraction
────────────────────────
• Extract hidden states at final position of each sequence
• Compute logits via lm_head

Stage 5: Sampling
────────────────────────
• Sample next tokens according to sampling config
• Apply temperature, top-p, top-k
• Update sequences with new tokens

═══════════════════════════════════════════════════════════════

Execution Modes:

1. Eager Mode:
   • Standard PyTorch execution
   • Flexible, easier debugging
   • Higher kernel launch overhead

2. CUDA Graph Mode:
   • Pre-captured execution graphs
   • Eliminates kernel launch overhead
   • Fixed batch shape (requires padding)
   • 10-30% latency improvement
```

### V1 Summary

```
vLLM V1 Key Improvements Summary:
═══════════════════════════════════════════════════════════════

1. Unified Scheduler
   - No prefill/decode distinction
   - Simple {request_id: num_tokens} representation
   - Decode always prioritized

2. Near-Zero Prefix Caching Overhead
   - Hash-based lookup (O(1))
   - <1% throughput impact at 0% hit rate
   - Enabled by default

3. Symmetric Multiprocessing
   - Scheduler in separate process from all workers
   - Incremental state updates only
   - Reduced IPC overhead

4. Persistent Batch
   - Pre-allocated tensors
   - NumPy vectorized updates
   - CUDA Graph compatible

5. Performance
   - 1.7x throughput vs V0
   - Consistent latency
   - Better multi-modal support

═══════════════════════════════════════════════════════════════
```

---

## 1. Paged Attention

### 1.1. Vấn Đề với KV Cache Truyền Thống

Trong transformer models, attention mechanism cần lưu trữ **Key** và **Value** tensors (KV cache) cho tất cả các tokens đã được xử lý. Cách tiếp cận truyền thống:

```python
# Traditional KV Cache Storage
kv_cache = torch.zeros(batch_size, num_layers, max_seq_len, num_heads, head_dim)
```

**Vấn Đề:**

1. **Pre-allocation**: Phải cấp phát memory cho `max_seq_len` ngay từ đầu, dù sequence thực tế có thể ngắn hơn
2. **Memory Fragmentation**: Mỗi sequence có length khác nhau, gây lãng phí memory
3. **Cannot Share**: Không thể share KV cache giữa các sequences (ví dụ beam search)
4. **Static**: Không thể resize khi sequence length thay đổi

**Ví Dụ Minh Họa:**

```
Traditional Memory Layout (max_seq_len=512):

Sequence 1 (actual length=100):
[████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 ↑ Used: 100 tokens      ↑ Wasted: 412 tokens

Sequence 2 (actual length=50):
[████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 ↑ Used: 50 tokens       ↑ Wasted: 462 tokens

Total Memory: 512 * 2 = 1024 tokens allocated
Actual Usage: 100 + 50 = 150 tokens
Efficiency: 150/1024 = 14.6%
```

### 1.2. Giải Pháp: Paged Attention

Paged Attention áp dụng ý tưởng **virtual memory paging** từ hệ điều hành vào quản lý KV cache.

#### Core Concepts

1. **Logical Blocks**: Sequence được chia thành các logical blocks có kích thước cố định (ví dụ: 16 tokens)
2. **Physical Blocks**: Memory thực tế được chia thành các physical blocks
3. **Block Table**: Mapping từ logical blocks → physical blocks (giống page table trong OS)

```
Logical View (Sequence):
[Token 0-15] [Token 16-31] [Token 32-47] [Token 48-63]
    ↓             ↓             ↓             ↓
Block Table:
Logical 0  →  Physical 5
Logical 1  →  Physical 12
Logical 2  →  Physical 3
Logical 3  →  Physical 8

Physical Memory:
[Block 0] [Block 1] [Block 2] [Block 3] ... [Block 12] ...
                      ↑ Logical 2         ↑ Logical 1
```

#### Memory Layout

```python
# Paged Attention Memory Layout
class PagedAttention:
    def __init__(self, block_size=16, num_blocks=1024):
        self.block_size = block_size  # tokens per block
        self.num_blocks = num_blocks   # total physical blocks

        # Physical KV cache: all blocks stored contiguously
        # Shape: (num_blocks, 2, num_layers, num_heads, block_size, head_dim)
        #         ↑           ↑
        #         blocks      K and V
        self.kv_cache = torch.zeros(
            num_blocks, 2, num_layers, num_heads, block_size, head_dim
        )

        # Block table for each sequence
        # block_tables[seq_id] = [phys_block_0, phys_block_1, ...]
        self.block_tables = {}

        # Free block pool
        self.free_blocks = list(range(num_blocks))
```

### 1.3. Cơ Chế Hoạt Động

#### Step 1: Block Allocation

```python
class BlockManager:
    def allocate_blocks(self, seq_id, num_tokens):
        """Allocate blocks for a sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise OutOfMemoryError("Not enough blocks")

        # Allocate physical blocks
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            phys_block = self.free_blocks.pop()
            allocated_blocks.append(phys_block)

        # Create block table mapping
        self.block_tables[seq_id] = allocated_blocks

        return allocated_blocks
```

#### Step 2: Attention Computation

```python
def paged_attention_forward(
    query,           # (batch, num_heads, head_dim)
    key_cache,       # (num_blocks, num_heads, block_size, head_dim)
    value_cache,     # (num_blocks, num_heads, block_size, head_dim)
    block_tables,    # (batch, max_num_blocks_per_seq)
    context_lens     # (batch,)
):
    """
    Compute attention with paged KV cache

    For each query:
    1. Lookup block_table to find physical blocks
    2. Load K, V from physical blocks
    3. Compute attention(Q, K, V)
    4. Return output
    """
    batch_size = query.shape[0]
    outputs = []

    for i in range(batch_size):
        # Get physical blocks for this sequence
        phys_blocks = block_tables[i, :context_lens[i]]

        # Gather K, V from physical blocks
        k = key_cache[phys_blocks]      # (num_blocks, num_heads, block_size, head_dim)
        v = value_cache[phys_blocks]    # (num_blocks, num_heads, block_size, head_dim)

        # Reshape to (num_heads, total_tokens, head_dim)
        k = k.reshape(-1, num_heads, head_dim)
        v = v.reshape(-1, num_heads, head_dim)

        # Standard attention computation
        q = query[i]  # (num_heads, head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)
        attn_weights = softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        outputs.append(output)

    return torch.stack(outputs)
```

#### Step 3: Block Sharing (for Beam Search)

```python
def share_blocks(parent_seq_id, child_seq_id):
    """
    Share blocks between sequences (e.g., beam search)
    Uses Copy-on-Write (CoW)
    """
    # Child shares parent's blocks initially
    self.block_tables[child_seq_id] = self.block_tables[parent_seq_id].copy()

    # Increment reference count
    for phys_block in self.block_tables[parent_seq_id]:
        self.ref_counts[phys_block] += 1

    # When child diverges from parent, allocate new block
    # (Copy-on-Write)
```

### 1.4. Ưu Điểm

1. **Near-Zero Waste**: Chỉ lãng phí memory ở block cuối cùng (tối đa `block_size - 1` tokens)
2. **Dynamic Growth**: Sequences có thể grow bằng cách allocate thêm blocks
3. **Memory Sharing**: Hỗ trợ sharing blocks giữa sequences (beam search, parallel sampling)
4. **Efficient Swapping**: Có thể swap blocks giữa GPU memory và CPU memory

**Ví Dụ Hiệu Quả:**

```
Paged Memory Layout (block_size=16):

Sequence 1 (length=100):
Blocks needed: ⌈100/16⌉ = 7 blocks
Memory used: 7 × 16 = 112 tokens
Waste: 112 - 100 = 12 tokens

Sequence 2 (length=50):
Blocks needed: ⌈50/16⌉ = 4 blocks
Memory used: 4 × 16 = 64 tokens
Waste: 64 - 50 = 14 tokens

Total Memory: 112 + 64 = 176 tokens allocated
Actual Usage: 100 + 50 = 150 tokens
Efficiency: 150/176 = 85.2%

Improvement: 85.2% / 14.6% = 5.8x better
```

### 1.5. Implementation Details

#### CUDA Kernel Optimization

vLLM sử dụng custom CUDA kernels để tối ưu paged attention:

```cuda
// Pseudo-code for paged attention kernel
__global__ void paged_attention_kernel(
    float* out,              // Output: (batch, num_heads, head_dim)
    float* query,            // Query: (batch, num_heads, head_dim)
    float* key_cache,        // Key cache: (num_blocks, num_heads, block_size, head_dim)
    float* value_cache,      // Value cache: (num_blocks, num_heads, block_size, head_dim)
    int* block_tables,       // Block tables: (batch, max_num_blocks)
    int* context_lens,       // Context lengths: (batch,)
    int num_heads,
    int head_dim,
    int block_size
) {
    int batch_idx = blockIdx.x;
    int head_idx = threadIdx.x;

    // Load block table for this sequence
    int* block_table = &block_tables[batch_idx * max_num_blocks];
    int context_len = context_lens[batch_idx];
    int num_blocks = (context_len + block_size - 1) / block_size;

    // Load query
    float* q = &query[batch_idx * num_heads * head_dim + head_idx * head_dim];

    // Compute attention over all blocks
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int phys_block = block_table[block_idx];

        // Load K, V from physical block
        float* k_block = &key_cache[phys_block * num_heads * block_size * head_dim];
        float* v_block = &value_cache[phys_block * num_heads * block_size * head_dim];

        // Compute attention for this block
        for (int token_idx = 0; token_idx < block_size; token_idx++) {
            float attn_score = dot_product(q, &k_block[token_idx * head_dim], head_dim);
            float attn_weight = exp(attn_score);  // Simplified, need softmax
            acc += attn_weight * v_block[token_idx];
        }
    }

    // Write output
    out[batch_idx * num_heads * head_dim + head_idx * head_dim] = acc;
}
```

---

## 2. Continuous Batching

### 2.1. Vấn Đề với Static Batching

Trong static batching truyền thống:

```python
# Static Batching
batch = collect_requests(batch_size=8)
outputs = model.generate(batch)
# Wait for ALL sequences to finish before starting new batch
wait_until_all_complete(outputs)
```

**Vấn Đề:**

1. **Head-of-Line Blocking**: Nếu 1 sequence trong batch dài, tất cả sequences khác phải chờ
2. **Low GPU Utilization**: Khi sequences kết thúc sớm, GPU bị idle
3. **High Latency**: Requests mới phải chờ batch hiện tại hoàn thành

**Ví Dụ:**

```
Static Batching Timeline:

Batch 1:
Seq 1: [████████] (50 tokens)  ⟶ Done at t=50
Seq 2: [████████] (50 tokens)  ⟶ Done at t=50
Seq 3: [████████████████████] (100 tokens) ⟶ Done at t=100
Seq 4: [█████████] (55 tokens) ⟶ Done at t=55

GPU Utilization:
t=0-50:  [████] 4 sequences active (100%)
t=50-55: [██░░] 2 sequences active (50%)
t=55-100:[█░░░] 1 sequence active (25%)

Average GPU Util = (100% × 50 + 50% × 5 + 25% × 45) / 100 = 63.75%

Batch 2 can only start at t=100 (after Seq 3 finishes)
```

### 2.2. Giải Pháp: Continuous Batching

Continuous batching cho phép **động thêm/xóa sequences** khỏi batch ở mỗi iteration.

#### Core Idea

```python
# Continuous Batching
running_sequences = []
waiting_queue = Queue()

while True:
    # Remove finished sequences
    running_sequences = [seq for seq in running_sequences if not seq.is_finished()]

    # Add new sequences from queue
    while len(running_sequences) < max_batch_size and not waiting_queue.empty():
        new_seq = waiting_queue.pop()
        running_sequences.append(new_seq)

    # Generate one token for all running sequences
    outputs = model.generate_one_token(running_sequences)

    # Update sequences
    for seq, output in zip(running_sequences, outputs):
        seq.append_token(output)
```

### 2.3. Cơ Chế Hoạt Động

#### Sequence States

```python
class SequenceState(Enum):
    WAITING = "waiting"      # In queue, not yet started
    RUNNING = "running"      # Actively generating tokens
    SWAPPED = "swapped"      # Temporarily moved to CPU memory
    FINISHED = "finished"    # Generation completed
```

#### Scheduler Logic

```python
class ContinuousBatchScheduler:
    def __init__(self, max_num_seqs=256):
        self.waiting = []      # Waiting sequences
        self.running = []      # Running sequences
        self.swapped = []      # Swapped sequences
        self.max_num_seqs = max_num_seqs

    def schedule(self):
        """
        Main scheduling logic called every iteration
        Returns: sequences to process this iteration
        """
        # Step 1: Remove finished sequences
        self.running = [seq for seq in self.running if not seq.is_finished()]

        # Step 2: Check if we can swap in sequences from CPU
        while self.swapped and len(self.running) < self.max_num_seqs:
            if self.block_manager.can_allocate(self.swapped[0]):
                seq = self.swapped.pop(0)
                self.block_manager.swap_in(seq)
                self.running.append(seq)
            else:
                break

        # Step 3: Add new sequences from waiting queue
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]

            # Check if we have enough memory
            if self.block_manager.can_allocate(seq):
                self.waiting.pop(0)
                self.block_manager.allocate(seq)
                self.running.append(seq)
            else:
                # Need to free memory by swapping out or preemption
                if not self._try_swap_out():
                    break

        # Step 4: Allocate blocks for running sequences (for new tokens)
        for seq in self.running:
            if not self.block_manager.can_append_slot(seq):
                # Out of memory, need to swap out lowest priority
                self._preempt_by_swap(seq)

        return self.running

    def _try_swap_out(self):
        """Swap out lowest priority running sequence to CPU"""
        if not self.running:
            return False

        # Find lowest priority sequence
        victim = min(self.running, key=lambda s: s.priority)

        # Swap to CPU
        self.block_manager.swap_out(victim)
        self.running.remove(victim)
        self.swapped.append(victim)

        return True
```

#### Timeline Comparison

```
Continuous Batching Timeline:

t=0:   Batch = [Seq1, Seq2, Seq3, Seq4]  (4 active)
t=50:  Seq1 done ⟶ Remove Seq1, Add Seq5
       Batch = [Seq2, Seq3, Seq4, Seq5]  (4 active)
t=50:  Seq2 done ⟶ Remove Seq2, Add Seq6
       Batch = [Seq3, Seq4, Seq5, Seq6]  (4 active)
t=55:  Seq4 done ⟶ Remove Seq4, Add Seq7
       Batch = [Seq3, Seq5, Seq6, Seq7]  (4 active)
...

GPU Utilization: ~100% throughout (always 4 sequences active)

Key Benefit: No wasted GPU cycles, new requests start immediately
```

### 2.4. Memory Management Integration

Continuous batching kết hợp với Paged Attention:

```python
def generate_iteration(self):
    """One iteration of continuous batching with paged attention"""

    # 1. Schedule: decide which sequences to process
    running_seqs = self.scheduler.schedule()

    # 2. Prepare inputs
    input_tokens = [seq.get_last_token() for seq in running_seqs]
    block_tables = [self.block_manager.get_block_table(seq.seq_id)
                    for seq in running_seqs]

    # 3. Model forward pass (with paged attention)
    logits = self.model.forward(
        input_tokens=input_tokens,
        block_tables=block_tables
    )

    # 4. Sample next tokens
    next_tokens = self.sampler.sample(logits, running_seqs)

    # 5. Update sequences
    for seq, token in zip(running_seqs, next_tokens):
        seq.append_token(token)

        # Allocate new block if current block is full
        if seq.get_len() % self.block_size == 0:
            self.block_manager.allocate_block(seq.seq_id)
```

### 2.5. Ưu Điểm

1. **Higher Throughput**: GPU utilization gần 100%
2. **Lower Latency**: Requests không phải chờ batch hiện tại hoàn thành
3. **Better Resource Usage**: Không lãng phí GPU cycles
4. **Fairness**: Có thể implement priority-based scheduling

**Performance Comparison:**

```
Metric                 | Static Batching | Continuous Batching
-----------------------|-----------------|--------------------
GPU Utilization        | 60-70%          | 95-99%
Average Latency        | High            | Low
Throughput (req/sec)   | 100             | 180-250
Memory Efficiency      | Low             | High (with paging)
```

---

## 3. Distributed Inference

### 3.1. Tại Sao Cần Distributed Inference?

Các LLMs hiện đại rất lớn:

- **GPT-3**: 175B parameters → ~350GB memory (FP16)
- **LLaMA-2 70B**: 70B parameters → ~140GB memory (FP16)
- **Mixtral 8x7B**: 47B active parameters → ~94GB memory

**Vấn Đề:** Một GPU không đủ memory để load model.

**Giải Pháp:** Phân tán model trên nhiều GPUs.

### 3.2. Tensor Parallelism (TP)

#### Ý Tưởng

Chia **layers** của model theo dimension để phân tán computation.

#### Column Parallel Linear

```python
class ColumnParallelLinear:
    """
    Split weight matrix by output dimension

    Original:
        Y = X @ W       where W: (in_features, out_features)

    Tensor Parallel (TP=2):
        W = [W1 | W2]   split by columns
        GPU0: Y1 = X @ W1
        GPU1: Y2 = X @ W2
        Y = [Y1 | Y2]   concatenate outputs
    """

    def __init__(self, in_features, out_features, tp_size):
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size

        # Each GPU holds a slice of the weight
        self.out_features_per_partition = out_features // tp_size
        self.weight = Parameter(
            torch.randn(in_features, self.out_features_per_partition)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_features)

        # Local matmul
        output = torch.matmul(x, self.weight)
        # output: (batch, seq_len, out_features_per_partition)

        # All-Gather across TP group to combine results
        output = all_gather(output, dim=-1)
        # output: (batch, seq_len, out_features)

        return output
```

#### Row Parallel Linear

```python
class RowParallelLinear:
    """
    Split weight matrix by input dimension

    Original:
        Y = X @ W       where W: (in_features, out_features)

    Tensor Parallel (TP=2):
        W = [W1]        split by rows
            [W2]
        X = [X1 | X2]   split input
        GPU0: Y1 = X1 @ W1
        GPU1: Y2 = X2 @ W2
        Y = Y1 + Y2     all-reduce (sum)
    """

    def __init__(self, in_features, out_features, tp_size):
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size

        # Each GPU holds a slice of the weight
        self.in_features_per_partition = in_features // tp_size
        self.weight = Parameter(
            torch.randn(self.in_features_per_partition, out_features)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_features)
        # Input is already split by previous layer

        # Local matmul
        output = torch.matmul(x, self.weight)
        # output: (batch, seq_len, out_features)

        # All-Reduce (sum) across TP group
        output = all_reduce(output, op=ReduceOp.SUM)
        # output: (batch, seq_len, out_features)

        return output
```

#### Transformer Layer với TP

```python
class TransformerLayerTP:
    def __init__(self, hidden_size, tp_size):
        self.tp_size = tp_size

        # Attention: QKV projection (column parallel)
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, tp_size
        )

        # Attention: Output projection (row parallel)
        self.o_proj = RowParallelLinear(
            hidden_size, hidden_size, tp_size
        )

        # MLP: Up projection (column parallel)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 2 * intermediate_size, tp_size
        )

        # MLP: Down projection (row parallel)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, tp_size
        )

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)

        # === Attention ===
        # QKV projection (column parallel, no all-reduce)
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*hidden_size/tp_size)

        # Split to Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention computation (local)
        attn_output = self.attention(q, k, v)

        # Output projection (row parallel, all-reduce)
        attn_output = self.o_proj(attn_output)  # all-reduce inside

        # === MLP ===
        # Up projection (column parallel)
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # Activation
        mlp_output = F.silu(gate) * up

        # Down projection (row parallel, all-reduce)
        mlp_output = self.down_proj(mlp_output)  # all-reduce inside

        return attn_output + mlp_output
```

#### Communication Pattern

```
Tensor Parallelism (TP=4) for one Transformer Layer:

Input: x (hidden_size=1024)
│
├─ GPU0: QKV_proj [1024 → 256]
├─ GPU1: QKV_proj [1024 → 256]
├─ GPU2: QKV_proj [1024 → 256]
└─ GPU3: QKV_proj [1024 → 256]
│
├─ GPU0: Attention (256 dims)
├─ GPU1: Attention (256 dims)
├─ GPU2: Attention (256 dims)
└─ GPU3: Attention (256 dims)
│
├─ GPU0: O_proj [256 → 1024]  ╮
├─ GPU1: O_proj [256 → 1024]  ├─ All-Reduce (Sum)
├─ GPU2: O_proj [256 → 1024]  │
└─ GPU3: O_proj [256 → 1024]  ╯
│
Output: x (hidden_size=1024) ← Same on all GPUs

Communication: 2 All-Reduce operations per layer (after attention, after MLP)
```

### 3.3. Pipeline Parallelism (PP)

#### Ý Tưởng

Chia model theo **layers** và phân tán lên các GPUs khác nhau.

```
Pipeline Parallelism (PP=4, 32 layers total):

GPU0: Layers 0-7    (Input Embedding + First 8 layers)
  ↓
GPU1: Layers 8-15
  ↓
GPU2: Layers 16-23
  ↓
GPU3: Layers 24-31  (Last 8 layers + Output Head)
```

#### Naive Pipeline

```python
class NaivePipeline:
    def forward(self, x, pipeline_stages):
        """
        Sequential execution across pipeline stages
        Problem: Only 1 GPU active at a time (GPU utilization = 25% for PP=4)
        """
        # Stage 0 (GPU0)
        x = pipeline_stages[0](x)
        send_to_next_stage(x, dst=1)

        # Stage 1 (GPU1)
        x = receive_from_prev_stage(src=0)
        x = pipeline_stages[1](x)
        send_to_next_stage(x, dst=2)

        # Stage 2 (GPU2)
        x = receive_from_prev_stage(src=1)
        x = pipeline_stages[2](x)
        send_to_next_stage(x, dst=3)

        # Stage 3 (GPU3)
        x = receive_from_prev_stage(src=2)
        x = pipeline_stages[3](x)

        return x
```

**Timeline:**

```
Naive Pipeline (1 microbatch):

Time →
GPU0: [████████]░░░░░░░░░░░░░░░░░░░░░░░░
GPU1: ░░░░░░░░[████████]░░░░░░░░░░░░░░░░
GPU2: ░░░░░░░░░░░░░░░░[████████]░░░░░░░░
GPU3: ░░░░░░░░░░░░░░░░░░░░░░░░[████████]

GPU Utilization: 25%
```

#### GPipe: Microbatch Pipeline

```python
class GPipePipeline:
    def forward(self, x, num_microbatches=4):
        """
        Split batch into microbatches and pipeline execution
        Better GPU utilization
        """
        microbatches = x.chunk(num_microbatches, dim=0)

        # Forward pass with pipelining
        outputs = []
        for microbatch in microbatches:
            # Each microbatch flows through pipeline
            # While microbatch[i] is at stage[j],
            # microbatch[i-1] can be at stage[j+1]
            output = self.pipeline_forward(microbatch)
            outputs.append(output)

        return torch.cat(outputs, dim=0)
```

**Timeline:**

```
GPipe Pipeline (4 microbatches):

Time →
         [MB0][MB1][MB2][MB3]
GPU0:    [██][██][██][██]░░░░░░░░░░░░
GPU1:    ░░░░[██][██][██][██]░░░░░░░░
GPU2:    ░░░░░░░░[██][██][██][██]░░░░
GPU3:    ░░░░░░░░░░░░[██][██][██][██]

GPU Utilization: ~75% (better than naive)

Bubble: ░ (GPU idle time)
```

### 3.4. Data Parallelism (DP)

#### Ý Tưởng

Replicate model trên tất cả GPUs, mỗi GPU xử lý một phần của batch.

```
Data Parallelism (DP=4):

GPU0: Model (full copy) + Batch[0:8]
GPU1: Model (full copy) + Batch[8:16]
GPU2: Model (full copy) + Batch[16:24]
GPU3: Model (full copy) + Batch[24:32]

Each GPU computes independently
(For training: gradients are averaged across GPUs)
```

**Trong Inference:**

```python
class DataParallelInference:
    def forward(self, requests, dp_size=4):
        """
        Distribute requests across DP replicas
        Each replica processes different requests independently
        """
        # Split requests by DP rank
        local_requests = requests[self.dp_rank::dp_size]

        # Each GPU processes independently (no communication)
        outputs = self.model(local_requests)

        # Gather outputs from all GPUs
        all_outputs = all_gather(outputs)

        return all_outputs
```

**Use Case:** Tăng throughput khi có nhiều concurrent requests, model đủ nhỏ để fit vào 1 GPU.

### 3.5. Expert Parallelism (EP) - Mixture of Experts

#### Ý Tưởng

Trong Mixture of Experts (MoE) models, có nhiều "expert" networks. Mỗi token được route tới một subset of experts.

```
MoE Layer:
                  ┌─ Expert 0 (GPU0)
                  ├─ Expert 1 (GPU0)
Input tokens ─────┼─ Expert 2 (GPU1)
  (routed)        ├─ Expert 3 (GPU1)
                  ├─ Expert 4 (GPU2)
                  ├─ ...
                  └─ Expert N (GPUM)

Router: Decide which tokens → which experts
```

#### Expert Parallel Implementation

```python
class MoELayerEP:
    def __init__(self, num_experts=8, ep_size=4):
        """
        num_experts: Total number of experts
        ep_size: Expert parallelism size (number of GPUs)

        Each GPU holds: num_experts // ep_size experts
        """
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size

        # Router (replicated on all GPUs)
        self.router = Router(hidden_size, num_experts)

        # Local experts (only those assigned to this GPU)
        self.experts = nn.ModuleList([
            Expert(hidden_size)
            for _ in range(self.experts_per_rank)
        ])

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        batch, seq_len, hidden_size = x.shape

        # 1. Router decides expert assignment
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        expert_weights, expert_indices = torch.topk(router_logits, k=2, dim=-1)
        # expert_indices: (batch, seq_len, 2) - top-2 experts per token

        # 2. Flatten tokens
        tokens = x.view(-1, hidden_size)  # (batch*seq_len, hidden_size)

        # 3. Group tokens by expert (All-to-All communication)
        # Tokens going to experts on different GPUs are sent via all-to-all
        expert_tokens = self.all_to_all_scatter(tokens, expert_indices)
        # expert_tokens[i]: tokens assigned to local expert i

        # 4. Process tokens with local experts
        expert_outputs = []
        for expert_id, expert in enumerate(self.experts):
            if len(expert_tokens[expert_id]) > 0:
                output = expert(expert_tokens[expert_id])
                expert_outputs.append(output)

        # 5. All-to-All to send results back to original GPUs
        outputs = self.all_to_all_gather(expert_outputs)

        # 6. Combine outputs with router weights
        final_output = self.combine_expert_outputs(outputs, expert_weights)

        return final_output.view(batch, seq_len, hidden_size)
```

#### Communication Pattern

```
Expert Parallelism (8 experts, EP=4):

GPU0 (Experts 0-1):
  Receives: Tokens routed to Expert 0, 1
  Computes: Expert 0(tokens), Expert 1(tokens)
  Sends: Outputs back to origin GPUs

GPU1 (Experts 2-3):
  Receives: Tokens routed to Expert 2, 3
  Computes: Expert 2(tokens), Expert 3(tokens)
  Sends: Outputs back to origin GPUs

GPU2 (Experts 4-5):
  ...

GPU3 (Experts 6-7):
  ...

Communication: All-to-All (scatter + gather) per MoE layer
```

### 3.6. Hybrid Parallelism

Trong thực tế, vLLM kết hợp các strategies:

```python
# Example: TP=2, PP=2, DP=2 (total 8 GPUs)
#
# Data Parallel Group 0:
#   Pipeline Stage 0 (Layers 0-15):
#     Tensor Parallel Group: [GPU0, GPU1]
#   Pipeline Stage 1 (Layers 16-31):
#     Tensor Parallel Group: [GPU2, GPU3]
#
# Data Parallel Group 1:
#   Pipeline Stage 0 (Layers 0-15):
#     Tensor Parallel Group: [GPU4, GPU5]
#   Pipeline Stage 1 (Layers 16-31):
#     Tensor Parallel Group: [GPU6, GPU7]

class HybridParallelConfig:
    tp_size: int = 2   # Tensor parallelism
    pp_size: int = 2   # Pipeline parallelism
    dp_size: int = 2   # Data parallelism

    # Total GPUs = tp_size × pp_size × dp_size = 8
```

**Topology:**

```
8 GPUs Hybrid Parallelism (TP=2, PP=2, DP=2):

DP Group 0:
  ┌────────────────────────────────────┐
  │ PP Stage 0: Layers 0-15            │
  │   GPU0 ←─TP─→ GPU1                │
  │   (hidden_dim shard 0 & 1)         │
  └────────────────────────────────────┘
                ↓ (Pipeline)
  ┌────────────────────────────────────┐
  │ PP Stage 1: Layers 16-31           │
  │   GPU2 ←─TP─→ GPU3                │
  │   (hidden_dim shard 0 & 1)         │
  └────────────────────────────────────┘

DP Group 1:
  ┌────────────────────────────────────┐
  │ PP Stage 0: Layers 0-15            │
  │   GPU4 ←─TP─→ GPU5                │
  │   (hidden_dim shard 0 & 1)         │
  └────────────────────────────────────┘
                ↓ (Pipeline)
  ┌────────────────────────────────────┐
  │ PP Stage 1: Layers 16-31           │
  │   GPU6 ←─TP─→ GPU7                │
  │   (hidden_dim shard 0 & 1)         │
  └────────────────────────────────────┘

DP Group 0 processes Batch[0:16]
DP Group 1 processes Batch[16:32]
(Independent, no communication between DP groups)
```

### 3.7. So Sánh Các Chiến Lược

| Strategy | Memory | Communication | Use Case |
|----------|--------|---------------|----------|
| **Tensor Parallel (TP)** | Reduce per-GPU memory (model sharded) | High (All-Reduce per layer) | Large models, low latency |
| **Pipeline Parallel (PP)** | Reduce per-GPU memory (layers sharded) | Low (point-to-point) | Very large models, can tolerate latency |
| **Data Parallel (DP)** | High (full model copy) | None (inference) | Small models, high throughput |
| **Expert Parallel (EP)** | Reduce per-GPU memory (experts sharded) | High (All-to-All per MoE layer) | MoE models |

**Khi Nào Dùng Gì:**

- **TP**: Model không fit 1 GPU, cần low latency → TP=2 hoặc TP=4
- **PP**: Model rất lớn, TP không đủ → Kết hợp TP + PP
- **DP**: Model fit 1 GPU, có nhiều concurrent requests → DP=N
- **EP**: MoE models (Mixtral, Switch Transformer) → EP = số GPUs

---

## 4. Chunked Prefill Scheduling

### 4.1. Vấn Đề: Prefill vs Decode Imbalance

LLM inference có 2 phases:

1. **Prefill Phase**: Xử lý toàn bộ input prompt (nhiều tokens cùng lúc)
2. **Decode Phase**: Generate từng token một (1 token mỗi iteration)

```python
# Prefill vs Decode
def generate(prompt_tokens, max_new_tokens):
    # Prefill: process all prompt tokens at once
    kv_cache = model.forward(prompt_tokens)  # e.g., 500 tokens

    # Decode: generate one token at a time
    for _ in range(max_new_tokens):
        next_token = model.forward(last_token, kv_cache)  # 1 token
        tokens.append(next_token)
```

**Vấn Đề:**

- **Prefill**: Compute-intensive (process 100s-1000s tokens), high throughput
- **Decode**: Memory-intensive (1 token), low throughput

Nếu mix prefill và decode trong cùng batch:

```
Batch with mixed prefill/decode:

Seq 1 (Prefill): [500 tokens] ─────────┐
Seq 2 (Decode):  [1 token]             ├── Batch size = 501 tokens
Seq 3 (Decode):  [1 token]             │
...                                     │
Seq 100 (Decode):[1 token]             ┘

Problem: Prefill dominates batch, decode sequences get blocked
```

### 4.2. Giải Pháp: Chunked Prefill

#### Ý Tưởng

Chia prefill phase thành nhiều **chunks** nhỏ, interleave với decode.

```python
# Chunked Prefill
def chunked_prefill(prompt_tokens, chunk_size=512):
    """
    Instead of processing all prompt tokens at once,
    split into chunks and process iteratively
    """
    chunks = [prompt_tokens[i:i+chunk_size]
              for i in range(0, len(prompt_tokens), chunk_size)]

    for chunk in chunks:
        kv_cache.extend(model.forward(chunk))

        # Between chunks, can process decode tokens from other sequences
```

#### Ví Dụ

```
Without Chunked Prefill:

Iteration 1:
  Seq 1 (Prefill): [500 tokens] ← Processes all at once
  Seq 2-100 (Decode): [100 tokens] ← Must wait

Total: 600 tokens in batch
Imbalance: 500 vs 100 (5:1 ratio)

---

With Chunked Prefill (chunk_size=100):

Iteration 1:
  Seq 1 (Prefill Chunk 1): [100 tokens]
  Seq 2-100 (Decode):      [100 tokens]
Total: 200 tokens (balanced)

Iteration 2:
  Seq 1 (Prefill Chunk 2): [100 tokens]
  Seq 2-100 (Decode):      [100 tokens]
Total: 200 tokens

...

Iteration 5:
  Seq 1 (Prefill Chunk 5): [100 tokens]
  Seq 2-100 (Decode):      [100 tokens]
Total: 200 tokens

Iteration 6:
  Seq 1 (Decode):          [1 token]   ← Now Seq 1 is also in decode phase
  Seq 2-100 (Decode):      [100 tokens]
Total: 101 tokens

Benefit: Better balance, decode sequences not blocked
```

### 4.3. Scheduling Algorithm

```python
class ChunkedPrefillScheduler:
    def __init__(self, chunk_size=512, max_num_batched_tokens=2048):
        self.chunk_size = chunk_size
        self.max_num_batched_tokens = max_num_batched_tokens

        self.running = []   # Sequences currently being processed
        self.waiting = []   # Sequences waiting to start

    def schedule(self):
        """
        Schedule sequences for next iteration

        Goals:
        1. Balance prefill and decode
        2. Maximize GPU utilization
        3. Minimize latency for decode sequences
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # Step 1: Always prioritize running decode sequences
        # (low latency for ongoing generations)
        decode_seqs = [seq for seq in self.running if seq.is_in_decode_phase()]
        for seq in decode_seqs:
            if num_batched_tokens + 1 <= self.max_num_batched_tokens:
                scheduled_seqs.append(seq)
                num_batched_tokens += 1

        # Step 2: Add running prefill sequences (chunked)
        prefill_running = [seq for seq in self.running if seq.is_in_prefill_phase()]
        for seq in prefill_running:
            # Determine chunk size for this iteration
            remaining_prefill = seq.get_remaining_prefill_len()
            chunk = min(remaining_prefill, self.chunk_size)

            if num_batched_tokens + chunk <= self.max_num_batched_tokens:
                scheduled_seqs.append((seq, chunk))
                num_batched_tokens += chunk

        # Step 3: Add new sequences from waiting queue (prefill)
        while self.waiting and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.waiting[0]
            prompt_len = seq.get_prompt_len()

            # Compute first chunk size
            chunk = min(prompt_len, self.chunk_size)

            if num_batched_tokens + chunk <= self.max_num_batched_tokens:
                # Can fit this sequence
                self.waiting.pop(0)
                self.running.append(seq)
                scheduled_seqs.append((seq, chunk))
                num_batched_tokens += chunk
            else:
                # Cannot fit, stop
                break

        return scheduled_seqs, num_batched_tokens

    def execute_schedule(self, scheduled_seqs):
        """Execute the scheduled sequences"""
        input_tokens = []
        seq_lens = []

        for item in scheduled_seqs:
            if isinstance(item, tuple):
                # Prefill (sequence, chunk_size)
                seq, chunk_size = item
                tokens = seq.get_next_prefill_chunk(chunk_size)
                input_tokens.extend(tokens)
                seq_lens.append(len(tokens))
            else:
                # Decode (sequence only, 1 token)
                seq = item
                token = seq.get_last_token()
                input_tokens.append(token)
                seq_lens.append(1)

        # Run model forward
        logits = self.model.forward(input_tokens, seq_lens)

        # Sample and update sequences
        # ...
```

### 4.4. Batch Composition Over Time

```
Timeline với Chunked Prefill:

t=0: New request (Seq 1, prompt_len=1000) arrives

t=1: Schedule
     Seq 1 (Prefill chunk 0-512): [512 tokens]

t=2: Schedule
     Seq 1 (Prefill chunk 512-1000): [488 tokens]

t=3: Schedule
     Seq 1 (Decode): [1 token]
     Seq 2 (Prefill chunk 0-512): [512 tokens] ← New request

t=4: Schedule
     Seq 1 (Decode): [1 token]
     Seq 2 (Prefill chunk 512-800): [288 tokens]
     Seq 3 (Prefill chunk 0-512): [512 tokens] ← New request

...

Observations:
- Decode sequences always scheduled (low latency)
- Prefill progresses in chunks (doesn't block decode)
- GPU utilization stays high (batch always near max_num_batched_tokens)
```

### 4.5. Tuning Parameters

**chunk_size:**

- **Lớn** (1024-2048):
  - ✅ Higher prefill throughput
  - ❌ Higher decode latency (decode blocked longer)

- **Nhỏ** (256-512):
  - ✅ Lower decode latency
  - ❌ Lower prefill throughput (more iterations needed)

**max_num_batched_tokens:**

- **Lớn** (4096-8192):
  - ✅ Higher throughput (more parallelism)
  - ❌ Higher memory usage

- **Nhỏ** (1024-2048):
  - ✅ Lower memory usage
  - ❌ Lower throughput

**Recommended:**

```python
# For latency-sensitive workloads (chatbots)
chunk_size = 256
max_num_batched_tokens = 2048

# For throughput-oriented workloads (batch processing)
chunk_size = 1024
max_num_batched_tokens = 8192
```

### 4.6. Ưu Điểm

1. **Balanced GPU Utilization**: Prefill và decode được mix hiệu quả
2. **Lower Decode Latency**: Decode sequences không bị block bởi long prefills
3. **Higher System Throughput**: GPU utilization cao và ổn định
4. **Fairness**: New requests không phải chờ lâu

**Performance:**

```
Metric                     | Without Chunked Prefill | With Chunked Prefill
---------------------------|-------------------------|---------------------
Avg Decode Latency (ms)    | 50-200 (high variance)  | 20-30 (stable)
GPU Utilization            | 60-90% (fluctuates)     | 90-95% (stable)
Throughput (tokens/sec)    | 1000                    | 1800
P99 Time to First Token    | 5000ms                  | 800ms
```

---

## 5. Automatic Prefix Caching

### 5.1. Vấn Đề: Redundant Computation

Nhiều scenarios có **common prefixes**:

1. **System Prompts**: "You are a helpful assistant..."
2. **Few-shot Examples**: Same examples used across requests
3. **Multi-turn Conversations**: Previous conversation history
4. **RAG**: Same context documents for multiple questions

**Vấn Đề:** Mỗi request phải compute KV cache từ đầu, dù prefix giống nhau.

```python
# Without Prefix Caching
Request 1: "You are a helpful assistant. User: What is 2+2?"
  → Compute KV cache for entire sequence

Request 2: "You are a helpful assistant. User: What is 3+3?"
  → Compute KV cache for entire sequence (redundant!)

Common prefix: "You are a helpful assistant. "
Redundant computation: ~50% of tokens
```

### 5.2. Giải Pháp: Automatic Prefix Caching

#### Ý Tưởng

Tự động detect common prefixes và **reuse KV cache blocks**.

```python
# With Prefix Caching
Request 1: "You are a helpful assistant. User: What is 2+2?"
  → Compute KV cache, store in cache
  → Hash prefix: hash("You are a helpful assistant. ")

Request 2: "You are a helpful assistant. User: What is 3+3?"
  → Detect same prefix
  → Reuse KV cache blocks from Request 1
  → Only compute: "User: What is 3+3?"

Computation saved: 50%
```

### 5.3. Data Structure: Radix Trie

vLLM sử dụng **Radix Trie** để efficiently store và lookup prefixes.

#### Radix Trie Structure

```python
class RadixTreeNode:
    def __init__(self):
        self.children = {}      # token → child node
        self.block_id = None    # KV cache block ID (if leaf)
        self.ref_count = 0      # Number of sequences sharing this node
        self.last_accessed = 0  # For eviction policy

class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode()

    def insert(self, token_ids, block_ids):
        """
        Insert a sequence and its KV cache blocks

        Args:
            token_ids: List of token IDs (prefix)
            block_ids: Corresponding KV cache block IDs
        """
        node = self.root

        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                node.children[token_id] = RadixTreeNode()

            node = node.children[token_id]

            # Store block ID at appropriate position
            if (i + 1) % self.block_size == 0:
                block_idx = i // self.block_size
                node.block_id = block_ids[block_idx]

        node.ref_count += 1
        node.last_accessed = time.time()

    def lookup(self, token_ids):
        """
        Find longest matching prefix and return cached blocks

        Returns:
            matched_length: Number of tokens matched
            block_ids: List of cached KV block IDs
        """
        node = self.root
        matched_length = 0
        block_ids = []

        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                # Prefix ends here
                break

            node = node.children[token_id]
            matched_length += 1

            # Collect block ID at block boundaries
            if (i + 1) % self.block_size == 0:
                block_ids.append(node.block_id)

        return matched_length, block_ids

    def remove(self, token_ids):
        """Remove a sequence (decrease ref count, potentially evict)"""
        # Traverse and decrease ref_count
        # If ref_count reaches 0, can free blocks
        pass
```

#### Trie Visualization

```
Example Sequences:
Seq 1: [1, 2, 3, 4, 5, 6]  (block_ids: [100, 101])
Seq 2: [1, 2, 3, 4, 7, 8]  (block_ids: [100, 102])
Seq 3: [1, 2, 9, 10]       (block_ids: [103])

Radix Trie (block_size=4):

                      root
                       |
                      [1]
                       |
                      [2]
                      / \
                    [3] [9]
                    |    |
                   [4]  [10]
      (block:100)  / \   └─ (block:103)
                 [5] [7]
                  |   |
                 [6] [8]
  (block:101) ───┘   └─ (block:102)

Lookup [1, 2, 3, 4, 7, 8]:
  Path: root → 1 → 2 → 3 → 4 → 7 → 8
  Match: 6 tokens
  Cached blocks: [100] (tokens 1-4)
  Compute: tokens 5-6 (blocks [102])

Lookup [1, 2, 3, 4, 5, 6]:
  Path: root → 1 → 2 → 3 → 4 → 5 → 6
  Match: 6 tokens
  Cached blocks: [100, 101] (all tokens)
  Compute: 0 tokens (fully cached!)
```

### 5.4. Prefix Cache Integration

#### Scheduler với Prefix Caching

```python
class PrefixCachingScheduler:
    def __init__(self, block_size=16):
        self.block_manager = BlockManager(block_size)
        self.prefix_cache = RadixTree(block_size)

    def add_request(self, request):
        """
        Add new request with prefix cache lookup
        """
        token_ids = request.prompt_token_ids

        # 1. Lookup prefix cache
        matched_len, cached_block_ids = self.prefix_cache.lookup(token_ids)

        # 2. Determine what needs to be computed
        if matched_len > 0:
            # Reuse cached blocks
            request.cached_block_ids = cached_block_ids
            request.cached_len = matched_len
            request.remaining_prompt = token_ids[matched_len:]

            # Increment reference counts
            for block_id in cached_block_ids:
                self.block_manager.increment_ref(block_id)
        else:
            # No cache hit, compute from scratch
            request.cached_block_ids = []
            request.cached_len = 0
            request.remaining_prompt = token_ids

        # 3. Allocate new blocks for remaining prompt
        num_new_blocks = (len(request.remaining_prompt) + self.block_size - 1) // self.block_size
        new_block_ids = self.block_manager.allocate(num_new_blocks)

        request.block_ids = cached_block_ids + new_block_ids

        # 4. Add to running queue
        self.running.append(request)

    def process_request(self, request):
        """Process request with prefix caching"""
        if request.is_in_prefill_phase():
            # Prefill: only compute remaining_prompt
            # (cached part is skipped)
            kv_cache = request.load_cached_kv_cache()
            new_kv = self.model.forward(
                request.remaining_prompt,
                kv_cache=kv_cache
            )
            request.kv_cache.extend(new_kv)
        else:
            # Decode: normal generation
            next_token = self.model.forward(
                request.get_last_token(),
                kv_cache=request.kv_cache
            )
            request.append_token(next_token)

    def finish_request(self, request):
        """Store request in prefix cache when done"""
        # Update prefix cache with this request's tokens and blocks
        self.prefix_cache.insert(
            token_ids=request.all_token_ids,
            block_ids=request.block_ids
        )

        # Decrease reference counts (or keep if cache enabled)
        for block_id in request.block_ids:
            if request.should_cache:
                # Keep in cache for future requests
                pass
            else:
                # Free blocks
                self.block_manager.decrement_ref(block_id)
                if self.block_manager.ref_count[block_id] == 0:
                    self.block_manager.free(block_id)
```

### 5.5. Cache Eviction Policy

Khi memory đầy, cần evict cached blocks.

```python
class CacheEvictionPolicy:
    def evict(self, prefix_cache, num_blocks_needed):
        """
        Evict least valuable cached blocks

        Policies:
        1. LRU (Least Recently Used)
        2. LFU (Least Frequently Used)
        3. Size-aware (evict longest sequences first)
        """
        # Traverse trie and find candidates
        candidates = []
        for node in prefix_cache.traverse():
            if node.ref_count == 0:  # Not currently in use
                score = self.compute_eviction_score(node)
                candidates.append((score, node))

        # Sort by score (lower = evict first)
        candidates.sort(key=lambda x: x[0])

        # Evict until we have enough free blocks
        evicted_blocks = 0
        for score, node in candidates:
            if evicted_blocks >= num_blocks_needed:
                break

            # Free blocks associated with this node
            self.block_manager.free(node.block_id)
            prefix_cache.remove_node(node)
            evicted_blocks += 1

    def compute_eviction_score(self, node):
        """
        Lower score = evict first

        Example: LRU policy
        """
        return node.last_accessed
```

### 5.6. Use Cases và Performance

#### Use Case 1: System Prompts

```python
# Common system prompt
system_prompt = "You are a helpful AI assistant. You provide accurate, concise answers."
# Tokenized: ~20 tokens

# 100 requests with same system prompt
requests = [
    f"{system_prompt}\nUser: {question}"
    for question in user_questions
]

# Without prefix caching:
# Compute KV cache for 20 tokens × 100 times = 2000 token computations

# With prefix caching:
# First request: Compute 20 tokens, cache
# Remaining 99 requests: Reuse cached 20 tokens
# Total: 20 + 99×0 = 20 token computations for prefix

# Savings: (2000 - 20) / 2000 = 99% reduction in prefix computation
```

#### Use Case 2: Few-shot Prompting

```python
# Few-shot prompt with 5 examples
few_shot_prompt = """
Example 1: Input: "Hello" → Output: "Greeting"
Example 2: Input: "Bye" → Output: "Farewell"
Example 3: Input: "Thanks" → Output: "Gratitude"
Example 4: Input: "Sorry" → Output: "Apology"
Example 5: Input: "Help" → Output: "Request"

Now classify: "{user_input}"
"""
# ~150 tokens

# 1000 classification requests
requests = [few_shot_prompt.format(user_input=inp) for inp in inputs]

# With prefix caching:
# First request: Compute 150 tokens
# Remaining 999: Reuse 150 tokens
# Savings: ~99% for prefix
```

#### Use Case 3: Multi-turn Conversation

```python
# Conversation history
conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I don't have real-time data."},
    {"role": "user", "content": "OK, tell me a joke"},
]
# ~100 tokens

# Next turn
new_turn = {"role": "user", "content": "Another joke please"}

# With prefix caching:
# Previous KV cache for 100 tokens is reused
# Only compute new turn (~5 tokens)
# Savings: 100/(100+5) = 95%
```

### 5.7. Performance Impact

```
Benchmark: 1000 requests with 50% common prefix (100 tokens)

Metric                          | Without Caching | With Caching | Improvement
--------------------------------|-----------------|--------------|------------
Time to First Token (avg)       | 500ms           | 150ms        | 3.3x faster
Total Prefill Tokens Computed   | 100,000         | 55,000       | 45% reduction
GPU Compute (TFLOPs)            | 50              | 27.5         | 45% reduction
Throughput (requests/sec)       | 20              | 35           | 1.75x higher
Cache Hit Rate                  | 0%              | 90%          | -
```

**Memory Overhead:**

```
Prefix Cache Memory = Radix Trie + Cached KV Blocks

Trie Memory:
  - ~32 bytes per node (pointers, metadata)
  - For vocabulary size 50k, max depth 1000 → ~50MB

Cached KV Blocks:
  - Depends on cache size setting
  - Example: 1000 cached blocks × 256KB/block = 256MB

Total Overhead: ~300MB (negligible compared to model size)
```

---

## 6. Model Quantization với LLM Compressor

### 6.1. Vấn Đề: Model Size và Memory Bottleneck

Các LLMs hiện đại cực kỳ lớn và tốn kém về memory:

```
Model Memory Requirements (FP16/BF16):

LLaMA-2 7B:  ~14 GB
LLaMA-2 13B: ~26 GB
LLaMA-2 70B: ~140 GB
Mixtral 8x7B: ~94 GB
GPT-3 175B:  ~350 GB

Inference Memory = Model Weights + KV Cache + Activations
```

**Vấn Đề:**

1. **High Memory Usage**: Models không fit vào consumer GPUs
2. **Expensive Hardware**: Cần GPUs với VRAM lớn (A100 80GB, H100)
3. **Low Throughput**: Ít GPU memory còn lại cho batching
4. **Slow Inference**: Memory bandwidth bottleneck

**Giải Pháp:** Model Quantization - Giảm precision của weights và activations.

### 6.2. Quantization Basics

#### Ý Tưởng

Chuyển từ high-precision (FP32, FP16) sang low-precision (INT8, INT4):

```python
# Original weights (FP16)
weights_fp16 = torch.randn(4096, 4096, dtype=torch.float16)
# Memory: 4096 × 4096 × 2 bytes = 32 MB

# Quantized weights (INT8)
weights_int8 = quantize(weights_fp16, dtype=torch.int8)
# Memory: 4096 × 4096 × 1 byte = 16 MB
# Savings: 50%

# Quantized weights (INT4)
weights_int4 = quantize(weights_fp16, dtype="int4")
# Memory: 4096 × 4096 × 0.5 bytes = 8 MB
# Savings: 75%
```

#### Quantization Formula

```python
# Symmetric Quantization
def quantize_symmetric(x, n_bits=8):
    """
    Map FP values to INT range symmetrically

    x ∈ [-max_val, max_val] → q ∈ [-2^(n-1), 2^(n-1)-1]
    """
    q_max = 2 ** (n_bits - 1) - 1
    scale = x.abs().max() / q_max

    # Quantize
    x_quant = torch.round(x / scale).clamp(-q_max - 1, q_max)

    return x_quant.to(torch.int8), scale

# Dequantize for computation
def dequantize(x_quant, scale):
    return x_quant.to(torch.float16) * scale

# Example
x = torch.tensor([0.5, -0.3, 0.8, -0.1])
x_quant, scale = quantize_symmetric(x, n_bits=8)
# x_quant = [64, -38, 102, -13] (INT8)
# scale = 0.0078 (0.8 / 102)
x_recovered = dequantize(x_quant, scale)
# x_recovered ≈ [0.499, -0.296, 0.796, -0.101]
```

#### Asymmetric Quantization

```python
def quantize_asymmetric(x, n_bits=8):
    """
    Map FP values to INT range asymmetrically

    x ∈ [min_val, max_val] → q ∈ [0, 2^n - 1]
    """
    q_min = 0
    q_max = 2 ** n_bits - 1

    scale = (x.max() - x.min()) / (q_max - q_min)
    zero_point = q_min - torch.round(x.min() / scale)

    # Quantize
    x_quant = torch.round(x / scale + zero_point).clamp(q_min, q_max)

    return x_quant.to(torch.uint8), scale, zero_point

def dequantize_asymmetric(x_quant, scale, zero_point):
    return (x_quant.to(torch.float16) - zero_point) * scale
```

### 6.3. LLM Compressor: Thư Viện Quantization của vLLM

**LLM Compressor** (https://github.com/vllm-project/llm-compressor) là thư viện chính thức của vLLM project để quantize models.

#### Tính Năng Chính

1. **Nhiều Phương Pháp Quantization:**
   - GPTQ (Generative Pre-trained Transformer Quantization)
   - AWQ (Activation-aware Weight Quantization)
   - SmoothQuant
   - SparseGPT (sparsity + quantization)
   - Simple Post-Training Quantization (PTQ)

2. **Nhiều Formats:**
   - Weight-only: W8A16, W4A16, NVFP4
   - Weight-Activation: W8A8 (INT8 hoặc FP8)
   - Mixed precision per layer

3. **Seamless vLLM Integration:**
   - Output trực tiếp safetensors format
   - Load ngay vào vLLM mà không cần conversion

4. **HuggingFace Compatible:**
   - Hỗ trợ tất cả transformers models
   - Tự động model architecture detection

#### Installation

```bash
pip install llmcompressor

# With vLLM integration
pip install llmcompressor[vllm]
```

### 6.4. Các Phương Pháp Quantization

#### 6.4.1. GPTQ (Generative Pre-trained Transformer Quantization)

**Ý Tưởng:** Minimize reconstruction error khi quantize weights.

```
Original problem:
  Y = X @ W                    (FP16)

After quantization:
  Y_quant = X @ W_quant        (INT4/INT8)

Goal: minimize ||Y - Y_quant||²

GPTQ Algorithm:
1. For each layer, collect calibration data X
2. Compute Hessian of loss w.r.t. weights: H = X^T @ X
3. Quantize weights column-by-column, minimizing error using H
4. Update remaining weights to compensate for quantization error
```

**Implementation với LLM Compressor:**

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# Define quantization recipe
recipe = GPTQModifier(
    scheme="W4A16",              # 4-bit weights, 16-bit activations
    targets="Linear",             # Quantize all Linear layers
    ignore=["lm_head"],          # Except output projection
    group_size=128,              # Group quantization (128 weights share scale)
    actorder=True,               # Reorder weights by activation magnitude
    dampening_frac=0.01,         # Hessian dampening for stability
)

# Apply quantization
oneshot(
    model="meta-llama/Llama-2-7b-hf",
    dataset="wikitext",           # Calibration dataset
    recipe=recipe,
    output_dir="Llama-2-7b-GPTQ-4bit",
    num_calibration_samples=512,
    max_seq_length=2048,
)
```

**GPTQ Process Visualization:**

```
Layer-wise GPTQ:

Layer 0 (self_attn.q_proj):
  Input: W (FP16) [4096, 4096]
  Calibration: Collect inputs X [batch, seq, 4096]
  Compute: Hessian H = X^T @ X [4096, 4096]

  For each column i in W:
    1. Quantize W[:, i] to 4-bit
    2. Compute error: e_i = W[:, i] - W_quant[:, i]
    3. Update remaining columns to minimize error:
       W[:, j>i] -= (e_i @ H[i, j>i]) / H[i, i]

  Output: W_quant (INT4) + scales + zeros
  Memory: 4096 × 4096 × 0.5 bytes = 8 MB (vs 32 MB FP16)

Repeat for all layers...
```

#### 6.4.2. AWQ (Activation-aware Weight Quantization)

**Ý Tưởng:** Protect "salient" weights (có activations lớn) khỏi quantization error.

```
Observation:
- Weights with large activations contribute more to output
- Quantizing them causes large errors

AWQ Strategy:
1. Analyze activation magnitudes during calibration
2. Scale up important weights BEFORE quantization
3. Scale down activations correspondingly
4. Quantize scaled weights (less error for important ones)

Math:
  Original: Y = X @ W
  AWQ:      Y = (X @ s⁻¹) @ (s @ W_quant)
            where s = scale factor per channel
```

**Implementation:**

```python
from llmcompressor.modifiers.quantization import AWQModifier

recipe = AWQModifier(
    scheme="W4A16",
    targets="Linear",
    group_size=128,
    version="awq_v2",           # AWQ version 2 (improved)
    zero_point=True,            # Use asymmetric quantization
)

oneshot(
    model="meta-llama/Llama-2-13b-hf",
    dataset="pileval",
    recipe=recipe,
    output_dir="Llama-2-13b-AWQ-4bit",
    num_calibration_samples=512,
)
```

**AWQ vs GPTQ:**

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Calibration Time | Slow (Hessian computation) | Fast (activation analysis) |
| Quantization Quality | Very good | Excellent (better for 4-bit) |
| Runtime Overhead | None | Minimal (channel scaling) |
| Use Case | W4A16, W8A16 | W4A16 (best for 4-bit) |

#### 6.4.3. SmoothQuant

**Ý Tưởng:** "Smooth" activation outliers để dễ quantize hơn.

```
Problem:
- Activations have large outliers (some channels >> others)
- Hard to quantize with uniform scale

SmoothQuant:
  Y = X @ W
  Y = (X @ diag(s)⁻¹) @ (diag(s) @ W)
  Y = X' @ W'

  where:
  - s smooths X (reduce outliers)
  - Correspondingly scales W
  - Both X' and W' are easier to quantize
```

**Implementation:**

```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

# Combine SmoothQuant + Quantization
recipe = [
    SmoothQuantModifier(
        smoothing_strength=0.5,   # Alpha parameter (0-1)
        mappings=[                # Which layers to smooth
            [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*self_attn"],
            [["re:.*gate_proj", "re:.*up_proj"], "re:.*mlp"],
        ]
    ),
    GPTQModifier(
        scheme="W8A8",            # 8-bit weights AND activations
        targets="Linear",
        ignore=["lm_head"],
    ),
]

oneshot(
    model="meta-llama/Llama-2-7b-hf",
    dataset="ultrachat",
    recipe=recipe,
    output_dir="Llama-2-7b-SmoothQuant-W8A8",
    num_calibration_samples=512,
)
```

**Use Case:** W8A8 quantization (cần quantize cả activations).

#### 6.4.4. SparseGPT + Quantization

**Ý Tưởng:** Kết hợp sparsity (nhiều weights = 0) và quantization.

```
Sparsity: Set 50% of smallest weights to 0
  - Reduce memory (compressed storage)
  - Reduce compute (skip zero weights)

Structured Sparsity (2:4):
  - In every 4 consecutive weights, 2 are zero
  - Hardware-friendly (NVIDIA Sparse Tensor Cores)

Example:
  Original: [0.3, 0.1, -0.5, 0.2]
  2:4 Sparse: [0.3, 0, -0.5, 0]  (2 largest kept, 2 smallest → 0)
```

**Implementation:**

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.pruning import SparseGPTModifier

recipe = [
    SparseGPTModifier(
        sparsity=0.5,             # 50% weights → 0
        targets="Linear",
        # Optional: structured sparsity
        # structure="2:4",        # 2:4 sparsity pattern
    ),
    GPTQModifier(
        scheme="W4A16",
        targets="Linear",
        ignore=["lm_head"],
    ),
]

oneshot(
    model="meta-llama/Llama-2-7b-hf",
    dataset="wikitext",
    recipe=recipe,
    output_dir="Llama-2-7b-Sparse50-GPTQ4bit",
    num_calibration_samples=512,
)
```

**Benefits:**

```
Sparse 50% + INT4:
  Memory: 4096 × 4096 × 0.5 bytes × 0.5 sparsity = 4 MB
  Compute: ~2x faster (half the weights)
  Accuracy: Slight degradation (2-3% perplexity increase)
```

### 6.5. Quantization Schemes Comparison

#### Weight-Only Quantization (WxA16)

```
W4A16: 4-bit weights, 16-bit activations
W8A16: 8-bit weights, 16-bit activations

Pros:
  ✅ Memory savings (4x for W4, 2x for W8)
  ✅ Minimal accuracy loss
  ✅ Easy to implement

Cons:
  ❌ Compute still uses FP16 (dequantize before matmul)
  ❌ Memory bandwidth bottleneck (load + dequantize)

Use Case: Memory-bound inference (large models)
```

#### Weight-Activation Quantization (WxAx)

```
W8A8: 8-bit weights, 8-bit activations
W4A4: 4-bit weights, 4-bit activations (experimental)

Pros:
  ✅ Memory savings
  ✅ Compute savings (INT8 matmul faster than FP16)
  ✅ Lower latency

Cons:
  ❌ Harder to maintain accuracy (activations sensitive)
  ❌ Requires calibration for activations

Use Case: Compute-bound inference, low latency
```

#### FP8 Quantization

```
W8A8 (FP8): 8-bit floating point (E4M3 or E5M2 format)

Pros:
  ✅ Better dynamic range than INT8
  ✅ Hardware support (H100, Ada GPUs)
  ✅ Better accuracy than INT8

Cons:
  ❌ Requires new hardware
  ❌ Limited software support

Use Case: H100/Ada GPUs, W8A8 with best accuracy
```

**Comparison Table:**

| Scheme | Memory Savings | Speed | Accuracy | Hardware |
|--------|----------------|-------|----------|----------|
| **W4A16** | 4x | 1.2-1.5x | 95-98% | All GPUs |
| **W8A16** | 2x | 1.1-1.3x | 98-99% | All GPUs |
| **W8A8 (INT8)** | 4x | 2-3x | 90-95% | Ampere+ |
| **W8A8 (FP8)** | 4x | 2-4x | 95-98% | Hopper, Ada |
| **W4A4** | 8x | 3-5x | 85-90% | Experimental |

### 6.6. Advanced Features

#### 6.6.1. Group Quantization

```python
# Per-Channel Quantization
# Each output channel has its own scale
for i in range(out_channels):
    scale[i] = W[:, i].abs().max() / 127
    W_quant[:, i] = (W[:, i] / scale[i]).round()

# Group Quantization (group_size=128)
# Every 128 weights share a scale
for i in range(0, in_features, 128):
    group = W[i:i+128, :]
    scale = group.abs().max() / 127
    W_quant[i:i+128, :] = (group / scale).round()

# Benefits:
# - Finer granularity than per-channel
# - Less overhead than per-weight
# - group_size=128 is sweet spot for 4-bit
```

#### 6.6.2. Mixed Precision Quantization

```python
# Different layers with different precision
recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["re:.*attn.*"],   # Attention layers: 4-bit
    ),
    GPTQModifier(
        scheme="W8A16",
        targets=["re:.*mlp.*"],    # MLP layers: 8-bit
    ),
    # lm_head: keep FP16 (most sensitive)
]

# Rationale:
# - Attention layers: less sensitive → 4-bit OK
# - MLP layers: more parameters → 8-bit for balance
# - Output layer: keep FP16 for accuracy
```

#### 6.6.3. Dynamic Quantization (Activation)

```python
# Static Quantization (calibration-based)
# Scales computed during calibration, fixed at runtime
def static_quant_activation(x, scale, zero_point):
    return (x / scale + zero_point).round().clamp(0, 255)

# Dynamic Quantization (runtime)
# Scales computed per-batch at runtime
def dynamic_quant_activation(x):
    scale = (x.max() - x.min()) / 255
    zero_point = -x.min() / scale
    return (x / scale + zero_point).round().clamp(0, 255), scale, zero_point

# Trade-off:
# Static: Fast (no runtime overhead), less accurate (distribution shift)
# Dynamic: Slower (compute scales), more accurate
```

### 6.7. LLM Compressor Workflow

#### Complete Example: Quantize LLaMA-2 70B

```python
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# Step 1: Define quantization recipe
recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
    group_size=128,
    actorder=True,
    dampening_frac=0.01,
)

# Step 2: Apply quantization with calibration
oneshot(
    model="meta-llama/Llama-2-70b-hf",
    dataset="ultrachat",              # Calibration dataset
    recipe=recipe,
    output_dir="./Llama-2-70b-GPTQ-4bit",

    # Calibration settings
    num_calibration_samples=512,      # Number of samples
    max_seq_length=2048,               # Max sequence length

    # Memory optimization (for large models)
    device_map="auto",                 # Auto device placement
    offload_gradients=True,            # Offload to CPU
    split_frac=1.0,                    # Process entire model

    # Output settings
    save_compressed=True,              # Save in compressed format
)

print("Quantization complete!")
print(f"Model saved to: ./Llama-2-70b-GPTQ-4bit")
```

#### Step 3: Load in vLLM

```python
from vllm import LLM, SamplingParams

# Load quantized model
llm = LLM(
    model="./Llama-2-70b-GPTQ-4bit",
    quantization="gptq",               # Specify quantization method
    dtype="float16",                   # Activation dtype
    tensor_parallel_size=4,            # Use 4 GPUs
    gpu_memory_utilization=0.95,
)

# Generate
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
prompts = ["Explain quantum computing in simple terms."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

#### Memory Comparison

```
LLaMA-2 70B Memory Usage:

FP16 (original):
  Weights: 70B × 2 bytes = 140 GB
  KV Cache (batch=32): ~20 GB
  Activations: ~5 GB
  Total: ~165 GB
  → Requires 2x A100 80GB (with TP=2)

W4A16 (GPTQ):
  Weights: 70B × 0.5 bytes = 35 GB
  KV Cache (batch=32): ~20 GB
  Activations: ~5 GB
  Total: ~60 GB
  → Fits in 1x A100 80GB!

Savings: 165 GB → 60 GB (2.75x reduction)
```

### 6.8. Calibration Dataset Selection

Quantization accuracy phụ thuộc vào calibration dataset:

```python
# Popular calibration datasets

# 1. WikiText (general knowledge)
dataset = "wikitext"
# Pros: Diverse, well-formed text
# Cons: May not match specific domain

# 2. C4 (web crawl)
dataset = "allenai/c4"
# Pros: Very diverse, large
# Cons: Noisy, may have artifacts

# 3. UltraChat (conversational)
dataset = "ultrachat"
# Pros: Good for chatbots
# Cons: Less diverse than web data

# 4. PileVal (multi-domain)
dataset = "pileval"
# Pros: Mix of domains
# Cons: Smaller than C4

# 5. Custom dataset (domain-specific)
from datasets import load_dataset
dataset = load_dataset("json", data_files="my_data.jsonl")
# Pros: Perfect match for your use case
# Cons: Need to prepare data
```

**Best Practice:**

```python
# Use domain-matched calibration data
# Example: Medical chatbot

from datasets import load_dataset

# Load medical Q&A dataset
medical_data = load_dataset("medmcqa", split="train")

# Format for calibration
def format_sample(sample):
    return f"Question: {sample['question']}\nAnswer: {sample['answer']}"

formatted = medical_data.map(lambda x: {"text": format_sample(x)})

# Quantize with medical data
oneshot(
    model="meta-llama/Llama-2-7b-hf",
    dataset=formatted,
    recipe=recipe,
    num_calibration_samples=512,
)
```

### 6.9. Evaluation và Quality Assessment

#### Perplexity Evaluation

```python
from llmcompressor.evaluation import evaluate_perplexity

# Evaluate quantized model
results = evaluate_perplexity(
    model="./Llama-2-7b-GPTQ-4bit",
    dataset="wikitext",
    split="test",
)

print(f"Perplexity: {results['perplexity']:.2f}")

# Compare with baseline
baseline = evaluate_perplexity(
    model="meta-llama/Llama-2-7b-hf",
    dataset="wikitext",
    split="test",
)

print(f"Baseline perplexity: {baseline['perplexity']:.2f}")
print(f"Degradation: {(results['perplexity'] / baseline['perplexity'] - 1) * 100:.1f}%")
```

**Example Results:**

```
LLaMA-2 7B Quantization Quality:

Method          | Perplexity | Degradation | Memory
----------------|------------|-------------|--------
FP16 (baseline) | 5.47       | 0%          | 14 GB
W8A16 (GPTQ)    | 5.51       | +0.7%       | 7 GB
W4A16 (GPTQ)    | 5.68       | +3.8%       | 3.5 GB
W4A16 (AWQ)     | 5.62       | +2.7%       | 3.5 GB
W8A8 (SmoothQ)  | 5.89       | +7.7%       | 3.5 GB

Recommendation:
- W4A16 (AWQ): Best 4-bit quality
- W8A16 (GPTQ): Safe choice, minimal degradation
- W8A8: Use only if compute-bound
```

#### Task-Specific Evaluation

```python
# For downstream tasks, use lm-evaluation-harness
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=./Llama-2-7b-GPTQ-4bit",
    tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge"],
    batch_size=8,
)

# Compare accuracy on benchmarks
for task, score in results["results"].items():
    print(f"{task}: {score['acc']:.1%}")
```

### 6.10. Performance Benchmarks

#### Latency Comparison

```
LLaMA-2 13B on A100 40GB (batch_size=1, seq_len=512):

Format     | Prefill (ms) | Decode (ms/token) | Memory (GB)
-----------|--------------|-------------------|-------------
FP16       | 45           | 18                | 26
W8A16      | 42           | 16                | 13
W4A16 AWQ  | 50           | 14                | 7
W8A8 FP8   | 28           | 8                 | 13

Notes:
- W4A16: Slower prefill (dequant overhead), faster decode (less memory BW)
- W8A8: Fastest (INT8/FP8 kernels), but needs Ampere+
```

#### Throughput Comparison

```
LLaMA-2 70B on 4×A100 80GB (batch_size=32, output_len=128):

Format     | Throughput (tok/s) | GPU Memory | Cost
-----------|--------------------|-----------|---------
FP16       | 1200               | 280 GB    | 4×A100
W4A16 AWQ  | 2800               | 120 GB    | 2×A100

Savings:
- 2.3x higher throughput
- 2.3x lower GPU memory
- 2x fewer GPUs needed
→ ~4x cost reduction
```

### 6.11. Best Practices

#### Quantization Strategy Selection

```python
# Decision tree for quantization method

if model_size < 10B and memory_ok:
    # Small model, no quantization needed
    quantization = None

elif latency_critical and hardware == "H100":
    # Low latency + new hardware → FP8
    recipe = GPTQModifier(scheme="W8A8_fp8")

elif model_size < 30B:
    # Medium model → W4A16 (AWQ for best quality)
    recipe = AWQModifier(scheme="W4A16")

elif model_size >= 30B and memory_limited:
    # Large model, memory constrained → W4A16 GPTQ
    recipe = GPTQModifier(scheme="W4A16", group_size=128)

elif need_activation_quant:
    # W8A8 for compute speedup
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.5),
        GPTQModifier(scheme="W8A8"),
    ]
```

#### Hyperparameter Tuning

```python
# GPTQ hyperparameters

# group_size: Trade-off between quality and memory
group_size = 128    # Recommended (sweet spot)
# 64:  Better quality, more memory for scales
# 256: Worse quality, less memory

# dampening_frac: Hessian dampening for stability
dampening_frac = 0.01  # Default
# Increase if quantization is unstable (NaN, inf)

# actorder: Reorder by activation magnitude
actorder = True     # Usually better quality, slower calibration

# num_calibration_samples
num_samples = 512   # Recommended
# 128:  Fast, may underfit
# 1024: Slow, diminishing returns
```

#### Monitoring Quantization Process

```python
import logging
from llmcompressor import oneshot

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Monitor progress
oneshot(
    model="meta-llama/Llama-2-7b-hf",
    dataset="wikitext",
    recipe=recipe,
    output_dir="./output",

    # Logging
    oneshot_device="cuda:0",
    splits={"calibration": "train[:512]"},

    # Callbacks for monitoring
    # (check logs for layer-wise progress)
)

# Example log output:
# INFO: Quantizing layer model.layers.0.self_attn.q_proj
# INFO: ... (progress for each layer)
# INFO: Perplexity after quantization: 5.68
```

### 6.12. Common Issues và Solutions

#### Issue 1: Accuracy Degradation

```python
# Problem: Perplexity increased >10%

# Solutions:
# 1. Use better quantization method
recipe = AWQModifier(scheme="W4A16")  # Instead of naive PTQ

# 2. Increase calibration samples
num_calibration_samples = 1024  # Instead of 512

# 3. Use domain-matched calibration data
dataset = "my_domain_data"  # Instead of wikitext

# 4. Mixed precision (sensitive layers in higher precision)
recipe = [
    GPTQModifier(scheme="W8A16", targets=["re:.*lm_head.*"]),
    GPTQModifier(scheme="W4A16", targets=["re:.*"]),
]

# 5. Adjust group_size
group_size = 64  # Finer granularity
```

#### Issue 2: OOM During Quantization

```python
# Problem: Out of memory during calibration

# Solutions:
# 1. Use gradient checkpointing
oneshot(
    model="meta-llama/Llama-2-70b-hf",
    gradient_checkpointing=True,  # Trade speed for memory
)

# 2. Reduce calibration batch size
oneshot(
    model="...",
    calibration_batch_size=1,  # Default is 8
)

# 3. Offload to CPU
oneshot(
    model="...",
    device_map="auto",        # Auto device placement
    offload_folder="./offload",  # Offload to disk if needed
)

# 4. Quantize in stages (layer-by-layer)
# (Advanced: requires custom script)
```

#### Issue 3: Slow Calibration

```python
# Problem: Calibration takes too long

# Solutions:
# 1. Reduce calibration samples
num_calibration_samples = 128  # Instead of 512

# 2. Use faster quantization method
recipe = AWQModifier(...)  # Faster than GPTQ

# 3. Reduce sequence length
max_seq_length = 1024  # Instead of 2048

# 4. Disable actorder (if using GPTQ)
recipe = GPTQModifier(actorder=False)
```

### 6.13. Future Directions

Emerging quantization techniques:

1. **QuIP (Quantization with Incoherence Processing)**
   - Randomized Hadamard Transform before quantization
   - Better accuracy for low-bit (2-bit, 3-bit)

2. **GGUF Format (llama.cpp)**
   - Optimized for CPU inference
   - LLM Compressor adding support

3. **MX Formats (Microscaling)**
   - Block floating point
   - Better dynamic range

4. **4-bit Activations (W4A4)**
   - Requires careful calibration
   - Potential 8x speedup

---

## 7. Context Parallelism

### 7.1. Vấn Đề: Long Context Bottleneck

Khi context length tăng lên rất lớn (100K - 1M tokens), các chiến lược parallelism hiện tại gặp hạn chế.

**Context Parallelism** (còn gọi là **Sequence Parallelism**) là kỹ thuật phân tán **sequence dimension** thay vì model weights. vLLM hiện đang tích cực phát triển CP với các approaches khác nhau:

```
Current vLLM CP Status (2025):
─────────────────────────────────────────────────────────────────

Implemented:
✅ Basic CP với All-Gather approach
✅ Chunked prefill với ring attention
✅ Integration với TP (Hybrid TP+CP)

In Development:
🔄 Full Ring Attention optimization
🔄 Ulysses integration (via Snowflake Arctic)
🔄 Unified Sequence Parallel (USP) = Ulysses + Ring

Research/RFC:
📝 RFC #22693: Context Parallelism && Sequence Parallelism
📝 RFC #26133: Support Context Parallelism with Fully Sharded KV Cache
```

**Các approaches chính:**

```
Long Context Challenges:

Input: 1M tokens context
Model: LLaMA-2 70B

Problems:
1. KV Cache Size:
   - 1M tokens × 70B model → ~280 GB KV cache (chỉ riêng cache!)
   - Không fit vào 1 GPU (A100 80GB)

2. Prefill Computation:
   - Attention: O(N²) complexity (N = context length)
   - 1M tokens: 1T operations
   - Single GPU: ~30-60 seconds prefill latency

3. Tensor Parallelism Limitation:
   - TP shards model weights, NOT context
   - KV cache still full on each GPU
   - Communication overhead increases with context length
```

**Observation:** TP, PP, DP không giải quyết vấn đề long context hiệu quả.

**Giải Pháp:** Context Parallelism (CP) - Shard context/sequence length dimension.

### 7.2. Context Parallelism: Core Idea

#### Concept

Thay vì shard model (TP) hoặc layers (PP), Context Parallelism **shards sequence/context dimension** (T dimension).

```
Traditional Tensor Parallelism (TP):
Input: [Batch, SeqLen, HiddenDim]
Split HiddenDim across GPUs

Context Parallelism (CP):
Input: [Batch, SeqLen, HiddenDim]
Split SeqLen across GPUs
```

**Visual Comparison:**

```
Tensor Parallelism (TP=4):
Sequence: [Token 0, Token 1, ..., Token 1M]
          ↓
Each GPU processes ALL tokens, but split hidden_dim:

GPU0: [Token 0-1M] with hidden_dim[0:256]
GPU1: [Token 0-1M] with hidden_dim[256:512]
GPU2: [Token 0-1M] with hidden_dim[512:768]
GPU3: [Token 0-1M] with hidden_dim[768:1024]

KV Cache per GPU: 1M tokens (full sequence)

---

Context Parallelism (CP=4):
Sequence: [Token 0, Token 1, ..., Token 1M]
          ↓ Split by sequence length
GPU0: [Token 0-250K]     with full hidden_dim
GPU1: [Token 250K-500K]  with full hidden_dim
GPU2: [Token 500K-750K]  with full hidden_dim
GPU3: [Token 750K-1M]    with full hidden_dim

KV Cache per GPU: 250K tokens (1/4 of sequence)
```

### 7.3. Context Parallelism Mechanisms

#### 7.3.1. Prefill Context Parallelism

Trong **prefill phase**, cả Q, K, V đều được shard theo sequence dimension.

##### Approach 1: All-Gather Strategy (Short Context)

```python
class PrefillContextParallelAttention:
    def __init__(self, cp_size=4):
        self.cp_size = cp_size
        self.cp_rank = get_cp_rank()

    def forward(self, q, k, v):
        """
        Input (per GPU):
            q: (batch, seq_len/cp_size, num_heads, head_dim)
            k: (batch, seq_len/cp_size, num_heads, head_dim)
            v: (batch, seq_len/cp_size, num_heads, head_dim)

        Strategy:
            1. Each GPU holds partial Q, K, V (split by seq_len)
            2. All-gather K, V to get full context
            3. Compute local attention for local Q chunk
            4. Concatenate outputs
        """
        # Step 1: All-gather K, V across CP group
        k_full = all_gather(k, dim=1, group=cp_group)
        # k_full: (batch, seq_len, num_heads, head_dim)

        v_full = all_gather(v, dim=1, group=cp_group)
        # v_full: (batch, seq_len, num_heads, head_dim)

        # Step 2: Compute attention for local Q chunk
        # Q[i] can attend to ALL K, V (full context)
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1))
        # attn_scores: (batch, seq_len/cp_size, num_heads, seq_len)

        attn_weights = F.softmax(attn_scores / sqrt(head_dim), dim=-1)

        output = torch.matmul(attn_weights, v_full)
        # output: (batch, seq_len/cp_size, num_heads, head_dim)

        # Step 3: Each GPU returns its chunk of output
        return output
```

**Timeline:**

```
All-Gather Prefill CP (CP=4, seq_len=1M):

Step 1: All-Gather K, V
GPU0: Send K[0:250K], V[0:250K] → All GPUs
GPU1: Send K[250K:500K], V[250K:500K] → All GPUs
GPU2: Send K[500K:750K], V[500K:750K] → All GPUs
GPU3: Send K[750K:1M], V[750K:1M] → All GPUs

Step 2: Compute local attention
GPU0: Q[0:250K] @ K[0:1M] → Attention[0:250K]
GPU1: Q[250K:500K] @ K[0:1M] → Attention[250K:500K]
GPU2: Q[500K:750K] @ K[0:1M] → Attention[500K:750K]
GPU3: Q[750K:1M] @ K[0:1M] → Attention[750K:1M]

Step 3: Concatenate outputs
Output: [Attention[0:250K], Attention[250K:500K], ...]

Pros:
✅ Simple implementation
✅ Good for moderate context lengths

Cons:
❌ All-gather requires holding full K, V in memory
❌ Memory bottleneck for very long context (>1M tokens)
```

##### Approach 2: Ring Attention (Very Long Context)

Khi context quá dài, không thể all-gather toàn bộ K, V. **Ring Attention** giải quyết bằng cách:

```
Ring Attention Idea:
- Each GPU computes attention incrementally
- K, V chunks are passed in a ring pattern
- Accumulate partial attention results

Inspired by: "Ring Attention with Blockwise Transformers" (Liu et al., 2023)
```

**Implementation:**

```python
class RingAttention:
    def __init__(self, cp_size=4):
        self.cp_size = cp_size
        self.cp_rank = get_cp_rank()

    def forward(self, q, k, v):
        """
        Ring Attention for very long context

        Each GPU:
        1. Holds local Q, K, V chunks
        2. Iteratively receives K, V from neighbors
        3. Computes partial attention and accumulates
        4. Passes K, V to next GPU in ring
        """
        batch, seq_chunk, num_heads, head_dim = q.shape

        # Initialize output and normalizer
        output = torch.zeros_like(q)
        lse = torch.zeros(batch, seq_chunk, num_heads)  # log-sum-exp

        # Current K, V chunks (initially local)
        k_current = k.clone()
        v_current = v.clone()

        # Ring iterations: receive K, V from all GPUs
        for step in range(self.cp_size):
            # Step 1: Compute attention with current K, V chunk
            attn_scores = torch.matmul(q, k_current.transpose(-2, -1))
            attn_scores = attn_scores / sqrt(head_dim)

            # Compute log-sum-exp for numerical stability
            max_scores = attn_scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(attn_scores - max_scores)

            # Step 2: Accumulate partial attention
            # Using online softmax (numerically stable)
            new_lse = torch.log(torch.exp(lse - max_scores.squeeze(-1)) +
                                 exp_scores.sum(dim=-1))

            # Update output with weighted average
            attn_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
            partial_output = torch.matmul(attn_weights, v_current)

            # Merge with previous output
            old_weight = torch.exp(lse - new_lse).unsqueeze(-1)
            new_weight = torch.exp(max_scores.squeeze(-1) - new_lse).unsqueeze(-1)

            output = output * old_weight + partial_output * new_weight
            lse = new_lse

            # Step 3: Send current K, V to next GPU, receive from previous
            if step < self.cp_size - 1:
                k_current = ring_send_recv(k_current, src=prev_rank, dst=next_rank)
                v_current = ring_send_recv(v_current, src=prev_rank, dst=next_rank)

        return output

def ring_send_recv(tensor, src, dst):
    """
    Send tensor to dst, receive from src in ring pattern
    Uses PyTorch distributed primitives
    """
    send_op = dist.P2POp(dist.isend, tensor, dst)
    recv_tensor = torch.empty_like(tensor)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, src)

    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    return recv_tensor
```

**Ring Attention Timeline:**

```
Ring Attention (CP=4, seq_len=1M):

Initial state:
GPU0: Q[0:250K], K[0:250K], V[0:250K]
GPU1: Q[250K:500K], K[250K:500K], V[250K:500K]
GPU2: Q[500K:750K], K[500K:750K], V[500K:750K]
GPU3: Q[750K:1M], K[750K:1M], V[750K:1M]

Iteration 1: Compute with local K, V
GPU0: Attention(Q[0:250K], K[0:250K], V[0:250K]) → partial_output_0
GPU1: Attention(Q[250K:500K], K[250K:500K], V[250K:500K]) → partial_output_1
...

Iteration 2: Ring shift K, V
GPU0 receives K[750K:1M], V[750K:1M] from GPU3
GPU1 receives K[0:250K], V[0:250K] from GPU0
GPU2 receives K[250K:500K], V[250K:500K] from GPU1
GPU3 receives K[500K:750K], V[500K:750K] from GPU2

GPU0: Attention(Q[0:250K], K[750K:1M], V[750K:1M]) → accumulate
...

Iteration 3: Ring shift again
GPU0 receives K[500K:750K], V[500K:750K] from GPU2
...

Iteration 4: Final ring shift
GPU0 receives K[250K:500K], V[250K:500K] from GPU1
...

Result: Each GPU has computed full attention for its Q chunk

Pros:
✅ Constant memory per GPU (no all-gather)
✅ Supports arbitrarily long context (1M+ tokens)
✅ Memory efficient: each GPU holds only 1/cp_size of KV cache

Cons:
❌ More complex implementation
❌ Communication overhead (cp_size iterations)
❌ Slightly higher latency than all-gather (for moderate context)
```

**Memory Analysis:**

```
Context Length = 1M tokens
Hidden Dim = 4096
CP Size = 4

All-Gather Approach:
Per GPU Memory:
- Local Q: 250K × 4096 = 1 GB
- Full K, V (after all-gather): 2 × 1M × 4096 = 8 GB
Total: ~9 GB per GPU

Ring Attention:
Per GPU Memory:
- Local Q: 250K × 4096 = 1 GB
- Local K, V: 2 × 250K × 4096 = 2 GB
Total: ~3 GB per GPU

Savings: 9 GB / 3 GB = 3x memory reduction
```

#### 7.3.2. Decode Context Parallelism

Trong **decode phase**, chỉ có 1 query token mới, nhưng phải attend to toàn bộ KV cache.

##### Strategy: Sharded KV Cache với All-Reduce

```python
class DecodeContextParallelAttention:
    def __init__(self, cp_size=4):
        self.cp_size = cp_size
        self.cp_rank = get_cp_rank()

    def forward(self, q, kv_cache_shard):
        """
        Decode phase with CP

        Input:
            q: (batch, 1, num_heads, head_dim) - single new token
            kv_cache_shard: (batch, seq_len/cp_size, num_heads, head_dim)
                            - local shard of KV cache

        Strategy:
            1. Compute partial attention with local KV shard
            2. All-reduce to combine results from all GPUs
        """
        # Step 1: Compute attention with local KV shard
        k_shard, v_shard = kv_cache_shard

        attn_scores = torch.matmul(q, k_shard.transpose(-2, -1))
        # attn_scores: (batch, 1, num_heads, seq_len/cp_size)

        attn_scores = attn_scores / sqrt(head_dim)

        # Step 2: Compute log-sum-exp (for numerical stability)
        max_score = attn_scores.max(dim=-1, keepdim=True).values
        # max_score: (batch, 1, num_heads, 1)

        # All-reduce to get global max
        global_max = all_reduce(max_score, op=ReduceOp.MAX, group=cp_group)

        # Step 3: Compute exp and partial sum
        exp_scores = torch.exp(attn_scores - global_max)
        local_sum = exp_scores.sum(dim=-1, keepdim=True)

        # All-reduce to get global sum
        global_sum = all_reduce(local_sum, op=ReduceOp.SUM, group=cp_group)

        # Step 4: Compute local attention output
        local_output = torch.matmul(exp_scores, v_shard)
        # local_output: (batch, 1, num_heads, head_dim)

        # Step 5: All-reduce to combine outputs
        global_output = all_reduce(local_output, op=ReduceOp.SUM, group=cp_group)

        # Step 6: Normalize
        final_output = global_output / global_sum

        return final_output
```

**KV Cache Distribution:**

```
Decode CP (CP=4, cached_len=1M):

KV Cache Distribution (round-robin):
Token 0 → GPU 0
Token 1 → GPU 1
Token 2 → GPU 2
Token 3 → GPU 3
Token 4 → GPU 0  (wrap around)
...

After 1M tokens:
GPU 0: KV[0, 4, 8, 12, ..., 999996] (250K tokens)
GPU 1: KV[1, 5, 9, 13, ..., 999997] (250K tokens)
GPU 2: KV[2, 6, 10, 14, ..., 999998] (250K tokens)
GPU 3: KV[3, 7, 11, 15, ..., 999999] (250K tokens)

New token decode:
Query: (batch, 1, num_heads, head_dim)

GPU 0: Attention(Q, KV[0, 4, 8, ...]) → partial_output_0
GPU 1: Attention(Q, KV[1, 5, 9, ...]) → partial_output_1
GPU 2: Attention(Q, KV[2, 6, 10, ...]) → partial_output_2
GPU 3: Attention(Q, KV[3, 7, 11, ...]) → partial_output_3

All-Reduce: Sum all partial outputs → final output
```

### 7.4. Hybrid Parallelism: TP + CP

Context Parallelism thường được **kết hợp với Tensor Parallelism** (TP).

```python
# Hybrid TP + CP
# Example: TP=2, CP=4 (total 8 GPUs)

class HybridTPCPAttention:
    def __init__(self, tp_size=2, cp_size=4):
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.tp_rank = get_tp_rank()
        self.cp_rank = get_cp_rank()

    def forward(self, x):
        # x: (batch, seq_len/cp_size, hidden_dim)

        # Step 1: QKV projection (TP column parallel)
        qkv = self.qkv_proj(x)  # Column parallel (TP)
        # qkv: (batch, seq_len/cp_size, 3*num_heads_per_tp*head_dim)

        q, k, v = qkv.chunk(3, dim=-1)

        # Step 2: Attention (CP across sequence)
        if self.is_prefill:
            output = ring_attention(q, k, v, cp_group=self.cp_group)
        else:
            output = decode_cp_attention(q, k, v, cp_group=self.cp_group)

        # Step 3: Output projection (TP row parallel)
        output = self.o_proj(output)  # Row parallel (TP), all-reduce inside

        return output
```

**GPU Layout:**

```
8 GPUs: TP=2, CP=4

CP Group 0:
  GPU 0 (TP rank 0, CP rank 0): Sequence[0:250K], hidden_dim[0:2048]
  GPU 2 (TP rank 0, CP rank 1): Sequence[250K:500K], hidden_dim[0:2048]
  GPU 4 (TP rank 0, CP rank 2): Sequence[500K:750K], hidden_dim[0:2048]
  GPU 6 (TP rank 0, CP rank 3): Sequence[750K:1M], hidden_dim[0:2048]

CP Group 1:
  GPU 1 (TP rank 1, CP rank 0): Sequence[0:250K], hidden_dim[2048:4096]
  GPU 3 (TP rank 1, CP rank 1): Sequence[250K:500K], hidden_dim[2048:4096]
  GPU 5 (TP rank 1, CP rank 2): Sequence[500K:750K], hidden_dim[2048:4096]
  GPU 7 (TP rank 1, CP rank 3): Sequence[750K:1M], hidden_dim[2048:4096]

Communication Pattern:
- CP: Ring communication within CP group (for attention)
- TP: All-reduce within TP group (for linear layers)
```

### 7.5. Configuration và Usage

#### vLLM Command Line

```bash
# Enable Context Parallelism in vLLM

# Basic CP (CP=4)
vllm serve meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1 \
    --context-parallel-size 4

# Hybrid TP + CP (TP=2, CP=4, total 8 GPUs)
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 2 \
    --context-parallel-size 4

# With long context
vllm serve gradientai/Llama-3-70B-Instruct-Gradient-1048k \
    --tensor-parallel-size 4 \
    --context-parallel-size 4 \
    --max-model-len 1048576  # 1M context
```

#### Python API

```python
from vllm import LLM, SamplingParams

# Initialize with CP
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,
    context_parallel_size=4,  # Enable CP
    max_model_len=262144,     # 256K context
    gpu_memory_utilization=0.95,
)

# Long context inference
prompt = "..." * 100000  # Very long prompt
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### 7.6. Performance Analysis

#### Memory Savings

```
Benchmark: LLaMA-2 70B, Context Length = 1M tokens

Without CP (TP=8):
Per GPU:
- Model weights: 70B / 8 = 8.75B params × 2 bytes = 17.5 GB
- KV cache: 1M tokens × 8192 hidden × 2 bytes = 16 GB
- Activations: ~5 GB
Total per GPU: ~38.5 GB
Total across 8 GPUs: 308 GB

With CP=4, TP=2 (8 GPUs):
Per GPU:
- Model weights: 70B / 2 = 35B params × 2 bytes = 70 GB / 4 (TP) = 17.5 GB
- KV cache: (1M / 4) tokens × 8192 hidden × 2 bytes = 4 GB
- Activations: ~5 GB
Total per GPU: ~26.5 GB
Total across 8 GPUs: 212 GB

Savings: 308 GB → 212 GB (31% reduction)
```

#### Latency Analysis

```
Prefill Latency (1M tokens, LLaMA-2 70B):

TP=8 only:
- Attention: O(N²) on each GPU
- Time: ~45 seconds

TP=2, CP=4 (Ring Attention):
- Attention: O((N/4)²) per GPU, but 4 ring iterations
- Communication: 4 × (P2P latency)
- Time: ~18 seconds

Speedup: 2.5x faster

Decode Latency (1M context):
TP=8 only:
- KV cache: 16 GB per GPU
- Memory BW limited: ~25 ms/token

TP=2, CP=4:
- KV cache: 4 GB per GPU
- Memory BW limited: ~15 ms/token
- All-reduce overhead: +2 ms
- Total: ~17 ms/token

Speedup: 1.5x faster
```

### 7.7. Use Cases

#### 1. Long Document QA

```python
# Scenario: Q&A over 500-page document
document = load_document("long_report.pdf")  # ~200K tokens

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,
    context_parallel_size=4,
    max_model_len=262144,
)

prompt = f"{document}\n\nQuestion: Summarize the key findings."
output = llm.generate([prompt])
```

#### 2. Multi-Document RAG

```python
# Scenario: RAG with 50 retrieved documents
documents = retrieve_documents(query, top_k=50)  # ~100K tokens total
context = "\n\n".join(documents)

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    context_parallel_size=2,
    max_model_len=131072,
)

prompt = f"Context:\n{context}\n\nQuestion: {query}"
output = llm.generate([prompt])
```

#### 3. Long Conversation History

```python
# Scenario: Multi-turn conversation with full history
conversation_history = load_conversation()  # 50K tokens

llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=1,
    context_parallel_size=2,
    max_model_len=65536,
)

new_message = "Tell me more about that topic."
full_prompt = f"{conversation_history}\nUser: {new_message}\nAssistant:"
output = llm.generate([full_prompt])
```

### 7.8. Limitations và Best Practices

#### Limitations

1. **Communication Overhead:**
   - Ring Attention requires `cp_size` iterations
   - Not beneficial for short context (<64K tokens)

2. **Implementation Complexity:**
   - More complex than TP/PP
   - Debugging distributed attention is harder

3. **Model Support:**
   - Not all models support CP yet
   - Requires specific attention implementation

#### Best Practices

```python
# When to use CP:

if context_length > 100_000:
    # Very long context → CP highly recommended
    cp_size = 4 or 8

elif context_length > 32_000:
    # Long context → CP beneficial
    cp_size = 2 or 4

else:
    # Normal context → CP not needed
    cp_size = 1  # Disabled

# Tuning CP size:
# - cp_size = 2: Minimal overhead, good for 32K-128K
# - cp_size = 4: Balanced, good for 128K-512K
# - cp_size = 8: Aggressive, good for 512K-2M

# Combine with TP:
if model_size > 70B:
    # Large model → Use both TP and CP
    tp_size = 4
    cp_size = 4
else:
    # Smaller model → CP only
    tp_size = 1
    cp_size = 4

# Memory tuning:
gpu_memory_utilization = 0.9  # Leave room for activation
```

#### Configuration Examples

```python
# Example 1: Long document (256K context)
# Model: LLaMA-2 70B
# Hardware: 8×A100 80GB

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,   # TP for model size
    context_parallel_size=4,  # CP for context length
    max_model_len=262144,
    gpu_memory_utilization=0.9,
)

# Example 2: Extreme long context (1M tokens)
# Model: Gradient Llama-3 70B 1M
# Hardware: 16×A100 80GB

llm = LLM(
    model="gradientai/Llama-3-70B-Instruct-Gradient-1048k",
    tensor_parallel_size=4,   # TP=4 for model
    context_parallel_size=4,  # CP=4 for context
    max_model_len=1048576,    # 1M context
    gpu_memory_utilization=0.95,
)

# Example 3: Moderate context, smaller model
# Model: LLaMA-2 13B
# Context: 128K tokens
# Hardware: 4×A100 40GB

llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=1,   # Model fits 1 GPU
    context_parallel_size=4,  # CP=4 for long context
    max_model_len=131072,
    gpu_memory_utilization=0.85,
)
```

### 7.9. Ulysses vs Ring Attention: Deep Comparison

vLLM đang tích hợp cả hai approaches chính cho Context Parallelism:

#### 7.9.1. Ulysses (DeepSpeed-Ulysses)

**Ý Tưởng:** Ulysses shard attention heads thay vì sequence. Mỗi GPU xử lý một subset của attention heads.

```python
class UlyssesAttention:
    """
    Ulysses Sequence Parallelism

    Key Insight:
    - Shard Q, K, V theo HEAD dimension
    - All-to-All để redistribute data
    - Mỗi GPU compute attention cho subset of heads

    Communication Pattern:
    1. All-to-All scatter: Q, K, V từ seq-split → head-split
    2. Local attention computation
    3. All-to-All gather: Output từ head-split → seq-split
    """

    def __init__(self, sp_size: int, num_heads: int):
        self.sp_size = sp_size
        self.num_heads = num_heads
        self.heads_per_rank = num_heads // sp_size

    def forward(self, q, k, v):
        """
        Ulysses attention forward pass

        Input (sequence-split):
            q, k, v: (batch, seq_len/sp_size, num_heads, head_dim)

        Step 1: All-to-All to head-split
            q, k, v: (batch, seq_len, num_heads/sp_size, head_dim)

        Step 2: Local attention (full sequence, partial heads)

        Step 3: All-to-All back to sequence-split
            output: (batch, seq_len/sp_size, num_heads, head_dim)
        """
        batch, seq_chunk, num_heads, head_dim = q.shape

        # Step 1: All-to-All (sequence-split → head-split)
        # Each GPU sends its seq chunk to all GPUs
        # Each GPU receives all seq but only its heads
        q_heads = all_to_all_seq_to_head(q, self.sp_size)
        k_heads = all_to_all_seq_to_head(k, self.sp_size)
        v_heads = all_to_all_seq_to_head(v, self.sp_size)
        # Shape: (batch, seq_len, heads_per_rank, head_dim)

        # Step 2: Local attention (standard FlashAttention)
        # Full sequence length, but only local heads
        attn_output = flash_attention(q_heads, k_heads, v_heads)
        # Shape: (batch, seq_len, heads_per_rank, head_dim)

        # Step 3: All-to-All (head-split → sequence-split)
        output = all_to_all_head_to_seq(attn_output, self.sp_size)
        # Shape: (batch, seq_chunk, num_heads, head_dim)

        return output

def all_to_all_seq_to_head(tensor, sp_size):
    """
    Redistribute tensor from sequence-split to head-split

    Input:  (batch, seq_len/sp_size, num_heads, head_dim) per GPU
    Output: (batch, seq_len, num_heads/sp_size, head_dim) per GPU
    """
    batch, seq_chunk, num_heads, head_dim = tensor.shape
    seq_len = seq_chunk * sp_size
    heads_per_rank = num_heads // sp_size

    # Reshape for all-to-all
    tensor = tensor.reshape(batch, seq_chunk, sp_size, heads_per_rank, head_dim)
    tensor = tensor.transpose(1, 2)  # (batch, sp_size, seq_chunk, ...)

    # All-to-All communication
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor)

    # Reshape back
    output = output.transpose(1, 2).reshape(batch, seq_len, heads_per_rank, head_dim)
    return output
```

**Ulysses Communication Pattern:**

```
Ulysses All-to-All (SP=4, seq=1M, heads=32):

Initial State (sequence-split):
GPU0: Q[0:250K],   heads[0:32]
GPU1: Q[250K:500K], heads[0:32]
GPU2: Q[500K:750K], heads[0:32]
GPU3: Q[750K:1M],  heads[0:32]

After All-to-All (head-split):
GPU0: Q[0:1M], heads[0:8]    ← Full sequence, 8 heads
GPU1: Q[0:1M], heads[8:16]   ← Full sequence, 8 heads
GPU2: Q[0:1M], heads[16:24]  ← Full sequence, 8 heads
GPU3: Q[0:1M], heads[24:32]  ← Full sequence, 8 heads

Communication Volume: 2 × All-to-All
- Each GPU sends: seq_chunk × num_heads × head_dim
- Each GPU receives: seq_len × heads_per_rank × head_dim
- Total per GPU: 2 × (seq_len × heads × head_dim) / sp_size

Note: Volume SAME regardless of sequence length!
```

#### 7.9.2. Ring Attention Detailed

**Ý Tưởng:** Pass K, V chunks in a ring pattern, accumulate partial attention.

```python
class RingAttentionDetailed:
    """
    Ring Attention với Online Softmax

    Key Innovation:
    - Process K, V in chunks (ring pattern)
    - Use online softmax to accumulate across chunks
    - Memory: O(seq_len / cp_size) per GPU
    """

    def __init__(self, cp_size: int):
        self.cp_size = cp_size
        self.cp_rank = dist.get_rank()

    def forward(self, q, k, v):
        """
        Ring Attention with online softmax accumulation

        Algorithm:
        for step in range(cp_size):
            1. Compute partial attention with current K, V chunk
            2. Accumulate using online softmax formula
            3. Ring-shift K, V to next GPU
        """
        batch, seq_chunk, num_heads, head_dim = q.shape

        # Initialize accumulators for online softmax
        # m: running max of attention scores
        # l: running sum of exp(scores - m)
        # o: running output (weighted by l)
        m = torch.full((batch, seq_chunk, num_heads), -float('inf'))
        l = torch.zeros(batch, seq_chunk, num_heads)
        o = torch.zeros(batch, seq_chunk, num_heads, head_dim)

        # Current K, V (will be ring-shifted)
        k_curr = k.clone()
        v_curr = v.clone()

        for step in range(self.cp_size):
            # Determine which chunk we're processing
            chunk_idx = (self.cp_rank - step) % self.cp_size

            # Apply causal mask if needed
            # (tokens can only attend to earlier tokens)
            if self.causal:
                mask = self._compute_causal_mask(
                    q_chunk_idx=self.cp_rank,
                    kv_chunk_idx=chunk_idx
                )
            else:
                mask = None

            # === Online Softmax Update ===
            # Compute attention scores for this K chunk
            scores = torch.matmul(q, k_curr.transpose(-2, -1)) / math.sqrt(head_dim)
            # scores: (batch, seq_chunk, num_heads, seq_chunk)

            if mask is not None:
                scores = scores.masked_fill(mask, -float('inf'))

            # Compute local max and exp
            m_chunk = scores.max(dim=-1).values  # (batch, seq_chunk, num_heads)
            exp_scores = torch.exp(scores - m_chunk.unsqueeze(-1))
            l_chunk = exp_scores.sum(dim=-1)

            # Compute local attention output
            o_chunk = torch.matmul(exp_scores, v_curr)
            o_chunk = o_chunk / l_chunk.unsqueeze(-1)

            # === Merge with accumulated state ===
            # Online softmax: merge new chunk with accumulated
            m_new = torch.maximum(m, m_chunk)

            # Rescale factors
            exp_m = torch.exp(m - m_new)
            exp_m_chunk = torch.exp(m_chunk - m_new)

            # Update accumulator
            l_new = exp_m * l + exp_m_chunk * l_chunk
            o_new = (exp_m.unsqueeze(-1) * l.unsqueeze(-1) * o +
                     exp_m_chunk.unsqueeze(-1) * l_chunk.unsqueeze(-1) * o_chunk) / l_new.unsqueeze(-1)

            # Update state
            m, l, o = m_new, l_new, o_new

            # === Ring shift K, V to next GPU ===
            if step < self.cp_size - 1:
                k_curr = self._ring_send_recv(k_curr)
                v_curr = self._ring_send_recv(v_curr)

        return o

    def _ring_send_recv(self, tensor):
        """
        Ring communication: send to next, receive from previous
        """
        send_rank = (self.cp_rank + 1) % self.cp_size
        recv_rank = (self.cp_rank - 1) % self.cp_size

        recv_tensor = torch.empty_like(tensor)

        # Overlap send/recv
        send_op = dist.P2POp(dist.isend, tensor, send_rank)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank)

        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

        return recv_tensor
```

**Ring Attention Communication Pattern:**

```
Ring Attention (CP=4, seq=1M):

Step 0: Process local K, V
GPU0: Q[0:250K] × K[0:250K]     → partial_0
GPU1: Q[250K:500K] × K[250K:500K] → partial_1
GPU2: Q[500K:750K] × K[500K:750K] → partial_2
GPU3: Q[750K:1M] × K[750K:1M]   → partial_3

Step 1: Ring shift K, V (GPU_i sends to GPU_(i+1))
GPU0 receives K[750K:1M] from GPU3
GPU1 receives K[0:250K] from GPU0
GPU2 receives K[250K:500K] from GPU1
GPU3 receives K[500K:750K] from GPU2

GPU0: Q[0:250K] × K[750K:1M]   → accumulate
...

Step 2: Ring shift again
...

Step 3: Final ring shift
After 4 steps: each GPU has computed attention over full sequence

Communication:
- cp_size rounds of P2P communication
- Each round: seq_chunk × num_heads × head_dim × 2 (K and V)
- Total: seq_len × num_heads × head_dim × 2 (same as data size)
```

#### 7.9.3. Ulysses vs Ring Attention Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Ulysses vs Ring Attention Comparison                  │
├──────────────────────┬──────────────────────┬───────────────────────────┤
│ Aspect               │ Ulysses              │ Ring Attention            │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ Communication Type   │ All-to-All           │ Point-to-Point (P2P)      │
│                      │ (collective)         │ (ring pattern)            │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ Communication Volume │ O(seq × heads × dim) │ O(seq × heads × dim)      │
│                      │ Fixed per step       │ cp_size rounds            │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ Communication Rounds │ 2 All-to-All         │ cp_size P2P rounds        │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ Parallelism Limit    │ num_heads            │ Unlimited                 │
│                      │ (cannot exceed)      │ (can split arbitrarily)   │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ GQA/MQA Support      │ ❌ Poor              │ ✅ Good                   │
│                      │ (few KV heads)       │ (split by sequence)       │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ FlashAttention       │ ✅ Optimal           │ ⚠️ Sub-optimal            │
│ Efficiency           │ (full seq per GPU)   │ (chunked sequences)       │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ Memory per GPU       │ O(seq_len)           │ O(seq_len / cp_size)      │
│                      │ (full seq needed)    │ (truly distributed)       │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ Bandwidth Util       │ Higher               │ Lower                     │
│                      │ (AlltoAll optimized) │ (P2P less efficient)      │
├──────────────────────┼──────────────────────┼───────────────────────────┤
│ Best For             │ MHA models           │ GQA/MQA models            │
│                      │ Small CP (2-8)       │ Large CP (8+)             │
│                      │ NVLink interconnect  │ Any interconnect          │
└──────────────────────┴──────────────────────┴───────────────────────────┘
```

#### 7.9.4. Unified Sequence Parallel (USP): Hybrid Approach

**USP kết hợp Ulysses và Ring Attention** để tận dụng ưu điểm của cả hai:

```python
class UnifiedSequenceParallel:
    """
    USP = Ulysses + Ring Attention (Hybrid)

    Strategy:
    1. First, apply Ulysses up to num_heads
    2. If more parallelism needed, add Ring Attention on top

    Example: 32 heads, want CP=16
    - Ulysses: SP=8 (limited by GQA num_kv_heads)
    - Ring: RP=2 (additional 2x)
    - Total: 8 × 2 = 16x parallelism
    """

    def __init__(self, num_heads: int, num_kv_heads: int, total_cp: int):
        # Ulysses limited by num_kv_heads (for GQA)
        self.ulysses_size = min(num_kv_heads, total_cp)

        # Ring handles the rest
        self.ring_size = total_cp // self.ulysses_size

        self.ulysses = UlyssesAttention(self.ulysses_size, num_heads)
        self.ring = RingAttentionDetailed(self.ring_size)

    def forward(self, q, k, v):
        """
        Hybrid execution:
        1. Ulysses All-to-All to head-split
        2. Ring attention over sequence chunks
        3. Ulysses All-to-All back to seq-split
        """
        # Step 1: Ulysses scatter (seq-split → head-split)
        q_heads = self.ulysses.scatter_to_heads(q)
        k_heads = self.ulysses.scatter_to_heads(k)
        v_heads = self.ulysses.scatter_to_heads(v)

        # Step 2: Ring attention (if ring_size > 1)
        if self.ring_size > 1:
            output = self.ring.forward(q_heads, k_heads, v_heads)
        else:
            output = flash_attention(q_heads, k_heads, v_heads)

        # Step 3: Ulysses gather (head-split → seq-split)
        output = self.ulysses.gather_from_heads(output)

        return output
```

**USP Topology:**

```
USP Example: CP=16 = Ulysses(8) × Ring(2)

16 GPUs organized as:
├── Ring Group 0 (Ulysses 8 GPUs)
│   ├── GPU 0  (heads 0-3,   seq chunk 0)
│   ├── GPU 1  (heads 4-7,   seq chunk 0)
│   ├── GPU 2  (heads 8-11,  seq chunk 0)
│   ├── GPU 3  (heads 12-15, seq chunk 0)
│   ├── GPU 4  (heads 16-19, seq chunk 0)
│   ├── GPU 5  (heads 20-23, seq chunk 0)
│   ├── GPU 6  (heads 24-27, seq chunk 0)
│   └── GPU 7  (heads 28-31, seq chunk 0)
│
└── Ring Group 1 (Ulysses 8 GPUs)
    ├── GPU 8  (heads 0-3,   seq chunk 1)
    ├── GPU 9  (heads 4-7,   seq chunk 1)
    ...
    └── GPU 15 (heads 28-31, seq chunk 1)

Communication:
1. Within each Ring Group: Ulysses All-to-All
2. Between Ring Groups: Ring attention P2P

Result: 16x sequence parallelism!
```

### 7.10. Comparison with Other Approaches

| Approach | Context Scaling | Memory | Latency | Complexity |
|----------|----------------|--------|---------|------------|
| **Vanilla Attention** | O(N²) | N | O(N²) | Low |
| **Flash Attention** | O(N²) optimized | N | O(N²) faster | Medium |
| **Tensor Parallel** | O(N²) | N (replicated) | O(N²) | Medium |
| **Ulysses** | O(N²) | N (full seq per GPU) | O(N²) + AlltoAll | Medium |
| **Ring Attention** | O((N/k)²) per GPU | N/k | O((N/k)²) + k×P2P | High |
| **USP (Hybrid)** | O((N/k)²) per GPU | N/k | Optimized | Very High |

**When to Use:**

- **Context < 32K:** Standard TP (no CP needed)
- **32K < Context < 128K:** Ulysses (if enough heads) or CP=2-4
- **128K < Context < 1M:** USP or Ring Attention với CP=4-8
- **Context > 1M:** USP hoặc Ring Attention với CP=8+ (essential)

### 7.10. Future Directions

Emerging techniques for long context:

1. **Hybrid CP + Sparse Attention:**
   - Combine CP with attention sparsity patterns
   - Reduce computation from O(N²) to O(N log N)

2. **Hierarchical Context Parallelism:**
   - Multi-level CP (e.g., CP across nodes, TP within nodes)
   - Better scaling for multi-node deployments

3. **Adaptive CP:**
   - Dynamically adjust cp_size based on runtime context length
   - Overhead reduction for variable-length inputs

4. **KV Cache Compression with CP:**
   - Compress KV cache within each CP shard
   - Further memory savings

---

## 8. Disaggregated Prefill and Decode

### 8.1. Vấn Đề: Phase Interference và Resource Mismatch

LLM inference có 2 phases với đặc điểm **trái ngược nhau**:

```python
# Phase 1: Prefill (Process Input Prompt)
def prefill_phase(prompt_tokens):
    """
    - Process nhiều tokens cùng lúc (100-10,000+ tokens)
    - Compute-bound (matmul intensive)
    - High throughput
    - Burstable workload
    """
    kv_cache = model.forward(prompt_tokens)  # e.g., 2048 tokens
    return kv_cache

# Phase 2: Decode (Generate Output)
def decode_phase(kv_cache):
    """
    - Process 1 token mỗi iteration
    - Memory-bandwidth-bound (load KV cache)
    - Low latency requirement
    - Continuous, predictable workload
    """
    for _ in range(max_new_tokens):
        next_token = model.forward(last_token, kv_cache)  # 1 token
        yield next_token
```

**Vấn Đề khi chạy chung trên cùng GPU:**

```
Monolithic System (Prefill + Decode cùng GPU):

GPU Timeline:
t=0:   [PREFILL────────────] (long, compute-intensive)
t=50:  [D][D][D][D][D][D]... (decode blocked, high latency)
t=100: [PREFILL────] (new request)
t=120: [D][D][D]... (decode latency spikes)

Problems:
1. Phase Interference:
   - Prefill blocks decode → high inter-token latency (ITL)
   - Decode underutilizes GPU → low throughput

2. Resource Mismatch:
   - Prefill needs compute (SM cores)
   - Decode needs bandwidth (memory BW)
   - Cannot optimize for both simultaneously

3. Tail ITL:
   - P99 ITL can be 10-100x higher than median
   - Unpredictable user experience
```

**Metrics:**

```
Monolithic vLLM (LLaMA-2 70B, mixed workload):

Time to First Token (TTFT): 500-2000ms (high variance)
Inter-Token Latency (ITL):
  - Median: 30ms
  - P99: 300ms (10x higher!)
  - P99.9: 1000ms (blocked by long prefills)

Throughput: 1500 tokens/sec
GPU Utilization: 70-85% (imbalanced)
```

### 8.2. Giải Pháp: Disaggregated Prefill and Decode

#### Ý Tưởng

Chạy **2 vLLM instances riêng biệt**:

1. **Prefill Instance**: Chuyên xử lý prefill phase
2. **Decode Instance**: Chuyên xử lý decode phase

```
Disaggregated Architecture:

┌─────────────────────────────────────────────────────────┐
│                      Client Requests                     │
└─────────────────────────────────────────────────────────┘
                          ↓
         ┌────────────────┴────────────────┐
         ↓                                  ↓
┌──────────────────────┐          ┌──────────────────────┐
│   Prefill Instance   │          │   Decode Instance    │
│  (GPU 0 or Cluster)  │          │  (GPU 1 or Cluster)  │
├──────────────────────┤          ├──────────────────────┤
│ • Process prompts    │   KV     │ • Generate tokens    │
│ • Compute KV cache   │  Cache   │ • Use cached KV      │
│ • Optimized for      │ Transfer │ • Optimized for      │
│   compute throughput │─────────▶│   low latency        │
│                      │          │                      │
│ Config:              │          │ Config:              │
│ • TP=4 (parallelism) │          │ • TP=2 (less memory) │
│ • Large batch        │          │ • Small batch        │
│ • No decode tasks    │          │ • No prefill tasks   │
└──────────────────────┘          └──────────────────────┘
```

#### KV Cache Transfer Connector

vLLM cung cấp **connector** để transfer KV cache từ prefill → decode instance:

```python
from vllm import LLM
from vllm.distributed.kv_transfer.kv_connector import (
    KVConnectorFactory,
    KVConnectorConfig,
)

# === Prefill Instance ===
prefill_config = KVConnectorConfig(
    kv_role="kv_producer",           # Produce KV cache
    kv_connector_type="TensorParallelConnector",
    kv_buffer_size=1e9,              # 1GB buffer
)

prefill_llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,          # TP=4 for high compute
    kv_connector_config=prefill_config,
    # Only prefill, no decode
    max_num_batched_tokens=16384,    # Large batch for prefill
)

# === Decode Instance ===
decode_config = KVConnectorConfig(
    kv_role="kv_consumer",           # Consume KV cache
    kv_connector_type="TensorParallelConnector",
    kv_buffer_size=1e9,
)

decode_llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,          # TP=2 (less parallelism needed)
    kv_connector_config=decode_config,
    # Only decode
    max_num_seqs=256,                # Many concurrent decodes
)
```

### 8.3. Cơ Chế Hoạt Động

#### Request Flow

```python
# Step-by-step request processing

# 1. Client sends request
request = {
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 100,
}

# 2. Prefill Instance processes prompt
prefill_output = prefill_llm.generate(
    prompts=[request["prompt"]],
    sampling_params=SamplingParams(max_tokens=0),  # Only prefill
)

# KV cache computed:
# - kv_cache shape: (num_layers, batch, num_heads, seq_len, head_dim)
# - For LLaMA-2 70B: ~2GB per request

# 3. KV Cache Transfer (via connector)
# - Prefill instance sends KV cache to decode instance
# - Transfer mechanism: shared memory, RDMA, or TCP/IP
# - Compressed format to reduce bandwidth

connector.transfer_kv_cache(
    request_id=request_id,
    kv_cache=kv_cache,
    metadata={
        "prompt_tokens": prompt_tokens,
        "num_layers": num_layers,
    }
)

# 4. Decode Instance receives KV cache and generates
decode_output = decode_llm.generate(
    prompts=[request["prompt"]],      # Metadata only
    kv_cache_from_prefill=kv_cache,   # Reuse prefill KV
    sampling_params=SamplingParams(max_tokens=100),
)

# 5. Stream tokens back to client
for token in decode_output:
    yield token
```

#### Timeline Comparison

```
Monolithic (Single Instance):

Request 1:
├─ Prefill: [████████] 200ms
└─ Decode:  [█][█][█][█]... (blocked by Request 2 prefill)
             ↑  ↑  ↑  ↑
            30ms 150ms (spike!) 35ms 40ms

Request 2:
├─ Prefill: [████████████] 400ms (blocks Request 1 decode)
└─ Decode:  [█][█][█]...

Problem: ITL spikes when prefills happen
─────────────────────────────────────────────────────────

Disaggregated (Separate Instances):

Prefill Instance (GPU 0):
Request 1: [████████] 200ms
Request 2:           [████████████] 400ms
Request 3:                         [██████] 150ms

Decode Instance (GPU 1):
Request 1:           [█][█][█][█][█][█]... (stable latency)
                      ↑  ↑  ↑  ↑  ↑  ↑
                     30ms 30ms 30ms 30ms 30ms
Request 2:                         [█][█][█]...
Request 3:                                   [█][█]...

Benefit: Consistent ITL, no interference
```

### 8.4. Configuration và Optimization

#### Prefill Instance Configuration

```python
prefill_llm = LLM(
    model="meta-llama/Llama-2-70b-hf",

    # Parallelism: High TP for compute throughput
    tensor_parallel_size=4,          # More GPUs for compute

    # Batching: Large batches for throughput
    max_num_batched_tokens=16384,    # 16K tokens per batch
    max_num_seqs=64,                 # Many concurrent prefills

    # Memory: Less memory needed (no long-running sequences)
    gpu_memory_utilization=0.95,

    # No decode optimization needed
    enable_chunked_prefill=False,    # Process entire prompts

    # KV Transfer
    kv_connector_config=KVConnectorConfig(
        kv_role="kv_producer",
        kv_connector_type="TensorParallelConnector",
        kv_buffer_size=2e9,          # 2GB buffer for transfers
    ),
)
```

#### Decode Instance Configuration

```python
decode_llm = LLM(
    model="meta-llama/Llama-2-70b-hf",

    # Parallelism: Lower TP (decode is memory-bound)
    tensor_parallel_size=2,          # Less parallelism needed

    # Batching: Optimize for low latency
    max_num_seqs=256,                # Many concurrent generations
    max_num_batched_tokens=4096,     # Smaller batches for latency

    # Memory: More memory for KV cache (many sequences)
    gpu_memory_utilization=0.90,
    enable_prefix_caching=True,      # Cache common prefixes

    # Decode optimization
    scheduler_config=SchedulerConfig(
        max_model_len=4096,
        # Prioritize fairness and low latency
    ),

    # KV Transfer
    kv_connector_config=KVConnectorConfig(
        kv_role="kv_consumer",
        kv_connector_type="TensorParallelConnector",
        kv_buffer_size=2e9,
    ),
)
```

### 8.5. KV Transfer Mechanisms

#### 8.5.1. Shared Memory (Single Node)

Fastest method khi prefill và decode instances trên cùng node:

```python
connector_config = KVConnectorConfig(
    kv_connector_type="SharedMemoryConnector",
    kv_role="kv_producer",  # or "kv_consumer"
    kv_buffer_device="cpu",          # Use CPU shared memory
    kv_buffer_size=5e9,              # 5GB shared memory pool
)

# Transfer via shared memory:
# - Zero-copy (no data movement)
# - Latency: <1ms
# - Bandwidth: CPU memory bandwidth (~100 GB/s)
```

#### 8.5.2. RDMA (Multi-Node)

Cho distributed deployments:

```python
connector_config = KVConnectorConfig(
    kv_connector_type="RDMAConnector",
    kv_role="kv_producer",
    kv_buffer_size=2e9,
    # RDMA-specific settings
    rdma_device="mlx5_0",            # InfiniBand device
    rdma_gid_index=0,
)

# Transfer via RDMA:
# - Direct GPU-to-GPU transfer
# - Latency: ~5-10ms
# - Bandwidth: 100-200 GB/s (InfiniBand)
```

#### 8.5.3. TCP/IP (Cross-Region)

Slowest nhưng linh hoạt nhất:

```python
connector_config = KVConnectorConfig(
    kv_connector_type="TCPConnector",
    kv_role="kv_consumer",
    kv_connector_address="192.168.1.100:8000",
    kv_buffer_size=1e9,
    # Compression to reduce bandwidth
    enable_compression=True,
)

# Transfer via TCP/IP:
# - Latency: 10-100ms (network dependent)
# - Bandwidth: 1-10 GB/s (network dependent)
# - Compression: ~3x reduction in transfer size
```

### 8.6. Advanced Techniques: Intra-GPU Disaggregation

#### Nexus System (Research: 2025)

Thay vì dùng 2 GPUs riêng biệt, **Nexus** thực hiện disaggregation **trong cùng GPU**:

```
Traditional Disaggregation:
GPU 0: [PPPPPPPPPPPP] (Prefill only)
GPU 1: [DDDDDDDDDDDD] (Decode only)
→ Resource waste (GPU 0 idle during decode-heavy periods)

Intra-GPU Disaggregation (Nexus):
GPU 0: [PPPP][DDD][PPP][DDDD]... (Dynamic partitioning)
       ↑ 60% SMs   ↑ 40% SMs
→ Same GPU handles both, but isolated

SM Allocation Example:
Total SMs: 132 (A100)
- Prefill: 80 SMs (when compute-heavy)
- Decode:  52 SMs (when latency-critical)
- Dynamically adjusted every batch
```

#### Cơ Chế SM Partitioning

```python
class IntraGPUDisaggregation:
    def __init__(self, total_sms=132):
        self.total_sms = total_sms
        self.prefill_sms = 0
        self.decode_sms = 0

    def partition_sms(self, workload_state):
        """
        Dynamic SM allocation based on workload

        Cost Model:
        - Prefill latency = f(num_sms, batch_size, seq_len)
        - Decode latency = g(num_sms, num_seqs, kv_cache_size)

        Optimize:
        minimize: max(prefill_latency, decode_latency)
        subject to: prefill_sms + decode_sms <= total_sms
        """
        # Analytical cost model (simplified)
        prefill_tokens = workload_state.prefill_tokens
        decode_seqs = workload_state.decode_seqs

        # Compute saturation thresholds
        prefill_sat = compute_saturation_point(prefill_tokens)
        decode_sat = compute_saturation_point(decode_seqs)

        # Allocate SMs proportionally, with hysteresis
        if prefill_tokens > threshold:
            self.prefill_sms = min(prefill_sat, self.total_sms * 0.7)
        else:
            self.prefill_sms = min(prefill_sat, self.total_sms * 0.3)

        self.decode_sms = self.total_sms - self.prefill_sms

        # Hysteresis: only change if delta > threshold
        if abs(self.prefill_sms - self.prev_prefill_sms) < delta:
            return  # No change to avoid thrashing

        # Apply new partition
        cuda.set_sm_partition(
            prefill_stream, self.prefill_sms,
            decode_stream, self.decode_sms
        )
```

#### Benefits của Intra-GPU Disaggregation

```
Benchmark: Nexus vs vLLM (LLaMA-2 70B, single A100):

Metric                    | vLLM Monolithic | Nexus Intra-GPU | Improvement
--------------------------|-----------------|-----------------|-------------
TTFT (median)             | 800ms           | 400ms           | 2x faster
ITL (median)              | 25ms            | 20ms            | 1.25x faster
ITL (P99)                 | 250ms           | 25ms            | 10x better!
TBT (time-between-tokens) | 30ms            | 12ms            | 2.5x faster
Throughput (tok/s)        | 1200            | 2640            | 2.2x higher

Key Benefit: Consistent tail latency (P99 ≈ median)
```

### 8.7. So Sánh Các Approaches

| Approach | Architecture | TTFT | ITL (P99) | Throughput | Cost | Complexity |
|----------|--------------|------|-----------|------------|------|------------|
| **Monolithic** | 1 instance, mixed workload | Medium | High (10x variance) | Medium | Low | Low |
| **Chunked Prefill** | 1 instance, chunked scheduling | Medium | Medium (3x variance) | High | Low | Medium |
| **Engine-level Disaggregation** | 2 instances, 2 GPUs | Low | Low (stable) | High | High (2x GPUs) | High |
| **Intra-GPU Disaggregation** | 1 instance, dynamic SM partition | Low | Very Low | Very High | Low | Very High |

### 8.8. Khi Nào Dùng Disaggregation?

#### ✅ Nên dùng khi:

1. **Tail Latency Critical:**
   ```
   Use Case: Chat applications, real-time assistants
   Requirement: P99 ITL < 50ms
   Solution: Disaggregated decode instance
   ```

2. **Mixed Workload:**
   ```
   Use Case: Platform serving both long prompts (RAG) and short prompts (chat)
   Prefill: 1000-10,000 tokens (RAG documents)
   Decode: 100-500 tokens (responses)
   Solution: Separate prefill and decode instances
   ```

3. **Different Optimization Goals:**
   ```
   Prefill Goal: Maximize throughput (batch processing)
   Decode Goal: Minimize latency (interactive)
   Solution: Tune each instance independently
   ```

#### ❌ Không nên dùng khi:

1. **Uniform Workload:**
   ```
   All requests: short prompts (~100 tokens) + short outputs (~50 tokens)
   → Monolithic instance with continuous batching is sufficient
   ```

2. **Cost-Sensitive:**
   ```
   Engine-level disaggregation doubles GPU cost
   → Use chunked prefill or intra-GPU disaggregation instead
   ```

3. **Single-Request Latency:**
   ```
   Only 1 request at a time
   → No batching benefits, disaggregation adds overhead
   ```

### 8.9. Implementation Guide

#### Setup: 2-Instance Disaggregation

```python
# === File: prefill_server.py ===
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer.kv_connector import KVConnectorConfig

def start_prefill_instance():
    config = KVConnectorConfig(
        kv_role="kv_producer",
        kv_connector_type="SharedMemoryConnector",
        kv_buffer_size=5e9,
    )

    llm = LLM(
        model="meta-llama/Llama-2-13b-hf",
        tensor_parallel_size=2,
        max_num_batched_tokens=8192,
        kv_connector_config=config,
    )

    # Prefill-only loop
    while True:
        request = receive_request()

        # Only prefill (max_tokens=0)
        output = llm.generate(
            prompts=[request.prompt],
            sampling_params=SamplingParams(max_tokens=0),
        )

        # KV cache automatically transferred via connector
        print(f"Prefilled request {request.id}, KV cache sent to decode instance")

# === File: decode_server.py ===
def start_decode_instance():
    config = KVConnectorConfig(
        kv_role="kv_consumer",
        kv_connector_type="SharedMemoryConnector",
        kv_buffer_size=5e9,
    )

    llm = LLM(
        model="meta-llama/Llama-2-13b-hf",
        tensor_parallel_size=1,          # Less TP needed
        max_num_seqs=128,
        kv_connector_config=config,
        enable_prefix_caching=True,
    )

    # Decode-only loop
    while True:
        # Receive KV cache from prefill instance (automatic via connector)
        request = receive_kv_cache()

        # Generate using prefilled KV cache
        outputs = llm.generate(
            prompts=[request.prompt],     # Metadata
            sampling_params=SamplingParams(
                max_tokens=request.max_tokens,
                temperature=0.7,
            ),
        )

        # Stream tokens back
        for output in outputs:
            stream_token(output.token)

# Start both
if __name__ == "__main__":
    import multiprocessing
    p1 = multiprocessing.Process(target=start_prefill_instance)
    p2 = multiprocessing.Process(target=start_decode_instance)
    p1.start()
    p2.start()
```

### 8.10. Monitoring và Debugging

#### Key Metrics

```python
from vllm import stats

# Prefill Instance Metrics
prefill_metrics = {
    "avg_prefill_time_ms": stats.get_avg_prefill_time(),
    "prefill_throughput_tokens_per_sec": stats.get_prefill_throughput(),
    "kv_cache_transfer_bandwidth_gbps": stats.get_kv_transfer_bandwidth(),
    "gpu_utilization": stats.get_gpu_utilization(),
}

# Decode Instance Metrics
decode_metrics = {
    "avg_itl_ms": stats.get_avg_inter_token_latency(),
    "p99_itl_ms": stats.get_p99_inter_token_latency(),
    "decode_throughput_tokens_per_sec": stats.get_decode_throughput(),
    "num_running_seqs": stats.get_num_running_sequences(),
}

# Alert if P99 ITL too high
if decode_metrics["p99_itl_ms"] > 50:
    print("WARNING: High tail latency detected!")
    # Scale decode instance or investigate bottleneck
```

### 8.11. Meta Production Implementation Details (2025)

Meta đã triển khai vLLM disaggregation trong production và chia sẻ nhiều optimizations quan trọng.

#### Key Optimizations từ Meta

**1. Larger Block Size:**

```python
# vLLM Default
block_size = 16  # tokens per block

# Meta Production
block_size = 128  # or 256

# Rationale:
# - Smaller blocks → nhiều small kernel launches
# - KV transfer overhead cao với small blocks
# - Larger blocks: ít transfers, better throughput

# Trade-off:
# - Larger blocks: higher memory waste (last block)
# - Smaller blocks: more flexible memory utilization
# - For disagg: larger is better (fewer transfers)
```

**2. Asynchronous KV Loading:**

```python
class AsyncKVLoader:
    """
    Meta's async KV loading: overlap KV load với decode step

    Standard Flow:
    1. Wait for KV transfer from prefill instance
    2. Load KV to GPU
    3. Start decode

    Async Flow:
    1. Start KV transfer (background)
    2. While transferring: run decode for OTHER requests
    3. When transfer complete: add new request to decode batch

    Benefit: Hide KV transfer latency behind decode computation
    """

    def __init__(self, connector):
        self.connector = connector
        self.pending_transfers = {}  # request_id → future

    async def async_load_kv(self, request_id, kv_cache):
        """Start async KV loading"""
        future = asyncio.create_task(
            self.connector.recv_kv_cache_async(request_id, kv_cache)
        )
        self.pending_transfers[request_id] = future

    def check_ready(self):
        """Check which requests have completed KV transfer"""
        ready = []
        for req_id, future in list(self.pending_transfers.items()):
            if future.done():
                kv_cache = future.result()
                ready.append((req_id, kv_cache))
                del self.pending_transfers[req_id]
        return ready

    def run_decode_with_overlap(self, decode_batch, new_requests):
        """
        Run decode while loading KV for new requests

        Timeline:
        t=0: Start KV load for new_requests (async)
        t=0: Run decode for decode_batch
        t=T: Decode complete
        t=T: Check if KV load complete
        t=T: Add ready requests to next decode batch
        """
        # Start async loads
        for req in new_requests:
            self.async_load_kv(req.id, req.kv_cache_placeholder)

        # Run decode (overlapped with KV transfer)
        outputs = self.model.decode(decode_batch)

        # Check completed transfers
        ready_requests = self.check_ready()

        return outputs, ready_requests
```

**3. Connector Types cho Different Deployments:**

```python
# Meta's Connector Recommendations:

# Single-node (8 GPUs same machine)
connector_config = {
    'type': 'SharedMemoryConnector',
    'buffer_size': 10 * 1024**3,  # 10GB
    'use_cuda_ipc': True,  # GPU-GPU direct
}
# Latency: <1ms, Bandwidth: ~300 GB/s

# Multi-node (InfiniBand)
connector_config = {
    'type': 'NixlConnector',  # NVIDIA NIXL
    'async_mode': True,
    'rdma_device': 'mlx5_0',
}
# Latency: 5-10ms, Bandwidth: 100-200 GB/s

# Cross-datacenter
connector_config = {
    'type': 'TCPConnector',
    'compression': 'zstd',  # ~3x compression
    'buffer_size': 1 * 1024**3,
}
# Latency: 50-100ms (acceptable with async loading)
```

**4. Dynamic Prefill/Decode Ratio:**

```python
class MetaLoadBalancer:
    """
    Meta's dynamic resource allocation

    Problem:
    - Prefill load varies (burst vs steady)
    - Fixed P:D ratio wastes resources

    Solution:
    - Monitor queue lengths
    - Dynamically scale prefill/decode instances
    - Use Kubernetes HPA or custom autoscaler
    """

    def compute_optimal_ratio(self, metrics):
        """
        Compute optimal prefill:decode GPU ratio

        Inputs:
        - prefill_queue_length: pending prefill requests
        - decode_throughput: current decode tokens/sec
        - avg_prompt_length: average prompt tokens
        - avg_output_length: average output tokens
        """
        # Time to prefill one request
        prefill_time = metrics.avg_prompt_length / metrics.prefill_throughput

        # Time to decode one request
        decode_time = metrics.avg_output_length / metrics.decode_throughput

        # Optimal ratio = decode_time / prefill_time
        # If decode takes longer → need more decode GPUs
        optimal_ratio = decode_time / prefill_time

        # Example:
        # avg_prompt = 1000, prefill_throughput = 10000 tok/s → prefill_time = 0.1s
        # avg_output = 500, decode_throughput = 500 tok/s → decode_time = 1s
        # ratio = 1 / 0.1 = 10 → need 10x more decode GPUs

        return optimal_ratio
```

#### Production Results (Meta, 2025)

```
Meta's Internal vLLM Disaggregation Deployment:

Workload: 1M requests/day, mixed prompt lengths (100-5000 tokens)

Before (Monolithic):
- TTFT P50: 600ms
- TTFT P99: 3000ms
- ITL P50:  25ms
- ITL P99:  180ms
- Throughput: 50K req/hour
- Cost: 100 A100 GPUs

After (Disaggregation):
- TTFT P50: 300ms (2x better)
- TTFT P99: 800ms (3.75x better)
- ITL P50:  20ms (1.25x better)
- ITL P99:  30ms (6x better!!!)
- Throughput: 80K req/hour (1.6x)
- Cost: 120 A100 GPUs (20% more, but worth it for latency)

Key Win: Consistent user experience (P99 ≈ P50)
```

### 8.12. Trade-offs và Considerations

#### Pros:

✅ **Tail Latency:** P99 ITL dramatically reduced (6-10x)
✅ **Tunable:** Independently optimize TTFT và ITL
✅ **Flexible Parallelism:** Different TP/PP for prefill vs decode
✅ **Better User Experience:** Consistent latency, no spikes

#### Cons:

❌ **Cost:** 2x GPUs for engine-level disaggregation
❌ **Complexity:** More moving parts, harder to debug
❌ **Transfer Overhead:** KV cache transfer adds 5-50ms latency
❌ **No Throughput Gain:** Same throughput as monolithic (or worse if transfer slow)

#### Recommendation:

```python
# Decision tree

if tail_latency_critical:  # Chat, real-time apps
    if budget_allows:
        use_disaggregation = True
        method = "engine_level"  # 2 instances, 2 GPU clusters
    else:
        use_disaggregation = True
        method = "intra_gpu"     # Nexus-style (if available)
else:
    use_disaggregation = False
    method = "chunked_prefill"   # Good enough for most cases
```

---

## Kết Luận và So Sánh

### Tổng Quan Các Tính Năng

| Tính Năng | Vấn Đề Giải Quyết | Cải Thiện | Trade-offs |
|-----------|-------------------|-----------|------------|
| **Paged Attention** | Memory fragmentation, static allocation | Memory efficiency 5-8x | Thêm indirection overhead (nhỏ) |
| **Continuous Batching** | Static batching, low GPU util, high latency | Throughput 2-3x, latency 10x thấp hơn | Phức tạp scheduler |
| **Distributed Inference** | Model không fit 1 GPU | Hỗ trợ models lớn (100B+) | Communication overhead, setup phức tạp |
| **Chunked Prefill** | Prefill/decode imbalance | Decode latency ổn định, throughput +50% | Prefill chậm hơn một chút |
| **Prefix Caching** | Redundant computation | TTFT 3-5x nhanh hơn cho requests có chung prefix | Memory overhead nhỏ |
| **Quantization** | High memory usage, expensive hardware | Memory 2-4x reduction, cost 50-75% lower | Accuracy degradation 2-8% |
| **Context Parallelism** | Long context (100K-1M tokens), KV cache bottleneck | Memory 3x reduction, latency 2-3x faster cho long context | Communication overhead, chỉ hiệu quả với long context |
| **Disaggregated Prefill/Decode** | Phase interference, tail ITL spikes | P99 ITL 6-10x better, consistent latency | 2x GPU cost (engine-level), transfer overhead, complexity |

### Khi Nào Dùng vLLM?

**Nên dùng vLLM khi:**

1. ✅ Serving LLMs ở production scale (high throughput, low latency)
2. ✅ Models lớn cần distributed inference (70B+)
3. ✅ Có nhiều requests với common prefixes (chatbots, RAG)
4. ✅ Cần optimize GPU utilization và reduce costs
5. ✅ Cần OpenAI-compatible API
6. ✅ Muốn deploy models lớn trên hardware hạn chế (với quantization)

**Không nên dùng vLLM khi:**

1. ❌ Training models (vLLM chỉ cho inference)
2. ❌ Single request, không cần batching
3. ❌ Model rất nhỏ (<1B parameters), overhead không đáng kể
4. ❌ Cần nhiều custom modifications (vLLM optimize cho standard transformers)

### So Sánh với Các Framework Khác

#### vLLM vs HuggingFace Transformers

| Aspect | HuggingFace | vLLM |
|--------|-------------|------|
| Ease of Use | ⭐⭐⭐⭐⭐ (very simple) | ⭐⭐⭐⭐ (need setup) |
| Throughput | ⭐⭐ (low) | ⭐⭐⭐⭐⭐ (very high) |
| Latency | ⭐⭐ (high) | ⭐⭐⭐⭐⭐ (very low) |
| Memory Efficiency | ⭐⭐ (poor) | ⭐⭐⭐⭐⭐ (excellent) |
| Distributed Support | ⭐⭐⭐ (basic) | ⭐⭐⭐⭐⭐ (advanced) |
| Model Support | ⭐⭐⭐⭐⭐ (all models) | ⭐⭐⭐⭐ (most popular) |

#### vLLM vs TensorRT-LLM

| Aspect | TensorRT-LLM | vLLM |
|--------|--------------|------|
| Performance | ⭐⭐⭐⭐⭐ (fastest) | ⭐⭐⭐⭐ (very fast) |
| Ease of Use | ⭐⭐ (complex setup) | ⭐⭐⭐⭐ (easier) |
| Flexibility | ⭐⭐ (more rigid) | ⭐⭐⭐⭐ (flexible) |
| Hardware Support | ⭐⭐⭐ (NVIDIA only) | ⭐⭐⭐⭐ (NVIDIA, AMD, CPU) |
| Python-native | ⭐⭐ (C++ core) | ⭐⭐⭐⭐⭐ (Python-friendly) |

#### vLLM vs Text Generation Inference (TGI)

| Aspect | TGI | vLLM |
|--------|-----|------|
| Throughput | ⭐⭐⭐⭐ (high) | ⭐⭐⭐⭐⭐ (higher) |
| Latency | ⭐⭐⭐⭐ (low) | ⭐⭐⭐⭐ (comparable) |
| Memory Efficiency | ⭐⭐⭐ (good) | ⭐⭐⭐⭐⭐ (better) |
| Prefix Caching | ⭐⭐⭐ (basic) | ⭐⭐⭐⭐⭐ (advanced) |
| Community Support | ⭐⭐⭐⭐ (HuggingFace) | ⭐⭐⭐⭐ (growing) |

### Best Practices

#### Deployment Configuration

```python
# Recommended vLLM configuration for production

from vllm import LLM, SamplingParams

# 1. Initialize LLM with optimal settings
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",

    # Tensor Parallelism (TP)
    tensor_parallel_size=4,  # Use 4 GPUs for TP

    # Memory management
    gpu_memory_utilization=0.95,  # Use 95% of GPU memory
    block_size=16,  # Paged attention block size

    # Prefix caching
    enable_prefix_caching=True,

    # Chunked prefill
    max_num_batched_tokens=8192,  # Max tokens in a batch
    max_num_seqs=256,  # Max sequences in continuous batch

    # Performance
    dtype="float16",  # Use FP16 for speed
    quantization="awq",  # Optional: quantization for memory savings
)

# 2. Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

# 3. Generate
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)
```

#### Monitoring và Tuning

```python
# Key metrics to monitor

from vllm import stats

# 1. GPU utilization
gpu_util = stats.get_gpu_utilization()
# Target: >90%

# 2. Cache hit rate (prefix caching)
cache_hit_rate = stats.get_cache_hit_rate()
# Target: >50% for workloads with common prefixes

# 3. Decode latency (inter-token latency)
decode_latency = stats.get_decode_latency_ms()
# Target: <30ms for 70B model on A100

# 4. Time to First Token (TTFT)
ttft = stats.get_ttft_ms()
# Target: <500ms for long prompts

# 5. Throughput
throughput = stats.get_throughput_tokens_per_sec()
# Target: Model-dependent, compare with baselines
```

### Future Directions

vLLM đang phát triển các tính năng:

1. **Multi-node Inference**: Pipeline Parallelism across nodes
2. **Speculative Decoding**: Speed up generation with draft models
3. **Advanced Quantization**: QuIP, GGUF format support, W4A4
4. **Dynamic Batching Improvements**: Even better scheduling algorithms
5. **More Hardware Support**: AMD GPUs, Intel GPUs, TPUs
6. **Vision-Language Models**: Improved multimodal support

---

## References

### Core vLLM

1. **vLLM Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" - [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) (2023)
2. **vLLM GitHub**: https://github.com/vllm-project/vllm
3. **vLLM Documentation**: https://docs.vllm.ai/
4. **vLLM Blog**: https://blog.vllm.ai/
5. **FlashAttention**: Efficient attention implementation (used in vLLM)

### vLLM V1 Architecture

6. **vLLM V1 Alpha Release**: [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (January 2025)
7. **Inside vLLM: Anatomy of a High-Throughput LLM Inference System**: [vLLM Blog Technical Deep Dive](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
8. **vLLM V1 Engine Architecture RFC**: [GitHub Issue #8779](https://github.com/vllm-project/vllm/issues/8779)
9. **vLLM 2024 Retrospective and 2025 Vision**: https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html

### Automatic Prefix Caching

10. **vLLM Automatic Prefix Caching**: https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html
11. **SGLang RadixAttention**: [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)
12. **Prefix Caching Comparison**: [SGLang vs vLLM - Token-Level Radix Tree vs Block-Level Hashing](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1)

### Quantization

13. **LLM Compressor**: https://github.com/vllm-project/llm-compressor
14. **LLM Compressor Docs**: https://docs.vllm.ai/projects/llm-compressor
15. **GPTQ Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2023)
16. **AWQ Paper**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
17. **SmoothQuant Paper**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2023)
18. **SparseGPT Paper**: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot" (2023)

### Distributed Computing

19. **Megatron-LM**: NVIDIA's framework for distributed training (inspiration for TP/PP)
20. **FasterTransformer**: NVIDIA's optimized inference library

### Context Parallelism / Sequence Parallelism

21. **Ring Attention Paper**: "Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al., 2023)
22. **DeepSpeed Ulysses**: [Ultra-Long Sequence Parallelism: Ulysses + Ring-Attention Technical Principles](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention)
23. **Snowflake Arctic Ulysses**: [Ulysses: Unlocking Low-Latency, High-Throughput Inference for Long-Context LLMs](https://www.snowflake.com/en/engineering-blog/ulysses-low-latency-llm-inference/)
24. **Long Context Attention (USP)**: https://github.com/feifeibear/long-context-attention
25. **vLLM Context Parallelism RFC**: [GitHub Issue #22693](https://github.com/vllm-project/vllm/issues/22693)
26. **vLLM CP with Ring Attention RFC**: [GitHub Issue #26133](https://github.com/vllm-project/vllm/issues/26133)
27. **vLLM Context Parallel Deployment**: https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/

### Disaggregated Inference

28. **Disaggregated Prefill vLLM Docs**: https://docs.vllm.ai/en/latest/features/disagg_prefill.html
29. **PyTorch Blog - Disaggregated Inference at Scale**: https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/
30. **DistServe Paper**: "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" (OSDI'24) - https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
31. **P/D-Serve Paper**: "P/D-Serve: Serving Disaggregated Large Language Model at Scale" - [arXiv:2408.08147](https://arxiv.org/html/2408.08147v1)
32. **Nexus Paper**: "Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving" (arXiv 2507.06608, 2025)
33. **vLLM Prefill-only optimizations RFC**: [GitHub Issue #19038](https://github.com/vllm-project/vllm/issues/19038)

### Continuous Batching & Scheduling

34. **Orca Paper**: "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI'22)
35. **vLLM Chunked Prefill**: https://docs.vllm.ai/en/latest/configuration/optimization/
36. **BatchLLM Paper**: "BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching" - [arXiv:2412.03594](https://arxiv.org/html/2412.03594v1)

---

**Tài liệu này cung cấp overview chi tiết về các tính năng chính của vLLM. Để hiểu sâu hơn, khuyến nghị:**

1. **Đọc source code:**
   - vLLM: https://github.com/vllm-project/vllm
   - LLM Compressor: https://github.com/vllm-project/llm-compressor

2. **Thực hành:**
   - Chạy thử nghiệm với các models khác nhau
   - Thử các phương pháp quantization khác nhau (GPTQ, AWQ, SmoothQuant)
   - Benchmark trên hardware thực tế

3. **Tối ưu:**
   - Profile performance với vLLM stats
   - Tune hyperparameters (batch size, chunked prefill, quantization settings)
   - Test với workload thực tế

4. **Cộng đồng:**
   - Tham gia vLLM GitHub Discussions
   - Join vLLM Slack (#sig-quantization, #llm-compressor)
   - Theo dõi updates và best practices

**Chúc bạn thành công trong việc deploy LLMs với vLLM! 🚀**
