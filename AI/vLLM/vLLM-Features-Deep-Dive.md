# vLLM - Tìm Hiểu Các Tính Năng Chính

## Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Paged Attention](#1-paged-attention)
3. [Continuous Batching](#2-continuous-batching)
4. [Distributed Inference](#3-distributed-inference)
5. [Chunked Prefill Scheduling](#4-chunked-prefill-scheduling)
6. [Automatic Prefix Caching](#5-automatic-prefix-caching)
7. [Model Quantization với LLM Compressor](#6-model-quantization-với-llm-compressor)
8. [Kết Luận và So Sánh](#kết-luận-và-so-sánh)

---

## Giới Thiệu

**vLLM** (Virtual Large Language Model) là một thư viện mã nguồn mở được thiết kế để tối ưu hóa việc serving và inference của các Large Language Models (LLMs). vLLM được phát triển bởi nhóm nghiên cứu tại UC Berkeley và đạt được hiệu suất cao hơn đáng kể so với các framework truyền thống như HuggingFace Transformers hay FasterTransformer.

### Những Vấn Đề vLLM Giải Quyết

1. **Memory Inefficiency**: KV cache trong attention mechanism chiếm lượng lớn GPU memory và thường bị fragmentation
2. **Low GPU Utilization**: Các framework truyền thống không tối ưu được GPU throughput
3. **Batching Inefficiency**: Static batching yêu cầu chờ toàn bộ batch hoàn thành, dẫn đến latency cao
4. **Distributed Complexity**: Việc deploy models lớn trên nhiều GPUs phức tạp
5. **Redundant Computation**: Các requests có prefix giống nhau vẫn phải tính toán lại từ đầu

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

1. **vLLM Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
2. **vLLM GitHub**: https://github.com/vllm-project/vllm
3. **vLLM Documentation**: https://vllm.readthedocs.io/
4. **FlashAttention**: Efficient attention implementation (used in vLLM)

### Quantization

5. **LLM Compressor**: https://github.com/vllm-project/llm-compressor
6. **LLM Compressor Docs**: https://docs.vllm.ai/projects/llm-compressor
7. **GPTQ Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2023)
8. **AWQ Paper**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
9. **SmoothQuant Paper**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2023)
10. **SparseGPT Paper**: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot" (2023)

### Distributed Computing

11. **Megatron-LM**: NVIDIA's framework for distributed training (inspiration for TP/PP)
12. **FasterTransformer**: NVIDIA's optimized inference library

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
