# Hướng dẫn tính toán VRAM GPU cho LLM Inference

## Tổng quan

Khi triển khai các mô hình ngôn ngữ lớn (LLM) cho inference, việc tính toán chính xác VRAM cần thiết là yếu tố quan trọng để đảm bảo hiệu suất tối ưu và tránh lỗi out-of-memory (OOM). Tài liệu này cung cấp công thức và hướng dẫn thực tế để ước lượng bộ nhớ VRAM.

## Các thành phần tiêu thụ VRAM

Tổng VRAM cần thiết cho LLM inference bao gồm 3 thành phần chính:

```
Total VRAM = Model Weights + KV Cache + Activations + Overhead
```

### 1. Model Weights (Trọng số mô hình)

Đây là thành phần chiếm nhiều bộ nhớ nhất, phụ thuộc vào số lượng tham số và định dạng precision.

#### Công thức tính:

```
Model Weight Memory = N × bytes_per_parameter
```

Trong đó:
- `N` = Số lượng parameters của mô hình
- `bytes_per_parameter` = Số bytes mỗi parameter (phụ thuộc precision)

#### Bảng precision và memory footprint:

| Precision Format | Bytes per Parameter | Ví dụ (7B params) |
|-----------------|---------------------|-------------------|
| FP32 (32-bit)   | 4 bytes             | 28 GB             |
| FP16 (16-bit)   | 2 bytes             | 14 GB             |
| BF16 (16-bit)   | 2 bytes             | 14 GB             |
| FP8 (8-bit)     | 1 byte              | 7 GB              |
| INT8 (8-bit)    | 1 byte              | 7 GB              |
| FP4 (4-bit)     | 0.5 bytes           | 3.5 GB            |
| INT4 (4-bit)    | 0.5 bytes           | 3.5 GB            |

#### Ví dụ tính toán:

**Llama 2 7B model:**
```
- FP32: 7B × 4 bytes = 28 GB
- FP16: 7B × 2 bytes = 14 GB
- BF16: 7B × 2 bytes = 14 GB
- FP8:  7B × 1 byte = 7 GB
- INT8: 7B × 1 byte = 7 GB
- FP4:  7B × 0.5 bytes = 3.5 GB
- INT4: 7B × 0.5 bytes = 3.5 GB
```

**Llama 2 70B model:**
```
- FP32: 70B × 4 bytes = 280 GB
- FP16: 70B × 2 bytes = 140 GB
- BF16: 70B × 2 bytes = 140 GB
- FP8:  70B × 1 byte = 70 GB
- INT8: 70B × 1 byte = 70 GB
- FP4:  70B × 0.5 bytes = 35 GB
- INT4: 70B × 0.5 bytes = 35 GB
```

### 2. KV Cache (Key-Value Cache)

KV Cache lưu trữ key và value tensors từ attention mechanism cho các token đã xử lý, giúp tăng tốc inference bằng cách tránh tính toán lại.

#### Công thức tính:

```
KV Cache Memory = 2 × num_layers × batch_size × sequence_length × hidden_dim × bytes_per_element
```

Trong đó:
- `2` = Hệ số cho cả keys và values
- `num_layers` = Số lượng transformer layers
- `batch_size` = Số lượng sequences xử lý đồng thời
- `sequence_length` = Độ dài sequence (input + output tokens)
- `hidden_dim` = Kích thước hidden dimension
- `bytes_per_element` = Bytes per element (thường là 2 cho FP16)

#### Ví dụ tính toán:

**Llama 2 7B (32 layers, hidden_dim = 4096, FP16):**

```
Batch size = 1, Sequence length = 2048:
KV Cache = 2 × 32 × 1 × 2048 × 4096 × 2 bytes
         = 1,073,741,824 bytes
         ≈ 1 GB

Batch size = 4, Sequence length = 2048:
KV Cache = 2 × 32 × 4 × 2048 × 4096 × 2 bytes
         ≈ 4 GB

Batch size = 1, Sequence length = 4096:
KV Cache = 2 × 32 × 1 × 4096 × 4096 × 2 bytes
         ≈ 2 GB
```

**Quan sát:**
- KV Cache tăng tuyến tính với batch size
- KV Cache tăng tuyến tính với sequence length
- Đối với batch inference, KV Cache có thể chiếm VRAM đáng kể

### 3. Activations (Kích hoạt)

Activations là các tensor trung gian được tạo ra trong quá trình forward pass. Kích thước phụ thuộc vào kiến trúc mô hình và batch size.

#### Ước lượng:

```
Activation Memory ≈ batch_size × sequence_length × hidden_dim × num_layers × bytes_per_element × factor
```

- `factor` thường dao động từ 4-8 tùy thuộc vào kiến trúc
- Activations chiếm ít bộ nhớ hơn nhiều so với weights và KV cache trong inference
- Thường chiếm khoảng 10-20% tổng VRAM cho batch size nhỏ

### 4. Overhead

Thêm khoảng 10-20% cho:
- Framework overhead (PyTorch, vLLM, etc.)
- Workspace memory
- Fragmentation
- CUDA context

## Công thức tổng quát

### Công thức đầy đủ:

```
Total VRAM = (N × precision_bytes) +
             (2 × L × B × S × H × precision_bytes) +
             (B × S × H × L × precision_bytes × act_factor) +
             overhead
```

Trong đó:
- `N` = Number of parameters
- `L` = Number of layers
- `B` = Batch size
- `S` = Sequence length
- `H` = Hidden dimension
- `precision_bytes` = Bytes per parameter (2 for FP16, 1 for INT8, etc.)
- `act_factor` = Activation memory factor (4-8)
- `overhead` = 10-20% of total

### Công thức đơn giản hóa (inference cơ bản):

```
Total VRAM ≈ Model Weights + KV Cache × 1.2
```

Hệ số 1.2 bù đắp cho activations và overhead.

## Ví dụ thực tế

### Case 1: Llama 2 7B, FP16, Single Request

**Thông số:**
- Parameters: 7B
- Precision: FP16 (2 bytes)
- Layers: 32
- Hidden dim: 4096
- Batch size: 1
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 7B × 2 = 14 GB

KV Cache = 2 × 32 × 1 × 2048 × 4096 × 2
         ≈ 1 GB

Activations ≈ 0.3 GB (ước lượng)

Overhead ≈ 15% × (14 + 1 + 0.3) = 2.3 GB

Total VRAM ≈ 14 + 1 + 0.3 + 2.3 = 17.6 GB
```

**Kết luận:** Cần GPU có ít nhất 20 GB VRAM (ví dụ: A100 40GB, RTX 4090 24GB)

### Case 2: Llama 2 7B, INT4, Batch Inference

**Thông số:**
- Parameters: 7B
- Precision: INT4 (0.5 bytes)
- Layers: 32
- Hidden dim: 4096
- Batch size: 8
- Sequence length: 1024

**Tính toán:**

```
Model Weights = 7B × 0.5 = 3.5 GB

KV Cache = 2 × 32 × 8 × 1024 × 4096 × 2
         ≈ 4 GB

Activations ≈ 0.5 GB

Overhead ≈ 15% × (3.5 + 4 + 0.5) = 1.2 GB

Total VRAM ≈ 3.5 + 4 + 0.5 + 1.2 = 9.2 GB
```

**Kết luận:** Có thể chạy trên GPU 12 GB (RTX 3090, RTX 4070 Ti)

### Case 3: Llama 2 70B, INT8, Single Request

**Thông số:**
- Parameters: 70B
- Precision: INT8 (1 byte)
- Layers: 80
- Hidden dim: 8192
- Batch size: 1
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 70B × 1 = 70 GB

KV Cache = 2 × 80 × 1 × 2048 × 8192 × 2
         ≈ 5.2 GB

Activations ≈ 1 GB

Overhead ≈ 15% × (70 + 5.2 + 1) = 11.4 GB

Total VRAM ≈ 70 + 5.2 + 1 + 11.4 = 87.6 GB
```

**Kết luận:** Cần GPU có ít nhất 90 GB VRAM (A100 80GB không đủ, cần 2× A100 hoặc H100 80GB)

### Case 4: Llama 2 7B, FP8, Batch Inference

**Thông số:**
- Parameters: 7B
- Precision: FP8 (1 byte)
- Layers: 32
- Hidden dim: 4096
- Batch size: 16
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 7B × 1 = 7 GB

KV Cache = 2 × 32 × 16 × 2048 × 4096 × 2
         ≈ 16 GB

Activations ≈ 1 GB

Overhead ≈ 15% × (7 + 16 + 1) = 3.6 GB

Total VRAM ≈ 7 + 16 + 1 + 3.6 = 27.6 GB
```

**Kết luận:** Cần GPU có ít nhất 32 GB VRAM (A100 40GB, RTX A6000)

### Case 5: Llama 2 70B, FP4, Single Request

**Thông số:**
- Parameters: 70B
- Precision: FP4 (0.5 bytes)
- Layers: 80
- Hidden dim: 8192
- Batch size: 1
- Sequence length: 4096

**Tính toán:**

```
Model Weights = 70B × 0.5 = 35 GB

KV Cache = 2 × 80 × 1 × 4096 × 8192 × 2
         ≈ 10.4 GB

Activations ≈ 1.5 GB

Overhead ≈ 15% × (35 + 10.4 + 1.5) = 7 GB

Total VRAM ≈ 35 + 10.4 + 1.5 + 7 = 53.9 GB
```

**Kết luận:** Có thể chạy trên GPU A100 80GB hoặc H100 80GB

### Case 6: Llama 2 13B, FP8, Production Workload

**Thông số:**
- Parameters: 13B
- Precision: FP8 (1 byte)
- Layers: 40
- Hidden dim: 5120
- Batch size: 8
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 13B × 1 = 13 GB

KV Cache = 2 × 40 × 8 × 2048 × 5120 × 2
         ≈ 13.4 GB

Activations ≈ 1.2 GB

Overhead ≈ 15% × (13 + 13.4 + 1.2) = 4.1 GB

Total VRAM ≈ 13 + 13.4 + 1.2 + 4.1 = 31.7 GB
```

**Kết luận:** Cần GPU có ít nhất 40 GB VRAM (A100 40GB)

### Case 7: Qwen/Qwen3-32B, FP8, Batch Inference

**Thông số:**
- Parameters: 32B
- Precision: FP8 (1 byte)
- Layers: 64
- Hidden dim: 5120
- Batch size: 4
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 32B × 1 = 32 GB

KV Cache = 2 × 64 × 4 × 2048 × 5120 × 2
         ≈ 10.7 GB

Activations ≈ 1.5 GB

Overhead ≈ 15% × (32 + 10.7 + 1.5) = 6.6 GB

Total VRAM ≈ 32 + 10.7 + 1.5 + 6.6 = 50.8 GB
```

**Kết luận:** Cần GPU có ít nhất 64 GB VRAM (A100 80GB, H100 80GB)

### Case 8: google/gemma-3-27b-it, FP8, Single Request

**Thông số:**
- Parameters: 27B
- Precision: FP8 (1 byte)
- Layers: 46
- Hidden dim: 4608
- Batch size: 1
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 27B × 1 = 27 GB

KV Cache = 2 × 46 × 1 × 2048 × 4608 × 2
         ≈ 1.7 GB

Activations ≈ 0.8 GB

Overhead ≈ 15% × (27 + 1.7 + 0.8) = 4.4 GB

Total VRAM ≈ 27 + 1.7 + 0.8 + 4.4 = 33.9 GB
```

**Kết luận:** Cần GPU có ít nhất 40 GB VRAM (A100 40GB)

### Case 9: meta-llama/Llama-3.3-70B-Instruct, FP8, Production Workload

**Thông số:**
- Parameters: 70B
- Precision: FP8 (1 byte)
- Layers: 80
- Hidden dim: 8192
- Batch size: 8
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 70B × 1 = 70 GB

KV Cache = 2 × 80 × 8 × 2048 × 8192 × 2
         ≈ 41.9 GB

Activations ≈ 2.5 GB

Overhead ≈ 15% × (70 + 41.9 + 2.5) = 17.2 GB

Total VRAM ≈ 70 + 41.9 + 2.5 + 17.2 = 131.6 GB
```

**Kết luận:** Cần GPU có ít nhất 2× A100 80GB hoặc H200 141GB

### Case 10: Qwen/Qwen2.5-7B-Instruct, FP8, Single Request

**Thông số:**
- Parameters: 7B
- Precision: FP8 (1 byte)
- Layers: 28
- Hidden dim: 3584
- Batch size: 1
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 7B × 1 = 7 GB

KV Cache = 2 × 28 × 1 × 2048 × 3584 × 2
         ≈ 0.8 GB

Activations ≈ 0.4 GB

Overhead ≈ 15% × (7 + 0.8 + 0.4) = 1.2 GB

Total VRAM ≈ 7 + 0.8 + 0.4 + 1.2 = 9.4 GB
```

**Kết luận:** Cần GPU có ít nhất 12 GB VRAM (RTX 3090, RTX 4070 Ti, RTX 4080)

### Case 11: zai-org/GLM-4.5, FP8, Batch Inference

**Thông số:**
- Parameters: 9B
- Precision: FP8 (1 byte)
- Layers: 40
- Hidden dim: 4096
- Batch size: 4
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 9B × 1 = 9 GB

KV Cache = 2 × 40 × 4 × 2048 × 4096 × 2
         ≈ 5.4 GB

Activations ≈ 0.6 GB

Overhead ≈ 15% × (9 + 5.4 + 0.6) = 2.3 GB

Total VRAM ≈ 9 + 5.4 + 0.6 + 2.3 = 17.3 GB
```

**Kết luận:** Cần GPU có ít nhất 20 GB VRAM (RTX 4090, A10, A100 40GB)

### Case 12: Qwen/QwQ-32B, FP8, Production Workload

**Thông số:**
- Parameters: 32B
- Precision: FP8 (1 byte)
- Layers: 64
- Hidden dim: 5120
- Batch size: 8
- Sequence length: 2048

**Tính toán:**

```
Model Weights = 32B × 1 = 32 GB

KV Cache = 2 × 64 × 8 × 2048 × 5120 × 2
         ≈ 21.5 GB

Activations ≈ 2.0 GB

Overhead ≈ 15% × (32 + 21.5 + 2.0) = 8.3 GB

Total VRAM ≈ 32 + 21.5 + 2.0 + 8.3 = 63.8 GB
```

**Kết luận:** Cần GPU có ít nhất 80 GB VRAM (A100 80GB, H100 80GB)

### Case 13: intfloat/multilingual-e5-large (Embedding Model), FP32, Batch Inference

**Thông số:**
- Parameters: 560M
- Precision: FP32 (4 bytes)
- Layers: 24
- Hidden dim: 1024
- Batch size: 32
- Sequence length: 512

**Tính toán:**

```
Model Weights = 560M × 4 = 2.24 GB

KV Cache = 2 × 24 × 32 × 512 × 1024 × 4
         ≈ 3.2 GB

Activations ≈ 0.5 GB

Overhead ≈ 15% × (2.24 + 3.2 + 0.5) = 0.9 GB

Total VRAM ≈ 2.24 + 3.2 + 0.5 + 0.9 = 6.84 GB
```

**Kết luận:** Cần GPU có ít nhất 8 GB VRAM (RTX 3070, RTX 4060 Ti, T4)

**Note:** Embedding models thường có sequence length ngắn hơn và batch size lớn hơn để tối ưu throughput.

### Case 14: BAAI/bge-reranker-v2-m3 (Reranker Model), FP32, Batch Inference

**Thông số:**
- Parameters: 568M
- Precision: FP32 (4 bytes)
- Layers: 24
- Hidden dim: 1024
- Batch size: 64
- Sequence length: 512

**Tính toán:**

```
Model Weights = 568M × 4 = 2.27 GB

KV Cache = 2 × 24 × 64 × 512 × 1024 × 4
         ≈ 6.4 GB

Activations ≈ 0.8 GB

Overhead ≈ 15% × (2.27 + 6.4 + 0.8) = 1.4 GB

Total VRAM ≈ 2.27 + 6.4 + 0.8 + 1.4 = 10.87 GB
```

**Kết luận:** Cần GPU có ít nhất 12 GB VRAM (RTX 3090, RTX 4070 Ti)

**Note:** Reranker models thường process nhiều document pairs cùng lúc, yêu cầu batch size lớn.

### Case 15: Alibaba-NLP/gte-multilingual-base (Embedding Model), FP16, Production Workload

**Thông số:**
- Parameters: 305M
- Precision: FP16 (2 bytes)
- Layers: 12
- Hidden dim: 768
- Batch size: 128
- Sequence length: 512

**Tính toán:**

```
Model Weights = 305M × 2 = 0.61 GB

KV Cache = 2 × 12 × 128 × 512 × 768 × 2
         ≈ 2.4 GB

Activations ≈ 0.6 GB

Overhead ≈ 15% × (0.61 + 2.4 + 0.6) = 0.5 GB

Total VRAM ≈ 0.61 + 2.4 + 0.6 + 0.5 = 4.11 GB
```

**Kết luận:** Cần GPU có ít nhất 6 GB VRAM (RTX 3060, RTX 4060, T4)

**Note:** Embedding models nhỏ hơn cho phép batch size rất lớn, tối ưu cho throughput cao trong production.

### Case 16: openai/whisper-large-v3-turbo (Speech-to-Text Model), FP16, Batch Inference

**Thông số:**
- Parameters: 809M
- Precision: FP16 (2 bytes)
- Encoder Layers: 32
- Decoder Layers: 32
- Hidden dim: 1280
- Batch size: 8
- Audio sequence length: 3000 frames (~30s audio)

**Tính toán:**

```
Model Weights = 809M × 2 = 1.62 GB

# Whisper sử dụng encoder-decoder architecture
# Encoder KV Cache (for audio features)
Encoder KV Cache = 2 × 32 × 8 × 3000 × 1280 × 2
                 ≈ 3.93 GB

# Decoder KV Cache (for text generation, avg 150 tokens)
Decoder KV Cache = 2 × 32 × 8 × 150 × 1280 × 2
                 ≈ 0.20 GB

# Cross-attention cache
Cross Attention = 32 × 8 × 150 × 3000 × 2
                ≈ 0.46 GB

Activations ≈ 1.2 GB

Overhead ≈ 15% × (1.62 + 3.93 + 0.20 + 0.46 + 1.2) = 1.1 GB

Total VRAM ≈ 1.62 + 3.93 + 0.20 + 0.46 + 1.2 + 1.1 = 8.51 GB
```

**Kết luận:** Cần GPU có ít nhất 12 GB VRAM (RTX 3090, RTX 4070 Ti, RTX 4080)

**Note:**
- Whisper models có encoder-decoder architecture, VRAM usage phức tạp hơn decoder-only LLMs
- Audio sequence length (~3000 frames cho 30s audio) tạo ra encoder KV cache lớn
- Batch inference với audio cần VRAM đáng kể do input sequence dài
- Đối với real-time streaming, batch_size=1 chỉ cần ~3-4 GB VRAM

## Đặc điểm VRAM cho Embedding và Reranker Models

### So sánh với LLM Models

Embedding và Reranker models có đặc điểm khác biệt về sử dụng VRAM so với LLM:

| Đặc điểm | LLM Models | Embedding/Reranker Models |
|----------|------------|---------------------------|
| Model size | 7B - 180B params | 100M - 600M params |
| Sequence length | 2048 - 8192 tokens | 512 - 1024 tokens |
| Batch size | 1 - 16 | 32 - 256 |
| VRAM dominance | Model Weights | KV Cache (do batch size lớn) |
| Precision preference | FP8/INT4 để giảm size | FP32/FP16 để đảm bảo chất lượng embedding |
| Use case | Text generation | Semantic search, retrieval, ranking |

### Best Practices cho Embedding/Reranker Models

**1. Precision Selection:**
- **FP32**: Dùng cho development và khi cần độ chính xác embedding cao nhất
- **FP16**: Tối ưu cho production, giảm 50% VRAM với minimal quality loss
- **INT8**: Chỉ dùng khi VRAM rất hạn chế, cần test kỹ quality impact

**2. Batch Size Tuning:**
```python
# Ví dụ: multilingual-e5-large với batch size optimization
# GPU: RTX 4090 24GB

# Conservative: batch_size = 32, ~7 GB VRAM
# Balanced: batch_size = 64, ~11 GB VRAM
# Aggressive: batch_size = 128, ~19 GB VRAM
```

**3. Sequence Length Optimization:**
- Embedding models: 256-512 tokens thường đủ cho most use cases
- Reranker models: 512 tokens cho document pairs
- Tránh padding unnecessary để giảm VRAM waste

**4. Production Deployment:**
- Sử dụng FP16 precision
- Dynamic batching để tối ưu throughput
- Monitor VRAM usage và adjust batch size theo load
- Consider CPU inference cho embedding models nhỏ (<500M params)

## Đặc điểm VRAM cho Speech-to-Text Models (Whisper)

### Kiến trúc Encoder-Decoder

Whisper và các Speech-to-Text models sử dụng encoder-decoder architecture, khác biệt với decoder-only LLMs:

```
┌─────────────────────────────────────────────────┐
│  Audio Input (30s = ~3000 frames)               │
│              ↓                                   │
│  Encoder (32 layers) → Encoder KV Cache (lớn)   │
│              ↓                                   │
│  Cross-Attention ← Decoder (32 layers)          │
│              ↓                                   │
│  Text Output (~150 tokens) → Decoder KV Cache   │
└─────────────────────────────────────────────────┘
```

### VRAM Components cho Whisper

**1. Model Weights:** ~1.6 GB (FP16) cho Whisper Large v3 Turbo
- Nhỏ hơn nhiều so với LLMs do chỉ ~800M parameters

**2. Encoder KV Cache:** Thành phần lớn nhất
- Phụ thuộc vào audio duration (3000 frames cho 30s audio)
- Tăng tuyến tính với batch size
- Ví dụ: 32 layers × 8 batch × 3000 frames ≈ 3.93 GB

**3. Decoder KV Cache:** Nhỏ hơn nhiều
- Chỉ phụ thuộc vào output text length (~150 tokens)
- Ví dụ: 32 layers × 8 batch × 150 tokens ≈ 0.20 GB

**4. Cross-Attention Cache:** Medium size
- Kết nối encoder và decoder
- Ví dụ: 32 layers × 8 batch × 150 × 3000 ≈ 0.46 GB

### So sánh VRAM: Whisper vs LLM vs Embedding

| Model Type | Model Size | Sequence Type | Dominant VRAM Component | Batch Size Range |
|------------|-----------|---------------|------------------------|------------------|
| LLM (Llama 7B) | 7B params | Text (2048 tokens) | Model Weights | 1-16 |
| Embedding | 300M-600M | Text (512 tokens) | KV Cache (batch lớn) | 32-256 |
| Whisper | 800M | Audio (3000 frames) | Encoder KV Cache | 1-16 |

### Best Practices cho Whisper Models

**1. Precision Selection:**
- **FP16**: Recommended cho production (quality tốt, VRAM hợp lý)
- **INT8**: Có thể sử dụng, giảm ~50% VRAM với minimal WER impact
- **FP32**: Chỉ dùng cho research hoặc khi cần độ chính xác tuyệt đối

**2. Batch Size Optimization:**

```python
# Audio duration: 30 seconds
# GPU: RTX 4090 24GB

# Single request (real-time): batch_size=1, ~3-4 GB VRAM
# Small batch: batch_size=4, ~5-6 GB VRAM
# Medium batch: batch_size=8, ~8-9 GB VRAM
# Large batch: batch_size=16, ~15-16 GB VRAM
```

**3. Audio Duration Impact:**
- Whisper processes audio in 30s chunks maximum
- Longer audio → chunking required → multiple passes
- Real-time streaming: Process audio as it arrives (batch_size=1)

**4. Production Deployment:**

```python
# Real-time transcription (latency-sensitive)
- batch_size = 1
- FP16 precision
- VRAM: ~3-4 GB
- GPU: RTX 3060 12GB or better

# Batch transcription (throughput-optimized)
- batch_size = 8-16
- FP16 precision
- VRAM: ~8-16 GB
- GPU: RTX 4090 24GB, A10 24GB

# Cost-optimized (high volume)
- batch_size = 16
- INT8 quantization
- VRAM: ~8-10 GB
- GPU: RTX 3090 24GB
```

**5. Memory Optimization Techniques:**
- **Chunked inference**: Process long audio in smaller segments
- **Streaming mode**: Process audio incrementally for real-time use
- **Batching strategy**: Group similar-length audio clips to reduce padding waste
- **Mixed precision**: Use FP16 for weights, FP32 for critical operations

### Whisper Model Variants VRAM Comparison

| Model Variant | Parameters | FP16 (batch=1) | FP16 (batch=8) | INT8 (batch=8) |
|---------------|-----------|----------------|----------------|----------------|
| whisper-tiny | 39M | ~1 GB | ~2 GB | ~1.5 GB |
| whisper-base | 74M | ~1.2 GB | ~2.5 GB | ~1.8 GB |
| whisper-small | 244M | ~1.5 GB | ~4 GB | ~2.5 GB |
| whisper-medium | 769M | ~2.5 GB | ~7 GB | ~4.5 GB |
| whisper-large-v3 | 1.55B | ~4 GB | ~12 GB | ~7 GB |
| whisper-large-v3-turbo | 809M | ~3 GB | ~8.5 GB | ~5 GB |

**Note:** Turbo variant có ít parameters hơn nhưng inference speed nhanh hơn ~8x so với large-v3.

## Chiến lược tối ưu hóa VRAM

### 1. Quantization (Lượng tử hóa)

**Giảm model weights:**
- FP8: Giảm 50% memory so với FP16 (7 GB cho 7B model)
- INT8: Giảm 50% memory so với FP16 (7 GB cho 7B model)
- FP4: Giảm 75% memory so với FP16 (3.5 GB cho 7B model)
- INT4: Giảm 75% memory so với FP16 (3.5 GB cho 7B model)

**Trade-off và use cases:**
- FP8: Độ chính xác cao hơn INT8, hỗ trợ tốt trên H100/H200
- INT8: Tương thích rộng rãi, độ chính xác tốt (giảm < 1%)
- FP4: Cân bằng tốt giữa size và quality cho inference
- INT4: Tiết kiệm memory tối đa (giảm 1-2% accuracy)

**Các phương pháp:**
- FP8 Quantization (NVIDIA Transformer Engine)
- GPTQ (GPT Quantization) - INT4/INT8
- AWQ (Activation-aware Weight Quantization) - INT4
- GGUF format (llama.cpp) - FP4/INT4
- SmoothQuant - INT8

### 2. KV Cache Optimization

**PagedAttention (vLLM):**
- Giảm fragmentation
- Tăng throughput lên đến 24×
- Cho phép batch size lớn hơn

**Sliding Window Attention:**
- Giới hạn KV cache cho các token gần đây
- Giảm memory với sequence dài

**Multi-Query Attention (MQA):**
- Chia sẻ keys/values giữa attention heads
- Giảm KV cache size đáng kể

### 3. Batch Size Tuning

**Dynamic Batching:**
- Điều chỉnh batch size dựa trên VRAM available
- Cân bằng giữa throughput và latency

**Continuous Batching:**
- Thêm requests mới vào batch đang chạy
- Tối ưu GPU utilization

### 4. Model Parallelism

**Tensor Parallelism:**
- Split model layers across GPUs
- Giảm VRAM per GPU
- Cần high-speed interconnect (NVLink)

**Pipeline Parallelism:**
- Different layers trên different GPUs
- Hiệu quả với model rất lớn

### 5. Offloading

**CPU Offloading:**
- Load model weights on-demand từ CPU RAM
- Giảm VRAM requirements
- Trade-off: Tăng latency

**Disk Offloading:**
- Extreme case cho model rất lớn
- Latency cao nhất

## Bảng tham khảo nhanh

### VRAM yêu cầu cho các model phổ biến (batch_size=1, seq_len=2048):

| Model                      | Parameters | FP32/FP16 | FP8/INT8 | FP4/INT4 | GPU khuyến nghị        |
|----------------------------|-----------|-----------|----------|----------|------------------------|
| **LLM Models**             |           |           |          |          |                        |
| Llama 2 7B                 | 7B        | ~18 GB    | ~10 GB   | ~6 GB    | RTX 4090, A10, A100    |
| Qwen 2.5 7B Instruct       | 7B        | ~18 GB    | ~9 GB    | ~6 GB    | RTX 3090, RTX 4080     |
| GLM-4.5                    | 9B        | ~23 GB    | ~12 GB   | ~7 GB    | RTX 4090, A10          |
| Llama 2 13B                | 13B       | ~28 GB    | ~16 GB   | ~9 GB    | A100 40GB, RTX 4090    |
| Gemma-3-27B-IT             | 27B       | ~58 GB    | ~34 GB   | ~19 GB   | A100 40GB, H100        |
| Qwen3-32B                  | 32B       | ~68 GB    | ~40 GB   | ~22 GB   | A100 80GB, H100        |
| QwQ-32B                    | 32B       | ~68 GB    | ~40 GB   | ~22 GB   | A100 80GB, H100        |
| Mixtral 8x7B               | 47B       | ~100 GB   | ~55 GB   | ~30 GB   | 2× A100, H100          |
| Llama 2 70B                | 70B       | ~150 GB   | ~88 GB   | ~45 GB   | 2× A100, H100          |
| Llama 3.3 70B Instruct     | 70B       | ~150 GB   | ~88 GB   | ~45 GB   | 2× A100, H100          |
| Mistral 7B                 | 7B        | ~18 GB    | ~10 GB   | ~6 GB    | RTX 4090, A10          |
| GPT-3 175B                 | 175B      | ~370 GB   | ~210 GB  | ~110 GB  | 4-8× A100              |
| Falcon 180B                | 180B      | ~380 GB   | ~215 GB  | ~115 GB  | 4-8× A100              |
| **Embedding Models**       |           |           |          |          |                        |
| gte-multilingual-base      | 305M      | ~2 GB     | ~1 GB    | ~0.6 GB  | RTX 3060, T4           |
| multilingual-e5-large      | 560M      | ~3 GB     | ~2 GB    | ~1 GB    | RTX 3070, T4           |
| **Reranker Models**        |           |           |          |          |                        |
| bge-reranker-v2-m3         | 568M      | ~3 GB     | ~2 GB    | ~1 GB    | RTX 3070, T4           |
| **Speech-to-Text Models**  |           |           |          |          |                        |
| whisper-large-v3-turbo     | 809M      | ~4 GB     | ~2 GB    | ~1.5 GB  | RTX 3070, RTX 4060 Ti  |

**Lưu ý về precision formats:**
- **FP8**: Hỗ trợ tốt nhất trên NVIDIA H100/H200 với Transformer Engine
- **INT8**: Tương thích rộng rãi, hỗ trợ trên hầu hết GPU từ Ampere trở lên
- **FP4**: Format mới, cần framework hỗ trợ (vLLM 0.3+, TRT-LLM)
- **INT4**: Phổ biến nhất cho extreme quantization, hỗ trợ rộng rãi

### GPU VRAM capacity:

| GPU Model       | VRAM    | Use Case                      |
|-----------------|---------|-------------------------------|
| RTX 3090        | 24 GB   | 7B models (INT4/INT8)        |
| RTX 4090        | 24 GB   | 7B models (FP16), 13B (INT4) |
| A10             | 24 GB   | 7B models production         |
| A100 40GB       | 40 GB   | 13B models (FP16)            |
| A100 80GB       | 80 GB   | 70B models (INT8)            |
| H100 80GB       | 80 GB   | 70B models (FP16)            |
| H200            | 141 GB  | 180B models                  |

## Công cụ và framework

### 1. vLLM
- PagedAttention tối ưu KV cache
- Continuous batching
- Throughput cao nhất

```bash
# Ước lượng VRAM
vllm serve <model> --max-model-len <seq_len> --gpu-memory-utilization 0.9
```

### 2. TensorRT-LLM
- Quantization FP8/INT8/INT4 optimization
- FP8 support với H100/H200 Transformer Engine
- Kernel fusion
- NVIDIA GPU optimization

### 3. llama.cpp
- CPU + GPU hybrid
- GGUF quantization format
- Flexible memory management

### 4. Text Generation Inference (TGI)
- Hugging Face solution
- Flash Attention support
- Quantization tích hợp

## Best Practices

### 1. Development Phase
- Sử dụng FP16 cho độ chính xác cao
- Test với batch_size = 1 trước
- Monitor VRAM usage với nvidia-smi

### 2. Production Phase
- Sử dụng INT8/INT4 để giảm chi phí
- Optimize batch_size cho throughput
- Implement dynamic batching
- Monitor và alert VRAM usage

### 3. Cost Optimization
- Choose phù hợp precision format
- Right-size GPU cho workload
- Sử dụng spot instances cho non-critical workloads

### 4. Monitoring Commands

```bash
# Real-time VRAM monitoring
nvidia-smi -l 1

# Detailed memory info
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv -l 1

# Process-level memory
nvidia-smi pmon -c 1
```

## Kết luận

Việc tính toán chính xác VRAM cho LLM inference giúp:
- Chọn GPU phù hợp với budget
- Tối ưu hóa chi phí cloud computing
- Tránh OOM errors trong production
- Maximize throughput và performance

**Quy tắc ngón tay cái:**
- Development: Model weights × 1.5
- Production (single request): Model weights × 1.3
- Production (batch inference): Model weights + (KV Cache × batch_size) × 1.2

**Lưu ý:** Luôn để dư 10-20% VRAM để xử lý spike và overhead không lường trước được.

## Tối ưu hóa cho vLLM Production với Batch Size Lớn

### Đặc điểm quan trọng của vLLM

vLLM tối ưu hóa inference thông qua:
1. **PagedAttention**: Quản lý KV cache hiệu quả, giảm fragmentation
2. **Continuous Batching**: Dynamic batching để tối ưu throughput
3. **GQA (Grouped Query Attention)**: Giảm KV cache size đáng kể

### Công thức KV Cache cho vLLM với GQA

**Công thức chính xác:**

```
KV Cache = 2 × num_layers × batch_size × sequence_length × (num_key_value_heads × head_dim) × 2 bytes

Trong đó:
- num_key_value_heads: Số KV heads (nhỏ hơn num_attention_heads trong GQA)
- head_dim = hidden_size / num_attention_heads
- 2 bytes = FP16 precision cho KV cache (standard trong vLLM)
```

**Lưu ý:** GQA giảm KV cache bằng cách share keys/values giữa nhiều attention heads. Ví dụ:
- Llama-3.3-70B: 64 attention heads nhưng chỉ 8 KV heads → Giảm KV cache 8x!
- Qwen2.5-7B: 28 attention heads nhưng chỉ 4 KV heads → Giảm KV cache 7x!

### Ví dụ thực tế: Production Mix trên H100-80GB

**Giả định:**
- GPU: H100-80GB (90% utilization = 72GB available)
- Average sequence length: 12K tokens (input + output)
- Precision: FP8 weights, FP16 KV cache

#### Case 17: GLM-4-9B với Batch Size 128 (Best Case)

**Thông số:**
- Parameters: 9B
- Layers: 40
- Hidden size: 4096
- Num attention heads: 32
- **Num KV heads: 2** (GQA ratio 16:1)
- Head dim: 128
- Max context: 128K
- Batch size: 128
- Avg sequence: 12K tokens

**Tính toán:**

```
Model Weights (FP8) = 9B × 1 = 9 GB

KV Cache (FP16, GQA) = 2 × 40 × 128 × 12,288 × (2 × 128) × 2
                      = 2 × 40 × 128 × 12,288 × 256 × 2
                      ≈ 40.3 GB

Activations ≈ 1.5 GB

Overhead (15%) = (9 + 40.3 + 1.5) × 0.15 = 7.6 GB

Total VRAM = 9 + 40.3 + 1.5 + 7.6 = 58.4 GB
```

**Kết luận:** ✅ Vừa khít với H100-80GB, có thể serve batch size 128!

**vLLM Command:**
```bash
vllm serve THUDM/glm-4-9b \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --dtype float8
```

---

#### Case 18: Qwen2.5-7B-Instruct với Batch Size 120

**Thông số:**
- Parameters: 7B
- Layers: 28
- Hidden size: 3584
- Num attention heads: 28
- **Num KV heads: 4** (GQA ratio 7:1)
- Head dim: 128
- Max context: 32K
- Batch size: 120 (giảm từ 128 để safe margin)
- Avg sequence: 12K tokens

**Tính toán:**

```
Model Weights (FP8) = 7B × 1 = 7 GB

KV Cache (FP16, GQA) = 2 × 28 × 120 × 12,288 × (4 × 128) × 2
                      ≈ 52.3 GB

Activations ≈ 1.2 GB

Overhead (15%) = (7 + 52.3 + 1.2) × 0.15 = 9.1 GB

Total VRAM = 7 + 52.3 + 1.2 + 9.1 = 69.6 GB
```

**Kết luận:** ✅ Safe với H100-80GB, có dư ~10GB cho spikes.

**vLLM Command:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 32768 \
  --max-num-seqs 120 \
  --gpu-memory-utilization 0.90 \
  --dtype float8
```

---

#### Case 19: Qwen3-32B với Batch Size 20 (Limited Throughput)

**Thông số:**
- Parameters: 32.8B
- Layers: 64
- Hidden size: 5120
- Num attention heads: 64
- **Num KV heads: 8** (GQA ratio 8:1)
- Head dim: 80
- Max context: 128K
- Batch size: 20 (constrained bởi VRAM)
- Avg sequence: 12K tokens

**Tính toán:**

```
Model Weights (FP8) = 32.8B × 1 = 32.8 GB

KV Cache (FP16, GQA) = 2 × 64 × 20 × 12,288 × (8 × 80) × 2
                      ≈ 29.3 GB

Activations ≈ 1.8 GB

Overhead (15%) = (32.8 + 29.3 + 1.8) × 0.15 = 9.6 GB

Total VRAM = 32.8 + 29.3 + 1.8 + 9.6 = 73.5 GB
```

**Kết luận:** ⚠️ Batch size rất nhỏ (20), throughput bị giới hạn nghiêm trọng. Consider multi-GPU setup.

**vLLM Command:**
```bash
vllm serve Qwen/Qwen3-32B \
  --max-model-len 131072 \
  --max-num-seqs 20 \
  --gpu-memory-utilization 0.92 \
  --dtype float8
```

---

#### Case 20: Gemma-3-27B-IT với Batch Size 24

**Thông số:**
- Parameters: 27B
- Layers: 46
- Hidden size: 4096
- Num attention heads: 64
- **Num KV heads: 16** (GQA ratio 4:1)
- Head dim: 64
- Max context: 32K
- Batch size: 24
- Avg sequence: 12K tokens

**Tính toán:**

```
Model Weights (FP8) = 27B × 1 = 27 GB

KV Cache (FP16, GQA) = 2 × 46 × 24 × 12,288 × (16 × 64) × 2
                      ≈ 33.1 GB

Activations ≈ 1.5 GB

Overhead (15%) = (27 + 33.1 + 1.5) × 0.15 = 9.2 GB

Total VRAM = 27 + 33.1 + 1.5 + 9.2 = 70.8 GB
```

**Kết luận:** ⚠️ Batch size thấp (24), không tối ưu cho production throughput cao.

**vLLM Command:**
```bash
vllm serve google/gemma-3-27b-it \
  --max-model-len 32768 \
  --max-num-seqs 24 \
  --gpu-memory-utilization 0.90 \
  --dtype float8
```

---

#### Case 21: QwQ-32B với Batch Size 20

**Thông số:**
- Parameters: 32B
- Layers: 64
- Hidden size: 5120
- Num attention heads: 40
- **Num KV heads: 8** (GQA ratio 5:1)
- Head dim: 128
- Max context: 32K
- Batch size: 20
- Avg sequence: 12K tokens

**Tính toán:**

```
Model Weights (FP8) = 32B × 1 = 32 GB

KV Cache (FP16, GQA) = 2 × 64 × 20 × 12,288 × (8 × 128) × 2
                      ≈ 31.5 GB

Activations ≈ 1.8 GB

Overhead (15%) = (32 + 31.5 + 1.8) × 0.15 = 9.8 GB

Total VRAM = 32 + 31.5 + 1.8 + 9.8 = 75.1 GB
```

**Kết luận:** ⚠️ Vừa khít với H100-80GB nhưng batch size chỉ 20. Consider giảm average sequence length hoặc multi-GPU.

**vLLM Command:**
```bash
vllm serve Qwen/QwQ-32B-Preview \
  --max-model-len 32768 \
  --max-num-seqs 20 \
  --gpu-memory-utilization 0.93 \
  --dtype float8
```

---

#### Case 22: Llama-3.3-70B-Instruct - Không khả thi Single GPU

**Thông số:**
- Parameters: 70B
- Layers: 80
- Hidden size: 8192
- Num attention heads: 64
- **Num KV heads: 8** (GQA ratio 8:1)
- Head dim: 128
- Max context: 128K

**Tính toán (batch size = 1):**

```
Model Weights (FP8) = 70B × 1 = 70 GB

KV Cache (FP16, GQA, batch=1) = 2 × 80 × 1 × 12,288 × (8 × 128) × 2
                                ≈ 2.5 GB

Activations ≈ 0.5 GB

Overhead (15%) = (70 + 2.5 + 0.5) × 0.15 = 10.9 GB

Total VRAM = 70 + 2.5 + 0.5 + 10.9 = 83.9 GB
```

**Kết luận:** ❌ Ngay cả với batch size 1, vẫn vượt H100-80GB. **Bắt buộc phải dùng 2x H100 với Tensor Parallelism.**

**vLLM Command (2x H100):**
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --dtype float8 \
  --tensor-parallel-size 2
```

---

### Bảng tổng hợp: Batch Size tối ưu cho H100-80GB

| Model | Max Context | **Khuyến nghị Batch Size** | Total VRAM | Throughput | Multi-GPU? |
|-------|-------------|---------------------------|------------|------------|------------|
| **GLM-4-9B** | 128K | **128** ✅ | 58.4 GB | Excellent | No |
| **Qwen2.5-7B-Instruct** | 32K | **120** ✅ | 69.6 GB | Excellent | No |
| **Gemma-3-27B-IT** | 32K | **24** ⚠️ | 70.8 GB | Limited | Recommended |
| **Qwen3-32B** | 128K | **20** ⚠️ | 73.5 GB | Poor | Recommended |
| **QwQ-32B** | 32K | **20** ⚠️ | 75.1 GB | Poor | Recommended |
| **Llama-3.3-70B** | 128K | **Không khả thi** ❌ | 83.9 GB | - | Required (2x+) |

**Lưu ý quan trọng:**
- ✅ **Excellent throughput**: Batch size ≥100, phù hợp production traffic cao
- ⚠️ **Limited/Poor throughput**: Batch size <30, cân nhắc multi-GPU hoặc giảm average sequence length
- ❌ **Không khả thi**: Model weights alone vượt 80GB, bắt buộc tensor parallelism

---

### Chiến lược tối ưu hóa cho Production

#### 1. Giảm Average Sequence Length

Nếu có thể giảm average sequence từ 12K → 8K tokens:

```python
# Ví dụ: Qwen3-32B với 8K average tokens
KV Cache = 2 × 64 × 20 × 8,192 × (8 × 80) × 2 ≈ 19.5 GB  # Giảm từ 29.3 GB
Total VRAM ≈ 63 GB  # Tiết kiệm 10.5 GB

# → Có thể tăng batch size lên 30 (+50% throughput!)
```

**Cách thực hiện:**
- Implement smart chunking cho long documents
- Optimize prompt engineering để giảm context
- Sử dụng sliding window attention

#### 2. Multi-GPU Tensor Parallelism

Cho models >30B parameters với batch size yêu cầu cao:

```bash
# Qwen3-32B trên 2x H100-80GB
vllm serve Qwen/Qwen3-32B \
  --max-model-len 131072 \
  --max-num-seqs 80 \        # Tăng từ 20 → 80 (4x throughput!)
  --gpu-memory-utilization 0.90 \
  --dtype float8 \
  --tensor-parallel-size 2

# Model weights: 32.8 GB / 2 = 16.4 GB per GPU
# KV Cache: Shared across GPUs
# Total per GPU: ~50-55 GB → Còn nhiều dư!
```

#### 3. Mixed Batch Processing

Kết hợp requests với sequence lengths khác nhau:

```python
# vLLM tự động optimize
# 70% requests: 4K tokens
# 20% requests: 8K tokens
# 10% requests: 16K tokens
# → Effective average: ~6K tokens (giảm từ 12K)
# → Tăng batch size capacity ~2x!
```

#### 4. FP8 Quantization cho H100

H100 có Transformer Engine hỗ trợ FP8 native:

```bash
# Với FP8 quantization:
# - Model weights: Giảm từ FP16 (2 bytes) → FP8 (1 byte) = 50% savings
# - KV Cache: Vẫn FP16 (cần accuracy cao)
# - Minimal quality loss (<1% perplexity)

vllm serve <model> \
  --dtype float8 \              # Bật FP8 cho weights
  --kv-cache-dtype fp16         # Giữ FP16 cho KV cache (accuracy)
```

---

### Monitoring và Debugging Commands

```bash
# 1. Real-time VRAM monitoring
watch -n 1 nvidia-smi

# 2. vLLM memory profiling
vllm serve <model> --enable-prefix-caching --log-level debug

# 3. Check actual batch size và throughput
curl http://localhost:8000/metrics | grep vllm_num_requests

# 4. Analyze KV cache usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

---

### Best Practices Summary

**✅ DO:**
- Sử dụng GQA-enabled models (Llama 3, Qwen 2.5, Gemma 3)
- Enable FP8 quantization trên H100
- Monitor average sequence length trong production
- Sử dụng vLLM continuous batching
- Set `--gpu-memory-utilization 0.90` để dành buffer cho spikes

**❌ DON'T:**
- Không assume batch_size=128 cho mọi model
- Không quên tính KV cache với actual sequence length
- Không dùng FP8 cho KV cache (quality loss đáng kể)
- Không ignore GQA ratio khi tính toán

---

## Tài liệu tham khảo

- [SkyMod - Practical Guide to Inference VRAM Consumption](https://skymod.tech/how-much-memory-does-your-llm-really-need-a-practical-guide-to-inference-vram-consumption/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Hugging Face TGI](https://github.com/huggingface/text-generation-inference)
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [GQA Paper: Fast Transformer Decoding](https://arxiv.org/abs/2305.13245)
- [H100 Transformer Engine Guide](https://docs.nvidia.com/deeplearning/transformer-engine/)
