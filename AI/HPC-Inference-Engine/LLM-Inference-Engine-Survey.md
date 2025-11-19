# Survey Các Techstack về Inference Engine cho LLM

## Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [AIBrix](#1-aibrix)
3. [KServe](#2-kserve)
4. [NVIDIA Dynamo](#3-nvidia-dynamo)
5. [LLM-D](#4-llm-d)
6. [Bảng so sánh tổng quan](#bảng-so-sánh-tổng-quan)
7. [So sánh chi tiết](#so-sánh-chi-tiết)
8. [Use Cases và Model Size Support](#use-cases-và-model-size-support)
9. [Case Studies thực tế về Performance và SLA/SLO](#case-studies-thực-tế-về-performance-và-slaslo)
10. [Kết luận và Khuyến nghị](#kết-luận-và-khuyến-nghị)
11. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

Trong bối cảnh các mô hình ngôn ngữ lớn (LLM) ngày càng phát triển về quy mô và độ phức tạp, việc lựa chọn inference engine phù hợp trở nên quan trọng hơn bao giờ hết. Các yếu tố then chốt bao gồm:

- **Performance**: Độ trễ (latency), thông lượng (throughput)
- **Cost-efficiency**: Chi phí vận hành trên đơn vị inference
- **Scalability**: Khả năng mở rộng theo nhu cầu
- **Routing intelligence**: Điều phối request hiệu quả
- **Caching mechanisms**: Tối ưu hóa tái sử dụng KV cache

Tài liệu này phân tích 4 techstack hàng đầu: **AIBrix**, **KServe**, **NVIDIA Dynamo**, và **LLM-D**, với trọng tâm vào cơ chế routing, caching, và performance.

---

## 1. AIBrix

### 1.1. Tổng quan

**AIBrix** là một framework cloud-native mã nguồn mở được phát triển bởi ByteDance và hiện nay được quản lý dưới dự án vLLM. AIBrix được thiết kế để cung cấp control plane có khả năng mở rộng và chi phí hiệu quả cho việc triển khai inference LLM ở quy mô doanh nghiệp.

**Nguồn**:
- Documentation: https://aibrix.readthedocs.io/latest/index.html
- GitHub: https://github.com/vllm-project/aibrix
- Blog chính thức vLLM: https://blog.vllm.ai/2025/02/21/aibrix-release.html

### 1.2. Tính năng cốt lõi

#### a. LLM Gateway với Routing thông minh
AIBrix mở rộng Envoy Gateway để tạo ra một LLM-aware gateway với khả năng:
- Phân tích token patterns
- Nhận biết prefix cache availability
- Tối ưu routing dựa trên compute overhead

#### b. High-density LoRA Support
Hỗ trợ triển khai hàng nghìn LoRA adapters với hiệu suất cao

#### c. Distributed Inference
Khả năng phân tán inference trên nhiều nodes

#### d. Autoscaling thông minh
Autoscaler được tối ưu hóa riêng cho workload LLM

#### e. GPU Hardware Fault Detection
Phát hiện và xử lý lỗi phần cứng GPU

### 1.3. Cơ chế Routing

AIBrix cung cấp **8+ routing strategies**, đây là một trong những điểm mạnh nhất của platform:

1. **random**: Route tới pod ngẫu nhiên
2. **least-request**: Route tới pod có ít request đang xử lý nhất
3. **throughput**: Route tới pod có tổng weighted tokens thấp nhất
4. **prefix-cache**: Route tới pod đã có KV cache khớp với prompt prefix
5. **least-busy-time**: Route tới pod có tổng thời gian xử lý tích lũy thấp nhất
6. **least-kv-cache**: Route tới pod có KV cache size nhỏ nhất
7. **least-latency**: Route tới pod có average latency thấp nhất
8. **prefix-cache-preble**: Route kết hợp cả prefix cache hits và pod load

**Dẫn chứng hiệu quả**: Theo bài báo "AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure" (arXiv:2504.03648v1), việc áp dụng least-GPU-memory strategies giúp giảm:
- **Mean latency: -19.2%**
- **P99 latency: -79%**

**Nguồn**:
- Gateway Routing Documentation: https://aibrix.readthedocs.io/latest/features/gateway-plugins.html
- Paper: https://arxiv.org/html/2504.03648v1

### 1.4. Cơ chế Caching

#### Distributed KV Cache Runtime
AIBrix giới thiệu **Distributed KV Cache Runtime**, một hệ thống mở rộng external cache services để quản lý KV cache được tạo động trong quá trình inference.

**Tính năng chính**:
- **Prefix Cache-aware Routing**: Cho multi-turn chat hoặc requests dựa trên partial summaries, gateway có thể route chúng tới cùng pods đã giữ relevant KV cache
- **Cross-Engine KV Reuse**: Tái sử dụng cache giữa các inference engines
- **KVCache Offloading**: Giảm memory pressure trên GPU

**Lợi ích thực tế**:
- Giảm redundant computation cho prompts tương tự
- Tăng cache hit rate cho production workloads
- Tối ưu hóa GPU memory utilization

**Nguồn**:
- AIBrix Blogs: https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/
- The New Stack: https://thenewstack.io/a-look-at-aibrix-an-open-source-llm-inference-platform/

### 1.5. Performance và Khả năng mở rộng

**Production-proven tại ByteDance**:
- Đã được triển khai để hỗ trợ multiple business use cases tại ByteDance
- Cloud-native architecture cho phép elastic scaling
- Dynamic autoscaling dựa trên actual demand

**Performance metrics**:
- P99 latency improvement: **79%**
- Mean latency reduction: **19.2%**
- Hỗ trợ high-density LoRA deployments (hàng nghìn adapters)

**Nguồn**:
- Run It On Cloud Blog: https://aymen-segni.com/index.php/2025/03/14/aibrix-revolutionizing-llm-inference-production-deployments/
- Medium Analysis: https://ai.gopubby.com/unlocking-scalable-llm-inference-introducing-aibrix-for-vllm-908e3d64bf74

### 1.6. Kiến trúc

**Core Components**:
1. **StormService**: Service orchestration layer
2. **Router**: Traffic routing với multiple strategies
3. **KVCache Offloading Framework**: Quản lý cache lifecycle
4. **Autoscaler**: LLM-tailored autoscaling logic
5. **AI Engine Runtime**: Riêng biệt cho inference execution

**Tech Stack**:
- Kubernetes-native
- Extends Envoy Gateway
- Integrates với vLLM engine
- Support cho multiple cloud providers

---

## 2. KServe

### 2.1. Tổng quan

**KServe** là một nền tảng model serving chuẩn mực trên Kubernetes, được thiết kế để triển khai và mở rộng các mô hình AI từ LLMs đến traditional ML models. KServe đã trở thành **CNCF Incubating Project**, đánh dấu sự trưởng thành và adoption rộng rãi trong cộng đồng.

**Nguồn**:
- Documentation: https://kserve.github.io/website/docs/intro
- Red Hat Developer Articles: https://developers.redhat.com/articles/2024/03/15/empower-conversational-ai-scale-kserve

### 2.2. Tính năng cốt lõi

#### a. GenAI-First Approach
- OpenAI-compatible APIs cho chat completions, streaming, embeddings
- Optimized cho modern LLM workloads

#### b. Framework Agnostic
Hỗ trợ đa dạng frameworks:
- TensorFlow, PyTorch, Scikit-Learn, XGBoost, ONNX
- Hugging Face Transformers
- Custom runtimes

#### c. Serverless Capabilities
- Auto-scaling xuống zero
- Không yêu cầu quản lý infrastructure thủ công
- Tự động load balancing

#### d. Production-Ready Features
- Canary deployments
- Blue-green deployments
- Multi-model serving
- Batch inference support

### 2.3. Cơ chế Routing

KServe sử dụng **InferenceGraph** CRD để xây dựng routing logic phức tạp:

**Routing Features**:
1. **Predictor Routing**: Direct traffic tới specific model versions
2. **Canary Rollouts**: Gradually shift traffic between versions
3. **A/B Testing**: Split traffic theo percentage
4. **Transformer Chaining**: Pre/post-processing pipelines

**Request Routing Flow**:
```
User Request → Ingress Gateway → InferenceService
→ Transformer (optional) → Predictor → Response
```

**Multi-node Serving**: Hỗ trợ distributed LLM serving across multiple nodes với intelligent request distribution.

**Nguồn**:
- KServe Documentation on InferenceGraph
- Red Hat Blog: https://developers.redhat.com/articles/2024/03/15/empower-conversational-ai-scale-kserve

### 2.4. Cơ chế Caching

#### LocalModelCache CRD
Đây là một **innovation quan trọng** của KServe, giải quyết vấn đề cold-start cho large models.

**Thành tựu**:
- Giảm startup time từ **15-20 phút xuống ~1 phút**
- Cache models cục bộ trên nodes để reuse
- Hỗ trợ multiple storage backends (S3, GCS, Azure, PVC)

**Cách hoạt động**:
1. Model được download lần đầu và cache trên node
2. Subsequent pod startups sử dụng cached model
3. Cache được quản lý lifecycle tự động

#### KV Cache Offloading
**Integration với LMCache**: Theo blog LMCache (https://blog.lmcache.ai/en/2025/05/16/how-lmcache-turbocharges-enterprise-llm-inference-frameworks/):

- KServe đã tích hợp hỗ trợ LMCache
- Tối ưu hóa memory management cho long conversations
- Breakthrough performance improvements
- Giảm inference costs while ensuring SLOs

**KV Cache Features**:
- Offload KV cache ra external storage khi cần
- Tối ưu cho multi-turn conversations
- Giảm GPU memory pressure

**Nguồn**:
- LMCache Blog: https://blog.lmcache.ai/en/2025/05/16/how-lmcache-turbocharges-enterprise-llm-inference-frameworks/
- Red Hat vLLM Article: https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today

### 2.5. Performance và Khả năng mở rộng

#### Autoscaling Intelligence
KServe cung cấp multiple autoscaling strategies:

1. **Token Throughput-based**: Scale dựa trên tokens/second
2. **Queue Depth-based**: Scale khi request queue depth cao
3. **GPU Utilization-based**: Scale dựa trên GPU metrics
4. **Scale-to-Zero**: Tiết kiệm chi phí khi không có traffic

#### Performance Benchmarks

Theo Red Hat benchmarks (https://www.redhat.com/en/blog/evaluating-llm-inference-performance-red-hat-openshift-ai):

- Tested với Caikit + TGIS trên OpenShift AI clusters (AWS)
- 4-minute load tests với llm-load-test
- Automatic scaling dựa trên request load
- Integration với ReplicaSets cho model copy serving

#### MLPerf Inference v5.0 Results
Theo MLCommons (https://mlcommons.org/2025/04/llm-inference-v5/):

- Benchmarks mới cho Llama 3.1 405B Instruct
- Llama 2 Chat 70B
- Framework chuẩn cho evaluating large-scale LLM inference

### 2.6. Kiến trúc

#### Control Plane Components
1. **InferenceService CRD**: Định nghĩa model deployment
2. **InferenceGraph CRD**: Multi-step inference workflows
3. **ServingRuntime**: Runtime templates cho specific frameworks
4. **ClusterServingRuntime**: Cluster-wide runtime definitions
5. **LocalModelCache CRD**: Model caching configuration

#### Data Plane Components
1. **Predictor**: Model serving component
2. **Transformer**: Pre/post-processing (optional)
3. **Explainer**: Model explanation (optional)

#### Tech Stack
- **Kubernetes-native**: Leverages K8s primitives
- **Knative Serving**: Serverless capabilities
- **Istio/Service Mesh**: Advanced traffic management
- **OpenShift Serverless**: Enterprise integration (Red Hat)
- **Prometheus/Grafana**: Observability

**Nguồn**:
- KServe Architecture Docs
- Kubernetes-Based LLM Inference Overview: https://rudeigerc.dev/posts/kubernetes-based-llm-inference-architectures-an-overview/

---

## 3. NVIDIA Dynamo

### 3.1. Tổng quan

**NVIDIA Dynamo** là một high-throughput, low-latency open-source inference serving framework được NVIDIA giới thiệu tại GTC 2025. Dynamo được thiết kế đặc biệt để phục vụ các reasoning AI models ở quy mô datacenter với performance đột phá.

**Đặc điểm nổi bật**:
- Up to **30x throughput boost** cho DeepSeek-R1 models trên NVIDIA Blackwell
- Industry-first record: **1.1 million tokens/second** với 72 NVIDIA Blackwell Ultra GPUs

**Nguồn**:
- NVIDIA Technical Blog: https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- GitHub: https://github.com/ai-dynamo/dynamo
- NVIDIA Developer Portal: https://developer.nvidia.com/dynamo

### 3.2. Tính năng cốt lõi

#### a. Multi-Backend Support
Hỗ trợ multiple inference backends:
- **vLLM**: Popular open-source engine
- **SGLang**: Structured generation language
- **TensorRT-LLM**: NVIDIA's optimized engine

#### b. Advanced Tool Calling & Multimodality
- Native support cho function/tool calling
- Multimodal processing (text, images, etc.)
- Structured output generation

#### c. KVBM (KV Buffer Management)
KV-cache management system được tối ưu hóa cao, tích hợp sâu với:
- vLLM
- TensorRT-LLM

#### d. Cloud-Native Kubernetes Deployment
Full Kubernetes orchestration cho enterprise-scale deployments

### 3.3. Cơ chế Routing

**Router Component**: Thành phần core điều hướng requests tới appropriate workers.

**Routing Features**:

1. **SLA-based Planner**:
   - Tối ưu performance dựa trên Service Level Agreements
   - Intelligent request distribution
   - Priority-aware routing

2. **Grove (Multinode Orchestration)**:
   - Quản lý distributed deployments
   - Cross-node request coordination
   - Load balancing across GPU clusters

3. **Model-aware Routing**:
   - Route requests based on model capabilities
   - Backend selection (vLLM vs TensorRT-LLM vs SGLang)
   - Hardware affinity routing

**Architecture Flow**:
```
Client → Frontend (OpenAI API :8000)
→ Router → etcd/NATS (message passing)
→ Workers (Python-based backends) → Response
```

**Nguồn**:
- NVIDIA Dynamo Documentation: https://docs.nvidia.com/dynamo/latest/index.html
- Collabnix Getting Started Guide: https://collabnix.com/getting-started-with-nvidia-dynamo-a-powerful-framework-for-distributed-llm-inference/

### 3.4. Cơ chế Caching

#### KVBM (KV Buffer Management)
**Core caching innovation** của Dynamo:

**Features**:
- Efficient KV-cache lifecycle management
- Deep integration với vLLM và TensorRT-LLM
- Optimized memory allocation/deallocation
- Support cho distributed cache across nodes

**Version 0.4 Enhancement**:
Theo NVIDIA Blog (https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/):

- **Disaggregation Architecture**: Tách riêng prefill và decode phases
- **4x faster inference performance** on NVIDIA Blackwell
- Prefill cache được tối ưu riêng
- Decode cache được quản lý độc lập

**Benefits**:
- Reduced memory fragmentation
- Better GPU utilization
- Lower latency cho multi-turn conversations
- Improved throughput for concurrent requests

### 3.5. Performance và Khả năng mở rộng

#### Record-Breaking Performance

**DeepSeek-R1 Performance** (NVIDIA Blackwell):
- **30x throughput increase** so với baseline
- Industry-first: **1.1 million tokens/second** aggregate throughput
- 72 NVIDIA Blackwell Ultra GPUs cluster

**Nguồn**:
- NVIDIA Blog: https://blogs.nvidia.com/blog/think-smart-dynamo-ai-inference-data-center/
- Technical Analysis: Signal65 research paper by Russ Fellows

#### Version 0.4 Improvements

**4x Performance Boost**:
- Disaggregation cho prefill/decode
- Optimized cho NVIDIA Blackwell architecture
- Reduced time-to-first-token (TTFT)
- More predictable time-per-output-token (TPOT)

**SLO-based Autoscaling**:
- Automatic scaling dựa trên SLA requirements
- Real-time observability metrics
- Predictive scaling algorithms

**Nguồn**:
- Dynamo 0.4 Release Blog: https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/

#### MoE Models Optimization

**NVIDIA GB200 NVL72 + Dynamo** (https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/):

- Powerful compounding effect for Mixture-of-Experts models
- Optimized cho DeepSeek R1 và Llama 4 models
- Technical white paper available for large-scale GPU cluster deployments

#### Cloud Provider Integration

Theo NVIDIA (https://blogs.nvidia.com/blog/think-smart-dynamo-ai-inference-data-center/):

Major cloud providers boosting AI inference with Dynamo:
- **AWS**: Native integration
- **Google Cloud**
- **Microsoft Azure**
- **Oracle Cloud Infrastructure (OCI)**

**AWS-specific optimizations**: https://developer.nvidia.com/blog/nvidia-dynamo-adds-support-for-aws-services-to-deliver-cost-efficient-inference-at-scale/

### 3.6. Kiến trúc

#### Core Components

1. **Frontend Layer**:
   - OpenAI-compatible API (port 8000)
   - Request validation và preprocessing
   - Response formatting

2. **Router**:
   - SLA-based planning
   - Worker selection
   - Load balancing

3. **Message Passing Infrastructure**:
   - **etcd**: Configuration và coordination
   - **NATS**: High-performance messaging
   - Low-latency communication

4. **Worker Layer**:
   - Python-based inference backends
   - vLLM / SGLang / TensorRT-LLM engines
   - GPU-optimized execution

5. **Observability Stack**:
   - **Prometheus**: Metrics collection
   - **Grafana**: Visualization
   - Health checks và monitoring
   - Real-time performance tracking

#### Disaggregation Architecture (v0.4+)

```
Request Flow:
Client → Frontend → Router
→ Prefill Workers (process prompts, cache KV)
→ Decode Workers (generate tokens, reuse cached KV)
→ Response
```

**Benefits của Disaggregation**:
- Specialized optimization cho từng phase
- Better resource utilization
- Improved throughput
- Reduced latency variance

**Tech Stack**:
- Kubernetes-native orchestration
- NVIDIA GPU optimizations (CUDA, TensorRT)
- Support multiple backends (vLLM, SGLang, TensorRT-LLM)
- etcd + NATS messaging
- Prometheus + Grafana observability

**Nguồn**:
- NVIDIA Documentation: https://docs.nvidia.com/dynamo/latest/index.html
- Medium Analysis: https://medium.com/byte-sized-ai/demystifying-nvidia-dynamo-a-high-performance-inference-framework-for-scalable-genai-f10be3d7032f

---

## 4. LLM-D

### 4.1. Tổng quan

**LLM-D** là một Kubernetes-native high-performance distributed inference framework được thiết kế để serve large generative AI models at scale. Framework này tập trung vào việc cung cấp "fastest time-to-value and competitive performance per dollar" cho most models across diverse hardware accelerators.

**Philosophy**: Well-lit paths for anyone to serve large generative AI models at scale

**Nguồn**:
- Official Website: https://llm-d.ai/
- GitHub: https://github.com/llm-d/llm-d
- Red Hat Developer: https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing

### 4.2. Tính năng cốt lõi

#### a. Wide Expert-Parallelism (Wide EP)
**Breakthrough feature** cho Mixture-of-Experts (MoE) models:

- Deploy very large MoE models như DeepSeek-R1
- Significantly reduces end-to-end latency
- Increases throughput bằng cách scale với Data Parallelism + Expert Parallelism
- Tận dụng fast accelerator networks

**Nguồn**:
- Red Hat Article: https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep

#### b. Prefill/Decode Disaggregation
Thiết kế lấy cảm hứng từ DeepSeek's inference system discussion:

- **Prefill Servers**: Xử lý input prompts
- **Decode Servers**: Generate output tokens
- **Benefits**:
  - Reduced time-to-first-token (TTFT)
  - More predictable time-per-output-token (TPOT)
  - Better resource utilization

#### c. Multi-Accelerator Support
**Hardware flexibility** across vendors:
- **NVIDIA GPUs**: Full support
- **AMD GPUs**: Production-ready (MI300X tested)
- **Google TPUs**: Supported
- **Intel XPUs**: In roadmap

**Nguồn**:
- AMD ROCm Blog: https://rocm.blogs.amd.com/artificial-intelligence/llm-d-distributed/README.html

#### d. Kubernetes-Native Architecture
Full integration với Kubernetes ecosystem:
- Native CRDs
- Kubernetes-native orchestration
- Cloud-agnostic deployment

### 4.3. Cơ chế Routing

LLM-D sử dụng **Inference Gateway (IGW)** - một official Kubernetes project extending Gateway API.

**Routing Features**:

1. **Predicted Latency Balancing** (v0.3):
   - Dự đoán latency cho mỗi request
   - Route tới workers có predicted latency thấp nhất
   - Adaptive learning từ historical data

2. **Prefix Cache-Aware Routing** (v0.3):
   - Optimized routing dựa trên prefix cache availability
   - Route requests với shared prefixes tới same workers
   - Maximize cache hit rate

3. **Expert-Aware Routing** (cho MoE models):
   - Route dựa trên expert distribution
   - Wide EP scheduling
   - Load balancing across expert replicas

4. **Multi-Model Routing**:
   - Support multiple models on same cluster
   - Intelligent model selection
   - Resource-aware scheduling

**Architecture**:
```
Request → Inference Gateway (IGW)
→ Predicted Latency Calculator
→ Prefix Cache Checker
→ Worker Selection (vLLM instances)
→ Response
```

**Nguồn**:
- LLM-D v0.3 Release: https://llm-d.ai/blog
- Kubernetes Gateway API Integration

### 4.4. Cơ chế Caching

#### KV Cache Manager
**Core caching component** với advanced features:

1. **Distributed Cache Coordination**:
   - Centralized KV Cache Manager
   - Cross-worker cache visibility
   - Intelligent cache placement

2. **Prefix Cache Optimization**:
   - Share common prompt prefixes
   - Reduce redundant computation
   - Improved routing cho cache hits

3. **Disaggregated Cache Architecture**:
   - Prefill phase: Generate và store KV cache
   - Decode phase: Reuse cached KV pairs
   - Separated memory management

**Caching Flow**:
```
Prefill Workers:
Input → Token Processing → KV Cache Generation → Store in Cache Manager

Decode Workers:
Cache Manager Lookup → Retrieve Cached KV → Token Generation
```

#### DeepSeek-Specific Optimizations

Theo Red Hat (https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference):

**DeepSeek-V3.2-Exp Integration**:
- Sparse Attention for long-context inference
- Ready for experimentation with Red Hat AI
- Optimized cache management cho long conversations

**DeepSeek Design Learnings**:
- Aggressive disaggregation strategy
- Remarkable performance at scale
- Cache-aware scheduling

### 4.5. Performance và Khả năng mở rộng

#### Latest Release Performance (v0.3)

**High Scale DeepSeek Serving**:
- Wide expert-parallelism cho massive MoE models
- Predicted latency balancing algorithm
- Better prefix cache routing

**Efficiency Metrics**:
- "Fastest time-to-value"
- "Competitive performance per dollar"
- Optimized cho diverse hardware accelerators

#### Hardware Performance

**AMD MI300X Cluster Integration** (AMD ROCm Blog):
- Production-ready distributed LLM serving
- Competitive với NVIDIA solutions
- Cost-effective alternative

**Multi-Accelerator Benchmarks**:
- Unified interface across vendors
- Consistent performance tuning
- Hardware-agnostic optimizations

#### Scalability Features

1. **Kubernetes Auto-scaling**:
   - HPA (Horizontal Pod Autoscaler) integration
   - Custom metrics từ Inference Gateway
   - Predictive scaling

2. **Multi-Node Deployments**:
   - Scale across multiple nodes seamlessly
   - Fast accelerator network utilization
   - Distributed coordination via K8s

3. **Cost Optimization**:
   - Efficient resource utilization
   - Right-sizing recommendations
   - Multi-tenant support

**Nguồn**:
- LLM-D Community Announcement: https://llm-d.ai/blog/llm-d-announce
- Red Hat Developer Articles

### 4.6. Kiến trúc

#### Core Components

1. **Inference Gateway (IGW)**:
   - Kubernetes Gateway API extension
   - Request routing và load balancing
   - Latency prediction engine
   - Prefix cache routing logic

2. **KV Cache Manager**:
   - Centralized cache coordination
   - Cross-worker cache visibility
   - Cache lifecycle management
   - Memory optimization

3. **Inference Scheduler**:
   - Wide EP scheduling cho MoE models
   - Disaggregated prefill/decode scheduling
   - Resource allocation optimization

4. **vLLM Engine Integration**:
   - Leading open-source LLM inference engine
   - Support wide range of models (Llama, DeepSeek, etc.)
   - Multi-accelerator backend

5. **Kubernetes Control Plane**:
   - Native CRDs cho model definitions
   - Infrastructure orchestration
   - Workload management

#### Technology Stack

**Core Technologies**:
- **vLLM**: Model server và inference engine
- **Kubernetes**: Orchestration platform
- **Inference Gateway API**: Extended Gateway API
- **Multi-vendor accelerators**: NVIDIA, AMD, Google TPU, Intel

**Integration Ecosystem**:
- OpenShift compatibility (Red Hat)
- Cloud-agnostic (AWS, GCP, Azure, on-prem)
- Standard Kubernetes tooling

#### Disaggregation Architecture

```
┌─────────────────────────────────────────┐
│         Inference Gateway (IGW)         │
│  - Latency Prediction                   │
│  - Prefix Cache Routing                 │
│  - Load Balancing                       │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────┐
│   Prefill   │  │   Decode    │
│   Workers   │  │   Workers   │
│  (vLLM)     │  │  (vLLM)     │
└──────┬──────┘  └──────┬──────┘
       │                │
       └────────┬────────┘
                │
       ┌────────▼─────────┐
       │  KV Cache Mgr    │
       │  - Coordination  │
       │  - Storage       │
       └──────────────────┘
```

**Design Principles**:
- Kubernetes-native (không phải wrapper)
- Open standards (Gateway API, CRDs)
- Multi-vendor hardware support
- Production-ready from day 1

**Nguồn**:
- GitHub Architecture: https://github.com/llm-d/llm-d
- Kubernetes LLM Inference Overview: https://rudeigerc.dev/posts/kubernetes-based-llm-inference-architectures-an-overview/

---

## Bảng so sánh tổng quan

| **Tiêu chí** | **AIBrix** | **KServe** | **NVIDIA Dynamo** | **LLM-D** |
|--------------|-----------|-----------|------------------|----------|
| **Nguồn gốc** | ByteDance → vLLM Project | CNCF Incubating | NVIDIA (GTC 2025) | Community-driven |
| **License** | Open Source | Open Source | Open Source | Open Source |
| **Kiến trúc** | Cloud-native, vLLM-focused | Kubernetes-native, Multi-framework | Multi-backend (vLLM/SGLang/TRT-LLM) | K8s-native, vLLM-based |
| **Routing Strategies** | 8+ strategies (prefix-cache, least-latency, etc.) | InferenceGraph, Canary, A/B testing | SLA-based, Model-aware | Predicted latency, Prefix-aware, Wide EP |
| **Caching** | Distributed KV Cache Runtime, Cross-engine reuse | LocalModelCache, LMCache integration, KV offloading | KVBM, Disaggregation (prefill/decode) | KV Cache Manager, Disaggregated architecture |
| **Performance Highlight** | -79% P99 latency, -19.2% mean latency | 15-20min → ~1min startup time | 30x throughput, 1.1M tokens/sec | Fastest time-to-value, cost-competitive |
| **Autoscaling** | LLM-tailored autoscaler | Token/Queue/GPU-based, Scale-to-zero | SLO-based, Real-time observability | Kubernetes HPA, Predictive scaling |
| **Hardware Support** | GPU (NVIDIA focus) | Multi-vendor | NVIDIA (optimized for Blackwell) | NVIDIA, AMD, Google TPU, Intel XPU |
| **Model Size Support** | Not specified explicitly | Large models (LocalModelCache optimized) | Tested with DeepSeek-R1, Llama 3.1 405B | DeepSeek-R1, Large MoE models |
| **Production Status** | Production at ByteDance | Wide adoption (Red Hat OpenShift AI, etc.) | Cloud providers (AWS, GCP, Azure, OCI) | Production-ready (v0.3+) |

---

## So sánh chi tiết

### 1. Routing Mechanisms

#### AIBrix: Champion of Routing Diversity

**Điểm mạnh**:
- **8+ routing strategies** - nhiều nhất trong 4 platforms
- **Prefix-cache aware routing** - tối ưu cho cache hits
- **Intelligent least-GPU-memory routing** - proven -79% P99 latency

**Use cases phù hợp**:
- Multi-turn conversations cần maximize cache reuse
- Workloads với diverse request patterns
- Production environments cần fine-tuning routing behavior

**Hạn chế**:
- Complexity trong configuration
- Cần expertise để chọn strategy phù hợp

#### KServe: Production-Grade Flexibility

**Điểm mạnh**:
- **InferenceGraph CRD** - powerful workflow orchestration
- **Canary và A/B testing** native support
- Transformer chaining cho pre/post-processing

**Use cases phù hợp**:
- Gradual rollouts và testing
- Complex multi-step inference pipelines
- Enterprise deployments cần governance

**Hạn chế**:
- Routing strategies ít chi tiết hơn AIBrix
- Cần Istio/Service Mesh cho advanced features

#### NVIDIA Dynamo: SLA-Centric Performance

**Điểm mạnh**:
- **SLA-based planner** - performance guarantees
- **Grove multinode orchestration** - datacenter scale
- Backend-aware routing (vLLM vs TensorRT-LLM vs SGLang)

**Use cases phù hợp**:
- Workloads với strict SLA requirements
- Multi-backend deployments
- NVIDIA hardware deployments

**Hạn chế**:
- Less granular routing options vs AIBrix
- Tối ưu nhất cho NVIDIA GPUs

#### LLM-D: AI-Driven Intelligence

**Điểm mạnh**:
- **Predicted latency balancing** - AI-driven routing
- **Wide EP scheduling** cho MoE models
- Adaptive learning từ historical data

**Use cases phù hợp**:
- DeepSeek và MoE models
- Multi-accelerator environments (AMD, NVIDIA, TPU)
- Cost-optimization focus

**Hạn chế**:
- Newer platform, less battle-tested
- Prediction accuracy depends on training data

### 2. Caching Strategies

#### AIBrix: Cross-Engine Innovation

**Điểm mạnh**:
- **Cross-Engine KV Reuse** - unique feature
- Distributed KV Cache Runtime
- Tight integration với vLLM

**Performance**:
- Contributes to -79% P99 latency improvement
- Efficient cache hit routing

**Hạn chế**:
- vLLM-centric (limited multi-engine support thực tế)

#### KServe: Two-Tier Caching Excellence

**Điểm mạnh**:
- **LocalModelCache**: 15-20min → ~1min startup - game changer
- **LMCache integration**: Breakthrough cost reductions
- KV Cache Offloading cho long conversations

**Performance**:
- Dramatically reduced cold-start times
- Better GPU memory utilization

**Hạn chế**:
- LocalModelCache requires persistent storage planning
- KV offloading adds network overhead

#### NVIDIA Dynamo: Disaggregation Mastery

**Điểm mạnh**:
- **KVBM** (KV Buffer Management) - highly optimized
- **Disaggregation architecture** (v0.4): 4x performance boost
- Separate prefill/decode caching

**Performance**:
- 4x faster inference on Blackwell
- Reduced memory fragmentation

**Hạn chế**:
- Disaggregation adds architectural complexity
- Optimal performance on NVIDIA hardware only

#### LLM-D: Centralized Coordination

**Điểm mạnh**:
- **KV Cache Manager** - centralized visibility
- Prefix cache routing optimization
- Disaggregated prefill/decode caching

**Performance**:
- Better prefix cache routing (v0.3)
- Cost-competitive caching strategy

**Hạn chế**:
- Centralized manager có thể thành bottleneck at extreme scale
- Newer implementation, less optimization history

### 3. Performance Comparison

| **Metric** | **AIBrix** | **KServe** | **NVIDIA Dynamo** | **LLM-D** |
|-----------|-----------|-----------|------------------|----------|
| **Latency Improvement** | -79% P99, -19.2% mean | Startup: 15-20min → ~1min | 4x faster (v0.4 disaggregation) | Fastest time-to-value |
| **Throughput** | Not specified | Not specified | **30x boost**, 1.1M tokens/sec | Competitive per dollar |
| **Hardware Efficiency** | Least-GPU-memory routing | GPU utilization-based autoscaling | Blackwell optimizations | Multi-vendor optimization |
| **Scalability** | Production at ByteDance | Wide enterprise adoption | Datacenter-scale (GB200 NVL72) | Kubernetes-native scale |
| **Cold Start** | Not specified | **Best-in-class** (~1min) | Fast (KVBM optimization) | Standard K8s pod startup |

**Winner by Category**:
- **Latency**: AIBrix (P99) / Dynamo (absolute with Blackwell)
- **Throughput**: NVIDIA Dynamo (1.1M tokens/sec record)
- **Cold Start**: KServe (LocalModelCache innovation)
- **Cost Efficiency**: LLM-D (multi-vendor, competitive per dollar)

### 4. Ưu điểm và Nhược điểm

#### AIBrix

**Ưu điểm** ✅:
1. **Routing đa dạng nhất** - 8+ strategies, fine-grained control
2. **Proven production performance** - ByteDance deployment
3. **Cross-engine KV reuse** - unique capability
4. **vLLM integration** - tight coupling với leading engine
5. **Cost-effective** - significant latency improvements

**Nhược điểm** ❌:
1. **vLLM lock-in** - limited multi-engine support
2. **Configuration complexity** - nhiều options cần expertise
3. **Documentation** - đang phát triển, chưa comprehensive như KServe
4. **Community** - smaller vs KServe/NVIDIA ecosystems
5. **Model size support unclear** - not explicitly documented

#### KServe

**Ưu điểm** ✅:
1. **CNCF maturity** - production-proven, wide adoption
2. **Framework agnostic** - TensorFlow, PyTorch, XGBoost, etc.
3. **LocalModelCache innovation** - best-in-class cold start
4. **Serverless capabilities** - scale-to-zero, cost savings
5. **Enterprise features** - canary, A/B testing, governance
6. **Strong ecosystem** - Red Hat, Kubeflow, etc.

**Nhược điểm** ❌:
1. **Complexity** - requires Kubernetes + Istio/Service Mesh expertise
2. **Overhead** - more components vs leaner alternatives
3. **Routing granularity** - less detailed than AIBrix
4. **Performance** - not the absolute leader (vs Dynamo)
5. **Resource consumption** - heavier footprint

#### NVIDIA Dynamo

**Ưu điểm** ✅:
1. **Performance leader** - 30x throughput, 1.1M tokens/sec record
2. **NVIDIA optimization** - best-in-class for NVIDIA GPUs (Blackwell, etc.)
3. **Disaggregation architecture** - 4x performance boost (v0.4)
4. **Multi-backend support** - vLLM, SGLang, TensorRT-LLM
5. **Cloud provider backing** - AWS, GCP, Azure, OCI integration
6. **SLA-based autoscaling** - production-grade reliability

**Nhược điểm** ❌:
1. **NVIDIA hardware bias** - optimal performance only on NVIDIA GPUs
2. **Newer platform** - less maturity vs KServe
3. **Complexity** - disaggregation adds operational overhead
4. **Vendor lock-in risk** - despite open-source, NVIDIA-centric
5. **Documentation** - still evolving, some gaps
6. **Multi-accelerator support** - limited vs LLM-D

#### LLM-D

**Ưu điểm** ✅:
1. **True multi-vendor support** - NVIDIA, AMD, Google TPU, Intel XPU
2. **Wide Expert-Parallelism** - best for MoE models (DeepSeek-R1)
3. **Cost-competitive** - "performance per dollar" focus
4. **Kubernetes-native** - true K8s primitives, not wrapper
5. **Predicted latency routing** - AI-driven intelligence
6. **Red Hat backing** - enterprise support path
7. **Fast time-to-value** - simplified deployment

**Nhược điểm** ❌:
1. **Newer platform** - less production history vs KServe/AIBrix
2. **vLLM dependency** - single engine focus (like AIBrix)
3. **Smaller community** - still building ecosystem
4. **Documentation gaps** - evolving, not comprehensive yet
5. **Prediction accuracy** - latency prediction needs tuning
6. **Centralized Cache Manager** - potential bottleneck at extreme scale

---

## Use Cases và Model Size Support

### 1. Use Cases Phổ Biến

#### AIBrix

**Ideal Use Cases**:
1. **Multi-turn conversational AI**:
   - Chat applications với long context
   - Prefix cache routing tối ưu cache hits
   - Ví dụ: Customer support chatbots, AI assistants

2. **High-density LoRA deployments**:
   - Serving hàng nghìn LoRA adapters
   - Production tại ByteDance cho diverse business units

3. **Cost-sensitive vLLM deployments**:
   - Maximize efficiency của vLLM infrastructure
   - Least-GPU-memory routing giảm costs

4. **Variable workload patterns**:
   - Multiple routing strategies cho different request types
   - Dynamic optimization based on load

**Real-world Example**: ByteDance production deployment serving multiple business use cases

**Nguồn**:
- AIBrix Blog: https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/
- vLLM Blog: https://blog.vllm.ai/2025/02/21/aibrix-release.html

#### KServe

**Ideal Use Cases**:
1. **Enterprise ML/AI platform**:
   - Unified serving cho LLMs + traditional ML
   - Governance, compliance, audit trails
   - Ví dụ: Red Hat OpenShift AI deployments

2. **Gradual model rollouts**:
   - Canary deployments cho new model versions
   - A/B testing với traffic splitting
   - Risk mitigation cho production updates

3. **Multi-framework environments**:
   - TensorFlow, PyTorch, Scikit-Learn cùng infrastructure
   - Không muốn lock-in vào single framework

4. **Serverless AI/ML**:
   - Scale-to-zero cho cost optimization
   - Burst workloads
   - Development/staging environments

5. **Batch + Real-time hybrid**:
   - Batch inference cho large datasets
   - Real-time scoring cho applications

**Real-world Examples**:
- Red Hat OpenShift AI (evaluations: https://www.redhat.com/en/blog/evaluating-llm-inference-performance-red-hat-openshift-ai)
- Enterprise Kubernetes environments
- MLPerf benchmarks (Llama 3.1 405B): https://mlcommons.org/2025/04/llm-inference-v5/

**Nguồn**:
- KServe Documentation
- Red Hat Developer: https://developers.redhat.com/articles/2024/03/15/empower-conversational-ai-scale-kserve

#### NVIDIA Dynamo

**Ideal Use Cases**:
1. **Reasoning AI models**:
   - DeepSeek-R1 style models
   - Chain-of-thought reasoning
   - Complex multi-step inference

2. **Datacenter-scale deployments**:
   - Large GPU clusters (GB200 NVL72)
   - Cloud provider AI services (AWS, GCP, Azure, OCI)
   - High-throughput requirements (1M+ tokens/sec)

3. **SLA-critical applications**:
   - Financial services
   - Healthcare AI
   - Real-time decision systems
   - SLA-based autoscaling

4. **Mixture-of-Experts (MoE) optimization**:
   - DeepSeek R1, Llama 4
   - Multi-backend coordination (vLLM + TensorRT-LLM)

5. **NVIDIA hardware environments**:
   - Blackwell, Hopper GPU clusters
   - NVIDIA DGX systems
   - NVLink-optimized workloads

**Real-world Examples**:
- Cloud provider AI inference services
- Industry-first 1.1M tokens/sec record (Signal65 research)
- DeepSeek-R1 production deployments

**Nguồn**:
- NVIDIA Dynamo Blog: https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- GB200 + Dynamo: https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/
- Cloud Integration: https://blogs.nvidia.com/blog/think-smart-dynamo-ai-inference-data-center/

#### LLM-D

**Ideal Use Cases**:
1. **DeepSeek-style MoE models**:
   - Wide Expert-Parallelism deployment
   - DeepSeek-R1, DeepSeek-V3.2-Exp
   - Sparse attention for long-context

2. **Multi-vendor hardware environments**:
   - Mixed AMD + NVIDIA clusters
   - Google TPU deployments
   - Cost optimization across accelerators

3. **Cost-conscious deployments**:
   - "Performance per dollar" optimization
   - Non-NVIDIA hardware (AMD MI300X)
   - Budget-constrained environments

4. **Kubernetes-native organizations**:
   - Already using K8s extensively
   - Want true K8s primitives (not wrappers)
   - GitOps workflows

5. **Prefill/Decode disaggregation workloads**:
   - Long input prompts
   - Short output generation
   - Optimize TTFT và TPOT separately

**Real-world Examples**:
- AMD MI300X cluster deployments (ROCm Blog)
- Red Hat AI platform integration
- DeepSeek production serving

**Nguồn**:
- LLM-D Website: https://llm-d.ai/
- Red Hat: https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing
- AMD ROCm: https://rocm.blogs.amd.com/artificial-intelligence/llm-d-distributed/README.html
- DeepSeek Scaling: https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep

### 2. Model Size Support

#### Small Models (< 10B parameters)

**Tất cả platforms đều support tốt**, nhưng có sự khác biệt:

| Platform | Support Level | Notes |
|---------|--------------|-------|
| **AIBrix** | ⭐⭐⭐⭐⭐ | Efficient routing, overhead thấp cho small models |
| **KServe** | ⭐⭐⭐⭐⭐ | Serverless scale-to-zero ideal cho small models |
| **NVIDIA Dynamo** | ⭐⭐⭐⭐ | Overkill cho small models, overhead không cần thiết |
| **LLM-D** | ⭐⭐⭐⭐⭐ | Kubernetes-native, efficient |

**Recommendation**: KServe hoặc LLM-D cho cost efficiency với scale-to-zero

#### Medium Models (10B - 30B parameters)

**Ví dụ**: Llama 2 13B, Mistral 7B, Qwen 14B

| Platform | Support Level | Notes |
|---------|--------------|-------|
| **AIBrix** | ⭐⭐⭐⭐⭐ | Sweet spot, routing strategies shine |
| **KServe** | ⭐⭐⭐⭐⭐ | LocalModelCache reduces startup time |
| **NVIDIA Dynamo** | ⭐⭐⭐⭐ | Good, nhưng chưa cần disaggregation |
| **LLM-D** | ⭐⭐⭐⭐⭐ | Multi-vendor support, cost-effective |

**Recommendation**: AIBrix cho vLLM users, KServe cho multi-framework

#### Large Models (30B - 70B parameters)

**Ví dụ**: Llama 2 70B, Llama 3 70B, DeepSeek 67B

| Platform | Support Level | Notes |
|---------|--------------|-------|
| **AIBrix** | ⭐⭐⭐⭐ | Efficient, prefix caching important |
| **KServe** | ⭐⭐⭐⭐⭐ | LocalModelCache critical (~1min startup), MLPerf tested |
| **NVIDIA Dynamo** | ⭐⭐⭐⭐⭐ | Optimized, disaggregation helps |
| **LLM-D** | ⭐⭐⭐⭐⭐ | Wide EP, multi-vendor hardware options |

**Evidence**:
- KServe: MLPerf Llama 2 Chat 70B benchmarks (https://mlcommons.org/2025/04/llm-inference-v5/)
- NVIDIA Dynamo: Optimized cho Llama models
- LLM-D: Red Hat testing với large models

**Recommendation**:
- NVIDIA hardware → NVIDIA Dynamo
- Multi-vendor → LLM-D
- Kubernetes enterprise → KServe

#### Very Large Models (70B - 200B parameters)

**Ví dụ**: Llama 3.1 70B-405B, Mixtral 8x22B, Grok-1

| Platform | Support Level | Notes |
|---------|--------------|-------|
| **AIBrix** | ⭐⭐⭐ | Possible, nhưng chưa documented extensively |
| **KServe** | ⭐⭐⭐⭐ | Proven với Llama 3.1 405B (MLPerf) |
| **NVIDIA Dynamo** | ⭐⭐⭐⭐⭐ | **Best choice**, tested với 405B, disaggregation critical |
| **LLM-D** | ⭐⭐⭐⭐ | Wide EP helps, multi-node essential |

**Evidence**:
- KServe: MLPerf Llama 3.1 405B Instruct benchmarks
- NVIDIA Dynamo: GB200 NVL72 optimizations cho 405B models
- Technical white paper available

**Recommendation**: NVIDIA Dynamo cho absolute performance, KServe cho flexibility

**Nguồn**:
- MLPerf v5.0: https://mlcommons.org/2025/04/llm-inference-v5/
- NVIDIA GB200: https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/

#### Extreme Scale Models (200B - 600B+ parameters)

**Ví dụ**: DeepSeek-R1 (671B MoE), Llama 4 (largest variants), GPT-4 scale

| Platform | Support Level | Notes |
|---------|--------------|-------|
| **AIBrix** | ⭐⭐ | Not designed for this scale, documentation missing |
| **KServe** | ⭐⭐⭐ | Possible với distributed serving, not optimized |
| **NVIDIA Dynamo** | ⭐⭐⭐⭐⭐ | **DeepSeek-R1 champion**, 30x throughput, 1.1M tokens/sec |
| **LLM-D** | ⭐⭐⭐⭐⭐ | **DeepSeek-R1 specialist**, Wide EP designed for this |

**Evidence**:
- **NVIDIA Dynamo**: 30x throughput boost cho DeepSeek-R1 on Blackwell (https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- **LLM-D**: Wide EP specifically cho DeepSeek-style MoEs (https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep)

**Special Considerations**:
- **Mixture-of-Experts (MoE)**: LLM-D Wide EP vs NVIDIA Dynamo performance
- **Disaggregation**: Critical at this scale (both support)
- **Hardware**: Multi-node essential
  - NVIDIA Dynamo: GB200 NVL72 (72 GPUs)
  - LLM-D: AMD MI300X clusters supported

**Recommendation**:
- NVIDIA hardware + budget → **NVIDIA Dynamo** (record performance)
- Multi-vendor + cost focus → **LLM-D** (AMD support)
- Both far superior to AIBrix/KServe at extreme scale

**Nguồn**:
- NVIDIA Dynamo DeepSeek-R1: https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- LLM-D Wide EP: https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
- AMD LLM-D: https://rocm.blogs.amd.com/artificial-intelligence/llm-d-distributed/README.html

### 3. Summary Matrix: Model Size vs Platform

```
Model Size Guide:

< 10B params:
├─ Best: KServe (serverless), LLM-D (K8s-native)
└─ Avoid: NVIDIA Dynamo (overkill)

10B - 30B params:
├─ Best: AIBrix (routing), KServe (multi-framework)
└─ Good: All platforms

30B - 70B params:
├─ NVIDIA GPU: NVIDIA Dynamo
├─ Multi-vendor: LLM-D
└─ Enterprise K8s: KServe

70B - 200B params:
├─ Best performance: NVIDIA Dynamo
├─ Best flexibility: KServe (MLPerf proven)
└─ Best cost: LLM-D (multi-vendor)

200B - 600B+ params (MoE):
├─ NVIDIA hardware: NVIDIA Dynamo (30x boost)
├─ AMD/Multi-vendor: LLM-D (Wide EP)
└─ Avoid: AIBrix, KServe (not optimized)
```

---

## Case Studies thực tế về Performance và SLA/SLO

Phần này trình bày các case studies thực tế từ production deployments, tập trung vào performance improvements, SLA/SLO achievement, và uptime metrics.

### 1. AIBrix tại ByteDance: Production Scale LLM Serving

#### Tổng quan Deployment
**Organization**: ByteDance (TikTok parent company)
**Timeline**: Started early 2024, production deployment 6+ months
**Scale**: Multiple business units across ByteDance
**Use cases**: Multi-turn conversations, high-density LoRA serving

#### Performance Metrics Achieved

**Latency Improvements**:
- **P99 Latency**: Giảm **79%** so với baseline
- **Mean Latency**: Giảm **19.2%**
- **Contributing factors**: LLM-tailored autoscaler + prefix-aware routing

**Throughput Enhancement**:
- **Throughput increase**: **50%** improvement
- **Token reuse**: Distributed KV cache tăng token reuse across nodes
- **Inference latency**: Giảm **70%** với distributed KV cache

**Cost Optimization**:
- **Cost reduction**: Up to **4.7x** lower costs trong low-traffic periods
- **Mechanism**: Dynamic adapter loading, scale-down overhead reduction
- **LoRA efficiency**: Thousands of LoRA adapters với high-density management

#### SLA/SLO Implementation

**SLO-Driven Optimization**:
- **SLO-driven GPU optimizer**: Dynamically adjusts resource allocations
- **Service guarantees**: Maintains SLA compliance across diverse workloads
- **Heterogeneous serving**: Optimizes cost efficiency while meeting performance targets

**Routing Intelligence**:
- **8+ routing strategies**: Chọn strategy phù hợp với workload characteristics
- **Prefix-cache routing**: Routes multi-turn chats tới pods với cached KV pairs
- **Load-aware routing**: Balances requests dựa trên pod capacity và busy time

#### Production Learnings

**Key Success Factors**:
1. **LLM-specific autoscaling**: Traditional autoscalers không hiệu quả cho LLM workloads
2. **Prefix cache awareness**: Critical cho multi-turn conversation performance
3. **Dynamic LoRA management**: Enables serving thousands of adapters efficiently
4. **GPU fault detection**: Proactive hardware failure handling

**Challenges Addressed**:
- High cold-start times cho large models
- Inefficient GPU utilization với traditional routing
- Cost inefficiency trong variable traffic patterns
- Complex multi-tenant LoRA serving

#### Business Impact

**Operational Efficiency**:
- Support multiple business units on shared infrastructure
- Reduced operational overhead với automated scaling
- Improved resource utilization across GPU clusters

**User Experience**:
- 79% reduction in tail latency = more consistent user experience
- 70% faster inference = more responsive applications
- High throughput = support more concurrent users

**Nguồn**:
- vLLM Blog: https://blog.vllm.ai/2025/02/21/aibrix-release.html
- AIBrix Paper: https://arxiv.org/html/2504.03648v1
- AIBrix Blogs: https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/

---

### 2. KServe trên Red Hat OpenShift AI: Enterprise Model Serving

#### Tổng quan Deployment
**Organization**: Red Hat customers (various enterprises)
**Platform**: Red Hat OpenShift AI (formerly OpenShift Data Science)
**Infrastructure**: AWS (self-managed installer-provisioned infrastructure)
**Release**: Generally available December 2023 (release 2.5)

#### Performance Testing Results

**Model Loading Performance** (204 models deployed):
- **Fastest load**: 36 seconds
- **Slowest load**: 83 seconds
- **90th percentile**: Less than 63 seconds
- **Performance variance**: Factor of 2.3 between fastest and slowest

**Testing Methodology**:
- **Automation**: CI automation, no human interaction
- **Transparency**: Results published publicly, reproducible
- **Consistency**: Single-model performance testing across releases
- **Regression detection**: Verifies no performance degradation

#### LLM Inference Performance Benchmarks

**Test Configuration**:
- **Stack**: KServe + Caikit + TGIS (Text Generation Inference Server)
- **Platform**: OpenShift Serverless + OpenShift Service Mesh
- **Tool**: llm-load-test with varying concurrent threads
- **Duration**: 4-minute load tests
- **Metric**: Time Per Output Token (TPOT) as primary latency measure

**Infrastructure**:
- **Cloud**: AWS (EC2 instances)
- **Autoscaling**: ReplicaSets for model copy serving
- **Load balancing**: Automatic request distribution

#### SLA/SLO Features

**Service Level Capabilities**:
- **Canary deployments**: Controlled model version updates, risk mitigation
- **Advanced metrics**: Real-time performance insights for SLA monitoring
- **Concurrent serving**: Multiple models without resource conflicts
- **Health checks**: Proactive failure detection and recovery

**Observability Stack**:
- **Monitoring**: Integrated metrics collection
- **Alerting**: SLO violation notifications
- **Logging**: Comprehensive request tracing
- **Dashboards**: Performance visualization

#### LocalModelCache Innovation

**Cold-Start Problem Solution**:
- **Before**: 15-20 minutes model loading time
- **After**: ~1 minute with LocalModelCache
- **Improvement**: **93-95% reduction** in startup time
- **Impact**: Faster autoscaling, better SLA compliance

**Cache Management**:
- **Storage backends**: S3, GCS, Azure Blob, PVC
- **Lifecycle**: Automatic cache invalidation and refresh
- **Multi-model**: Shared cache across model versions

#### Production Reliability

**High Availability Features**:
- **Kubernetes-native**: Leverages K8s reliability primitives
- **Serverless scaling**: Scale-to-zero and rapid scale-up
- **Multi-replica**: Horizontal scaling for redundancy
- **Service mesh**: Circuit breaking, retry logic, timeout handling

**Continuous Validation**:
- **Regression testing**: Automated performance validation per release
- **Scale testing**: Validates platform capacity limits
- **Long-running tests**: Stability verification over extended periods

#### Enterprise Adoption

**Red Hat Customer Base**:
- Wide adoption across financial services, healthcare, retail
- Integration với existing OpenShift deployments
- Enterprise support và SLA guarantees từ Red Hat

**MLPerf Benchmarks**:
- **Llama 3.1 405B Instruct**: Industry-standard benchmark results
- **Llama 2 Chat 70B**: Production-scale validation
- **Framework**: MLPerf Inference v5.0 compliance

**Nguồn**:
- Red Hat Blog (Continuous Performance Validation): https://www.redhat.com/en/blog/continuous-performance-and-scale-validation-red-hat-openshift-ai-model-serving-stack
- Red Hat Blog (LLM Evaluation): https://www.redhat.com/en/blog/evaluating-llm-inference-performance-red-hat-openshift-ai
- Red Hat Developer: https://developers.redhat.com/articles/2024/03/15/empower-conversational-ai-scale-kserve
- MLPerf v5.0: https://mlcommons.org/2025/04/llm-inference-v5/

---

### 3. NVIDIA Dynamo trên Cloud Providers: Record-Breaking Performance

#### Tổng quan Deployment
**Cloud Partners**: AWS, Google Cloud, Microsoft Azure, Oracle Cloud Infrastructure
**Launch**: GTC 2025
**Target**: Datacenter-scale reasoning AI models
**License**: Apache 2.0 (open-source)

#### AWS Integration Case Study

**Amazon EKS Deployment**:
- **Instance types**: P6 instances (NVIDIA Blackwell GPUs)
- **Support services**:
  - Amazon S3: Model storage
  - Elastic Fabric Adapter (EFA): Low-latency inter-node communication
  - Amazon EKS: Kubernetes orchestration
  - Network Load Balancer: Traffic distribution
  - Amazon CloudWatch / Prometheus: Observability

**Performance Results**:
- **GPU utilization**: Enhanced through intelligent scheduling
- **Request throughput per dollar**: Increased significantly
- **Margin growth**: Sustainable for production-scale AI workloads

#### Record-Breaking DeepSeek-R1 Performance

**Configuration**:
- **Model**: DeepSeek-R1 (671B parameters, MoE)
- **Hardware**: NVIDIA GB200 NVL72 (72 Blackwell Ultra GPUs)
- **Framework**: NVIDIA Dynamo

**Performance Achievements**:
- **Throughput boost**: Up to **30x** improvement vs baseline
- **Aggregate throughput**: Industry-first **1.1 million tokens/second**
- **Source**: Signal65 research paper by Russ Fellows, principal analyst

**Technical Innovations**:
- **Disaggregation**: Separate prefill and decode phases
- **KVBM**: Optimized KV-cache buffer management
- **SLA-based planner**: Performance guarantees through intelligent routing

#### Llama Model Performance

**Llama 70B on NVIDIA Hopper**:
- **Throughput increase**: More than **2x** vs baseline
- **Configuration**: Standard vLLM backend with Dynamo orchestration

**Llama 3.1 405B**:
- **Target model**: One of the largest open models
- **Optimization**: GB200 NVL72 hardware + Dynamo software
- **Result**: Production-ready serving at scale

#### Version 0.4 Breakthrough

**4x Performance Improvement**:
- **Architecture**: Disaggregation on NVIDIA Blackwell
- **Prefill optimization**: Dedicated prefill workers
- **Decode optimization**: Separate decode workers với cached KV reuse
- **TTFT improvement**: Reduced time-to-first-token
- **TPOT predictability**: More consistent time-per-output-token

**SLO-Based Autoscaling**:
- **Real-time observability**: Continuous metrics collection
- **Predictive scaling**: Anticipates demand changes
- **SLA compliance**: Automatic resource adjustment to meet targets

#### Multi-Cloud Production Readiness

**Google Cloud**:
- **Platform**: AI Hypercomputer
- **Optimization**: Dynamo recipes for enterprise-scale LLM inference
- **Integration**: Native GCP services integration

**Microsoft Azure**:
- **Instance types**: ND GB200-v6 GPUs
- **Service**: Azure Kubernetes Service (AKS)
- **Capability**: Multi-node LLM inference at scale

**Oracle Cloud Infrastructure (OCI)**:
- **Infrastructure**: OCI Superclusters
- **Feature**: Multi-node inferencing with Dynamo
- **Scale**: Datacenter-level deployments

#### SLA/SLO Implementation

**Service Level Features**:
- **SLA-based planner**: Routes requests to ensure SLA compliance
- **Priority-aware routing**: Different SLAs for different request classes
- **Performance guarantees**: Predictable latency under load
- **Observability**: Real-time SLA violation detection

**Production Reliability**:
- **etcd + NATS**: Reliable messaging infrastructure
- **Health checks**: Continuous worker health monitoring
- **Fault tolerance**: Automatic failover mechanisms
- **Graceful degradation**: Maintains service under partial failures

#### Mixture-of-Experts (MoE) Optimization

**GB200 NVL72 + Dynamo Synergy**:
- **Target models**: DeepSeek R1, Llama 4 (large variants)
- **Compounding effect**: Hardware + software optimization
- **Technical white paper**: Available for large-scale GPU cluster deployments
- **Expert parallelism**: Efficient expert routing and caching

#### Enterprise Adoption Indicators

**Cloud Provider Integration**:
- All major cloud providers offer managed Dynamo services
- Native integration với cloud-specific services (S3, EFA, CloudWatch, etc.)
- Enterprise support channels through cloud vendors

**Production Readiness**:
- Open-source (Apache 2.0) = transparency, no vendor lock-in
- Multi-backend support (vLLM, SGLang, TensorRT-LLM) = flexibility
- Datacenter-scale proven (GB200 NVL72 clusters)

**Nguồn**:
- NVIDIA Blog (Cloud Integration): https://blogs.nvidia.com/blog/think-smart-dynamo-ai-inference-data-center/
- NVIDIA Blog (v0.4 Release): https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/
- NVIDIA Blog (GB200 + MoE): https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/
- AWS Integration: https://developer.nvidia.com/blog/nvidia-dynamo-adds-support-for-aws-services-to-deliver-cost-efficient-inference-at-scale/
- AWS EKS Blueprint: https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/GPUs/nvidia-dynamo

---

### 4. AMD MI300X với vLLM/LLM-D: Multi-Vendor Success

#### Tổng quan Deployment
**Hardware**: AMD Instinct MI300X GPUs
**Software**: vLLM (LLM-D integration)
**Focus**: Cost-competitive alternative to NVIDIA
**Status**: Production-ready with Red Hat backing

#### Hardware Specifications Advantage

**Memory Leadership**:
- **MI300X memory**: 192 GB HBM3
- **H100 memory**: 80 GB (standard) / 94 GB (enhanced)
- **Advantage**: **2.4x memory capacity** vs standard H100

**Bandwidth Superiority**:
- **MI300X bandwidth**: 5.3 TB/s
- **H100 bandwidth**: 3.3-3.9 TB/s
- **Advantage**: **1.6x higher bandwidth**

#### Performance Benchmarks

**vLLM on MI300X vs TGI**:

**Llama 3.1 405B Performance**:
- **Throughput**: **1.5x higher** than Text Generation Inference (TGI)
- **TTFT (Time to First Token)**: **1.7x faster** than TGI
- **Impact**: Better user experience cho large models

**Llama 3.1 70B Performance**:
- **Throughput**: **1.8x higher** than TGI
- **TTFT**: **5.1x faster** than TGI (dramatic improvement)
- **Use case**: Production chatbots, conversational AI

#### MI300X vs H100 Direct Comparison

**Memory-Bound Scenarios** (MI300X advantages):
- **Long output sequences**: MI300X outperforms due to memory capacity
- **Strict latency constraints**: Better TPOT (Time Per Output Token)
- **Request throughput**: Nearly **2x** higher trong certain workloads
- **Latency reduction**: Significantly lower cho memory-intensive tasks

**Mixtral 8x7B Benchmarks**:
- **Performance uplift**: 22% to **194%** (almost 3x) vs H100
- **Batch size range**: 1 to 1024
- **Consistency**: Strong performance across different batch sizes

#### Speculative Decoding Performance

**Speedup Achievements**:
- **Eager mode**: 1.26x to 1.71x speedup
- **Compile mode**: 1.75x to **2.99x speedup** (nearly 3x)
- **Batch size**: 1 (most challenging scenario)
- **Technique**: Draft model generates candidate tokens, main model verifies

#### Large Model Capacity Advantages

**Single-GPU Large Models**:
- **Llama 3.1 405B**: Can fit on single MI300X (192 GB)
- **DeepSeek V3/R1**: Feasible with MI300X memory
- **Small models (≤30B)**: Efficient in TP1 mode (no tensor parallelism overhead)

**Cost Benefits**:
- **Fewer nodes required**: Large memory = fewer GPUs needed
- **Infrastructure cost reduction**: Less networking overhead
- **System reliability**: Simpler topology = fewer failure points

#### LLM-D Integration

**Wide Expert-Parallelism**:
- **Target**: DeepSeek-style MoE models
- **AMD Support**: Production-ready on MI300X clusters
- **Red Hat backing**: Enterprise support path through Red Hat AI

**Predicted Latency Balancing**:
- **MI300X-specific tuning**: Routing optimized for MI300X characteristics
- **Adaptive learning**: Historical performance data improves routing

**Prefill/Decode Disaggregation**:
- **Memory advantage**: MI300X large memory benefits both phases
- **Prefill**: Handle large prompts efficiently
- **Decode**: Fast token generation with ample cache space

#### Production Deployment Insights

**AMD ROCm Ecosystem**:
- **Software stack**: ROCm (Radeon Open Compute)
- **vLLM support**: Full compatibility with AMD GPUs
- **Optimization**: Continuous performance improvements

**Cost-Performance Analysis**:
- **Price positioning**: AMD typically more competitive than NVIDIA
- **Performance per dollar**: Strong value proposition
- **TCO (Total Cost of Ownership)**: Lower due to fewer GPUs needed

#### SLA/SLO Considerations

**Latency Predictability**:
- **TPOT consistency**: Memory-bound workloads show stable latency
- **TTFT advantage**: 1.7x-5.1x faster = better SLA compliance
- **Throughput**: Higher throughput = more headroom for traffic spikes

**Reliability**:
- **Simpler deployments**: Fewer GPUs = fewer failure points
- **Memory capacity**: Reduces OOM (Out of Memory) failures
- **Production stability**: vLLM + MI300X extensively tested

#### Real-World Use Case: DeepSeek V3.2-Exp

**Configuration**:
- **Model**: DeepSeek-V3.2-Exp (sparse attention, long-context)
- **Platform**: Red Hat AI + vLLM + MI300X
- **Status**: Day 0 support, ready for experimentation

**Capabilities**:
- **Long-context inference**: Sparse attention mechanism
- **Memory efficiency**: MI300X capacity handles long contexts
- **Production-ready**: Red Hat enterprise support

#### Business Impact

**Multi-Vendor Strategy**:
- **Risk mitigation**: Not dependent on single GPU vendor
- **Price negotiation**: Competitive alternatives strengthen position
- **Innovation access**: Both AMD and NVIDIA innovations available

**Cost Optimization**:
- **Lower acquisition cost**: AMD competitive pricing
- **Fewer GPUs**: Large memory reduces cluster size
- **Power efficiency**: MI300X competitive power/performance ratio

**Performance Competitiveness**:
- **Memory-bound leadership**: MI300X wins in many LLM scenarios
- **Large model capability**: 405B models on single GPU
- **Throughput**: Nearly 2x in optimal configurations

**Nguồn**:
- AMD ROCm Blog (Best Practices): https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html
- vLLM Blog (AMD Serving): https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html
- AMD Technical Article: https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus.html
- AMD Speculative Decoding: https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html
- Valohai Analysis: https://valohai.com/blog/amd-gpu-performance-for-llm-inference/
- Red Hat DeepSeek: https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference

---

### 5. Key Metrics Summary: SLA/SLO trong Production LLM Serving

#### Standard SLA/SLO Metrics

**Latency Metrics**:

1. **Time to First Token (TTFT)**:
   - **Definition**: Time từ khi nhận request đến khi generate token đầu tiên
   - **Typical SLO**: 95% requests < 200ms (chatbots), < 500ms (batch processing)
   - **Impact**: User perception of responsiveness

2. **Time Per Output Token (TPOT)**:
   - **Definition**: Inter-token latency during generation
   - **Typical SLO**: 95% < 50ms (streaming chat), < 100ms (standard)
   - **Impact**: Streaming experience quality

3. **End-to-End Latency (E2EL)**:
   - **Definition**: Total time từ request đến complete response
   - **Typical SLO**: 95% < 2s (short outputs), < 10s (long outputs)
   - **Impact**: Overall user satisfaction

4. **P95/P99 Latency**:
   - **P95**: 95% requests nhanh hơn threshold
   - **P99**: 99% requests nhanh hơn threshold
   - **Typical**: P95 < 2x median, P99 < 4x median
   - **Impact**: SLA compliance, tail latency elimination

**Throughput Metrics**:

1. **Tokens per Second (TPS)**:
   - **Definition**: Total tokens generated per second
   - **Range**: 100s-1000s TPS (single GPU) to 1M+ TPS (cluster)
   - **Impact**: Cost per inference, concurrent users capacity

2. **Requests per Second (RPS)**:
   - **Definition**: Completed requests per second
   - **Typical**: 10-100 RPS (single GPU), 1000+ RPS (cluster)
   - **Impact**: User capacity, scalability

3. **Goodput**:
   - **Definition**: Requests/second meeting SLO targets
   - **Formula**: (Successful requests / Total requests) × RPS
   - **Typical Target**: > 95% of throughput meets SLO
   - **Impact**: Real-world effective performance

**Availability Metrics**:

1. **Uptime Percentages**:
   - **99.9% (three nines)**: 8.76 hours downtime/year, 43.8 min/month
   - **99.95%**: 4.38 hours downtime/year, 21.9 min/month
   - **99.99% (four nines)**: 52.6 minutes downtime/year, 4.38 min/month
   - **Target**: 99.9%-99.99% for enterprise LLM services

2. **Mean Time Between Failures (MTBF)**:
   - **Definition**: Average time between system failures
   - **Typical**: 720+ hours (30+ days) for production systems

3. **Mean Time to Recovery (MTTR)**:
   - **Definition**: Average time to restore service after failure
   - **Typical**: < 5 minutes (automated recovery), < 30 minutes (manual)

#### Case Study Performance Comparison

| **Platform** | **Latency Improvement** | **Throughput Improvement** | **Key Metric** |
|-------------|------------------------|---------------------------|----------------|
| **AIBrix** | P99: -79%, Mean: -19.2% | +50% | Cost: 4.7x reduction (low traffic) |
| **KServe** | Startup: 15-20min → 1min | N/A | Cold-start: 93-95% reduction |
| **NVIDIA Dynamo** | v0.4: 4x faster | 30x (DeepSeek-R1), 2x (Llama 70B) | Record: 1.1M tokens/sec |
| **AMD MI300X + vLLM** | TTFT: 1.7-5.1x faster | 1.5-1.8x | Throughput: Nearly 2x (memory-bound) |

#### Best Practices cho SLA/SLO Achievement

**1. Baseline Establishment**:
- Benchmark current performance (TTFT, TPOT, throughput)
- Identify bottlenecks (compute, memory, network, storage)
- Set realistic SLO targets dựa trên user requirements

**2. Monitoring và Observability**:
- Real-time metrics collection (Prometheus, CloudWatch)
- Dashboard visualization (Grafana)
- Alerting on SLO violations
- Request tracing (distributed tracing)

**3. Autoscaling Strategy**:
- Predictive scaling (dựa trên historical patterns)
- Reactive scaling (dựa trên current metrics)
- Scale-to-zero cho cost optimization (non-critical services)
- Warm pools cho fast scale-up

**4. Caching Optimization**:
- Prefix caching cho multi-turn conversations
- KV cache management cho memory efficiency
- Model caching cho fast cold-starts (KServe LocalModelCache)
- Response caching cho duplicate requests

**5. Routing Intelligence**:
- Prefix-aware routing (AIBrix)
- Predicted latency routing (LLM-D)
- SLA-based routing (NVIDIA Dynamo)
- Load balancing across replicas

**6. Hardware Selection**:
- Memory-bound workloads: AMD MI300X (192 GB)
- Compute-intensive: NVIDIA Blackwell/Hopper
- Cost-sensitive: AMD MI300X hoặc multi-tenant NVIDIA
- Extreme scale: GB200 NVL72 clusters

**7. Reliability Engineering**:
- Multi-replica deployments (horizontal scaling)
- Health checks và automatic failover
- Graceful degradation strategies
- Canary deployments cho updates (KServe)

**8. Continuous Optimization**:
- Regular benchmarking (KServe CI validation)
- A/B testing new configurations
- Performance regression detection
- Cost-performance analysis

#### Nguồn tổng hợp metrics:
- BentoML LLM Metrics Guide: https://bentoml.com/llm/inference-optimization/llm-inference-metrics
- Anyscale Metrics Documentation: https://docs.anyscale.com/llm/serving/benchmarking/metrics
- NVIDIA Benchmarking Concepts: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
- Galileo AI Reliability Metrics: https://galileo.ai/blog/llm-performance-metrics

---

## Kết luận và Khuyến nghị

### 1. Lựa Chọn Theo Scenario

#### Scenario 1: Startup/Small Team với Budget Constraints
**Recommendation**: **LLM-D** hoặc **KServe**

**Lý do**:
- **LLM-D**:
  - "Performance per dollar" focus
  - Multi-vendor hardware (AMD cheaper than NVIDIA)
  - Kubernetes-native, leverage existing K8s skills
  - Fast time-to-value

- **KServe**:
  - CNCF maturity, strong community support
  - Serverless scale-to-zero saves costs
  - Wide framework support (không lock-in)

**Tránh**: NVIDIA Dynamo (NVIDIA hardware premium), AIBrix (less documentation)

#### Scenario 2: Enterprise với NVIDIA Infrastructure
**Recommendation**: **NVIDIA Dynamo**

**Lý do**:
- Best-in-class performance trên NVIDIA GPUs (30x throughput)
- SLA-based autoscaling cho production reliability
- Cloud provider backing (AWS, GCP, Azure, OCI)
- GB200 NVL72 optimization cho latest hardware
- Technical support từ NVIDIA

**Alternative**: KServe (nếu cần multi-framework beyond LLMs)

#### Scenario 3: ByteDance-Style Multi-Business-Unit Platform
**Recommendation**: **AIBrix**

**Lý do**:
- High-density LoRA support (thousands of adapters)
- 8+ routing strategies cho diverse workloads
- Prefix-cache routing cho multi-turn conversations
- Production-proven tại ByteDance
- vLLM optimization (nếu committed to vLLM)

**Note**: Requires vLLM commitment và routing expertise

#### Scenario 4: DeepSeek-R1 hoặc Large MoE Models
**Recommendation**: **NVIDIA Dynamo** (NVIDIA GPUs) hoặc **LLM-D** (Multi-vendor)

**Lý do**:
- **NVIDIA Dynamo**:
  - 30x throughput cho DeepSeek-R1
  - Industry record 1.1M tokens/sec
  - Disaggregation optimized cho MoE

- **LLM-D**:
  - Wide Expert-Parallelism designed cho MoE
  - AMD MI300X support (cost alternative)
  - Predicted latency balancing

**Tránh**: AIBrix, KServe (không optimized cho extreme scale MoE)

#### Scenario 5: Multi-Cloud/Hybrid Cloud Enterprise
**Recommendation**: **KServe**

**Lý do**:
- CNCF standard, cloud-agnostic
- Wide adoption (Red Hat, Kubeflow, etc.)
- Framework flexibility (LLM + traditional ML)
- Enterprise governance features (canary, A/B testing)
- Kubernetes-native (standard across clouds)

**Alternative**: LLM-D (nếu focus LLM-only + multi-vendor GPUs)

#### Scenario 6: Research/Academia với Diverse Accelerators
**Recommendation**: **LLM-D**

**Lý do**:
- True multi-vendor support (NVIDIA, AMD, Google TPU, Intel)
- Open-source, community-driven
- Kubernetes-native cho flexible experimentation
- DeepSeek optimizations cho cutting-edge research
- Cost-effective (AMD, TPU options)

#### Scenario 7: Gradual Migration từ Traditional ML tới LLM
**Recommendation**: **KServe**

**Lý do**:
- Unified platform cho ML + LLM
- Không cần separate infrastructure
- Gradual adoption path
- Team expertise reuse (single platform)
- Mature ecosystem

### 2. Decision Tree

```
START: Chọn LLM Inference Engine

Model Size?
├─ < 70B params
│  ├─ Multi-framework needed? → YES → KServe
│  ├─ vLLM committed? → YES → AIBrix
│  └─ Cost-focused? → YES → LLM-D
│
├─ 70B - 200B params
│  ├─ NVIDIA GPUs? → YES → NVIDIA Dynamo
│  ├─ Multi-vendor GPUs? → YES → LLM-D
│  └─ Enterprise K8s? → YES → KServe
│
└─ 200B+ params (MoE)
   ├─ NVIDIA hardware + budget → NVIDIA Dynamo (30x boost)
   └─ AMD/Multi-vendor → LLM-D (Wide EP)

Special Considerations:
├─ LoRA-heavy workloads → AIBrix
├─ Serverless needed → KServe
├─ SLA-critical → NVIDIA Dynamo
└─ Multi-cloud → KServe
```

### 3. Technology Maturity Assessment

| Platform | Maturity | Community | Production Readiness |
|---------|----------|-----------|---------------------|
| **AIBrix** | 🟡 Emerging | 🟡 Growing | 🟢 Production at ByteDance |
| **KServe** | 🟢 Mature | 🟢 Large (CNCF) | 🟢 Wide adoption |
| **NVIDIA Dynamo** | 🟡 New (2025) | 🟢 NVIDIA-backed | 🟢 Cloud providers |
| **LLM-D** | 🟡 Emerging | 🟡 Growing | 🟢 Red Hat backing |

**Maturity Implications**:
- **KServe**: Safest choice, proven track record
- **NVIDIA Dynamo**: Cutting-edge performance, rapid evolution
- **AIBrix**: Production-proven but smaller ecosystem
- **LLM-D**: Modern design, multi-vendor focus

### 4. Final Recommendations

#### For Most Organizations:
**Start with KServe** - mature, flexible, well-supported

**Consider upgrading to**:
- **NVIDIA Dynamo** nếu performance critical + NVIDIA infrastructure
- **LLM-D** nếu multi-vendor GPUs + cost optimization
- **AIBrix** nếu vLLM-committed + complex routing needs

#### For Specific Workloads:

1. **High-throughput production LLM serving**: NVIDIA Dynamo
2. **Multi-framework AI platform**: KServe
3. **vLLM optimization với complex routing**: AIBrix
4. **DeepSeek/MoE models**: NVIDIA Dynamo (NVIDIA) hoặc LLM-D (multi-vendor)
5. **Cost-conscious multi-cloud**: LLM-D
6. **Enterprise governance**: KServe

### 5. Future Outlook

**Emerging Trends** (based on latest releases):

1. **Disaggregation** (NVIDIA Dynamo 0.4, LLM-D):
   - Separate prefill/decode becoming standard
   - 4x performance improvements proven
   - Expect adoption across all platforms

2. **Multi-Accelerator Support** (LLM-D leading):
   - AMD MI300X production-ready
   - Google TPU, Intel XPU support growing
   - NVIDIA dominance being challenged

3. **AI-Driven Routing** (LLM-D predicted latency):
   - Intelligent routing beyond static rules
   - Adaptive learning from workload patterns
   - Expect AIBrix, KServe to follow

4. **Cache Innovations** (KServe LocalModelCache, AIBrix Cross-Engine):
   - Cold-start problem largely solved (15min → 1min)
   - Cross-engine KV reuse emerging
   - Distributed cache coordination maturing

5. **MoE Optimization** (NVIDIA Dynamo, LLM-D):
   - Wide Expert-Parallelism critical
   - DeepSeek-R1 driving innovation
   - Expect specialized MoE engines

**Platforms to Watch**:
- **NVIDIA Dynamo**: Rapid iteration (v0.4 already 4x boost), NVIDIA backing
- **LLM-D**: Multi-vendor champion, Red Hat enterprise path
- **KServe**: CNCF maturity, steady innovation (LMCache integration)
- **AIBrix**: ByteDance production learnings, vLLM synergy

---

## Tài liệu tham khảo

### AIBrix

1. **Official Documentation**:
   - https://aibrix.readthedocs.io/latest/index.html

2. **GitHub Repository**:
   - https://github.com/vllm-project/aibrix

3. **Academic Paper**:
   - "AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure" - https://arxiv.org/html/2504.03648v1

4. **Blog Posts**:
   - vLLM Official Blog: https://blog.vllm.ai/2025/02/21/aibrix-release.html
   - AIBrix Blogs: https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/
   - AIBrix v0.1.0 Release: https://aibrix.github.io/posts/2024-11-12-v0.1.0-release/

5. **Technical Analysis**:
   - Gateway Routing Documentation: https://aibrix.readthedocs.io/latest/features/gateway-plugins.html
   - The New Stack Analysis: https://thenewstack.io/a-look-at-aibrix-an-open-source-llm-inference-platform/
   - Run It On Cloud Blog: https://aymen-segni.com/index.php/2025/03/14/aibrix-revolutionizing-llm-inference-production-deployments/
   - Medium Analysis: https://ai.gopubby.com/unlocking-scalable-llm-inference-introducing-aibrix-for-vllm-908e3d64bf74
   - CTOL Digital Solutions: https://www.ctol.digital/news/aibrix-brings-scalable-cost-efficient-llm-inference-kubernetes/

### KServe

1. **Official Documentation**:
   - https://kserve.github.io/website/docs/intro

2. **Red Hat Integration**:
   - "Empower conversational AI at scale with KServe": https://developers.redhat.com/articles/2024/03/15/empower-conversational-ai-scale-kserve
   - "Evaluating LLM inference performance on Red Hat OpenShift AI": https://www.redhat.com/en/blog/evaluating-llm-inference-performance-red-hat-openshift-ai
   - "Why vLLM is the best choice for AI inference today": https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today

3. **Performance & Benchmarks**:
   - "LMCache Turbocharges Enterprise LLM Inference Frameworks": https://blog.lmcache.ai/en/2025/05/16/how-lmcache-turbocharges-enterprise-llm-inference-frameworks/
   - MLPerf Inference v5.0: https://mlcommons.org/2025/04/llm-inference-v5/
   - "LLM-Inference-Bench" (arXiv): https://arxiv.org/html/2411.00136v1
   - ACM SC '24 Workshop: https://dl.acm.org/doi/10.1109/SCW63240.2024.00178

4. **Technical Resources**:
   - LLM Inference Speed Benchmarks: https://sparecores.com/article/llm-inference-speed
   - Kubernetes-Based LLM Inference Architectures Overview: https://rudeigerc.dev/posts/kubernetes-based-llm-inference-architectures-an-overview/

### NVIDIA Dynamo

1. **Official Documentation**:
   - https://docs.nvidia.com/dynamo/latest/index.html
   - NVIDIA Developer Portal: https://developer.nvidia.com/dynamo

2. **GitHub Repository**:
   - https://github.com/ai-dynamo/dynamo

3. **NVIDIA Technical Blogs**:
   - "Introducing NVIDIA Dynamo" (Main announcement): https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
   - "Dynamo 0.4 Delivers 4x Faster Performance": https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/
   - "GB200 NVL72 and Dynamo Boost MoE Performance": https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/
   - "AWS, Google, Microsoft and OCI Boost AI Inference": https://blogs.nvidia.com/blog/think-smart-dynamo-ai-inference-data-center/
   - "Dynamo Adds AWS Services Support": https://developer.nvidia.com/blog/nvidia-dynamo-adds-support-for-aws-services-to-deliver-cost-efficient-inference-at-scale/

4. **Technical Analysis**:
   - Medium - "Demystifying NVIDIA Dynamo": https://medium.com/byte-sized-ai/demystifying-nvidia-dynamo-a-high-performance-inference-framework-for-scalable-genai-f10be3d7032f
   - Collabnix Getting Started Guide: https://collabnix.com/getting-started-with-nvidia-dynamo-a-powerful-framework-for-distributed-llm-inference/

### LLM-D

1. **Official Website**:
   - https://llm-d.ai/
   - Blog: https://llm-d.ai/blog
   - Community Announcement: https://llm-d.ai/blog/llm-d-announce

2. **GitHub Repository**:
   - https://github.com/llm-d/llm-d

3. **Red Hat Integration**:
   - "llm-d: Kubernetes-native distributed inferencing": https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing
   - "Scaling DeepSeek-style MoEs with vLLM and llm-d using Wide EP": https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
   - "DeepSeek-V3.2-Exp on vLLM, Day 0": https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference

4. **Hardware Integration**:
   - AMD ROCm Blog: "AMD Integrates llm-d on AMD Instinct MI300X": https://rocm.blogs.amd.com/artificial-intelligence/llm-d-distributed/README.html

5. **Technical Resources**:
   - Kubernetes-Based LLM Inference Architectures Overview: https://rudeigerc.dev/posts/kubernetes-based-llm-inference-architectures-an-overview/

### General LLM Inference Resources

1. **Academic Papers**:
   - "LLM-Inference-Bench" (arXiv 2024): https://arxiv.org/html/2411.00136v1
   - "MoE-Inference-Bench" (arXiv 2024): https://arxiv.org/html/2508.17467
   - AIBrix Paper (arXiv): https://arxiv.org/html/2504.03648v1

2. **Industry Benchmarks**:
   - MLPerf Inference v5.0 (April 2025): https://mlcommons.org/2025/04/llm-inference-v5/
   - Spare Cores LLM Benchmarks: https://sparecores.com/article/llm-inference-speed

3. **Technical Blogs**:
   - LMCache Blog: https://blog.lmcache.ai/
   - The New Stack: https://thenewstack.io/
   - Red Hat Developer: https://developers.redhat.com/

---

## Phụ lục: Quick Reference

### Command Cheatsheet

#### AIBrix
```yaml
# Gateway routing strategy example
routing_strategy: "prefix-cache-preble"
# or: random, least-request, throughput, least-kv-cache, etc.
```

#### KServe
```yaml
# InferenceService với LocalModelCache
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-model
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "s3://models/llama-70b"
      resources:
        limits:
          nvidia.com/gpu: "4"
```

#### NVIDIA Dynamo
```bash
# Quickstart example
docker run -p 8000:8000 \
  nvidia/dynamo:latest \
  --model Qwen/Qwen3-0.6B
```

#### LLM-D
```yaml
# Wide EP deployment for DeepSeek
apiVersion: llm-d.ai/v1alpha1
kind: InferenceDeployment
metadata:
  name: deepseek-r1
spec:
  model: deepseek-ai/deepseek-r1
  expertParallelism: 16  # Wide EP
  prefillReplicas: 4
  decodeReplicas: 8
```

### Performance Metrics Summary

| Metric | AIBrix | KServe | NVIDIA Dynamo | LLM-D |
|--------|--------|--------|---------------|-------|
| **P99 Latency** | -79% | N/A | 4x faster (v0.4) | Competitive |
| **Mean Latency** | -19.2% | N/A | N/A | N/A |
| **Throughput** | N/A | N/A | 30x, 1.1M tok/s | Per-dollar focus |
| **Cold Start** | N/A | 15-20min → 1min | Fast | Standard K8s |
| **Model Size** | Not specified | Up to 405B | Up to 671B (DeepSeek-R1) | Up to 671B |

---

**Tài liệu này được biên soạn dựa trên nghiên cứu các nguồn chính thức, blog kỹ thuật, academic papers và production deployments tính đến tháng 11 năm 2025.**

**Version**: 1.0
**Last Updated**: 2025-11-19
