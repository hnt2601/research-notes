# KServe: Hướng dẫn Triển khai LLM Chuyên sâu

## Mục lục

1. [Giới thiệu về KServe](#1-giới-thiệu-về-kserve)
2. [Kiến trúc KServe](#2-kiến-trúc-kserve)
3. [Các tính năng cốt lõi](#3-các-tính-năng-cốt-lõi)
4. [Scale-to-Zero Mechanism](#4-scale-to-zero-mechanism)
5. [Caching và Storage](#5-caching-và-storage)
6. [Routing với InferenceGraph](#6-routing-với-inferencegraph)
7. [Autoscaling](#7-autoscaling)
8. [Fallback và Error Handling](#8-fallback-và-error-handling)
9. [Triển khai LLM với KServe](#9-triển-khai-llm-với-kserve)
10. [Ví dụ thực tế: Triển khai Gemma-3-27B-IT](#10-ví-dụ-thực-tế-triển-khai-gemma-3-27b-it)
11. [Best Practices và Troubleshooting](#11-best-practices-và-troubleshooting)

---

## 1. Giới thiệu về KServe

### 1.1 Tổng quan

**KServe** là một nền tảng inference model native Kubernetes, cung cấp giải pháp chuẩn hóa, không phụ thuộc cloud cho việc serving cả predictive và generative machine learning models.

### 1.2 Các khả năng chính

| Tính năng | Mô tả |
|-----------|-------|
| Multi-protocol Support | REST v1/v2, gRPC, OpenAI-compatible APIs |
| Framework Agnostic | TensorFlow, PyTorch, scikit-learn, XGBoost, Hugging Face |
| Advanced Deployments | Canary rollouts, A/B testing, traffic splitting |
| Production-grade | Observability, health checking, distributed tracing |
| Serverless | Scale-to-zero on CPU và GPU |

### 1.3 So sánh với các giải pháp khác

| Tiêu chí | KServe | TFServing | TorchServe | Triton |
|----------|--------|-----------|------------|--------|
| Multi-framework | ✅ | ❌ | ❌ | ✅ |
| Kubernetes-native | ✅ | ❌ | ❌ | ❌ |
| Scale-to-zero | ✅ | ❌ | ❌ | ❌ |
| InferenceGraph | ✅ | ❌ | ❌ | ❌ |
| OpenAI API | ✅ | ❌ | ❌ | ✅ |

---

## 2. Kiến trúc KServe

### 2.1 Tổng quan kiến trúc

KServe triển khai kiến trúc phân tán với sự tách biệt rõ ràng giữa **Control Plane** và **Data Plane**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE                                │
│  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────────┐   │
│  │ InferenceService │ │ ServingRuntime   │ │ InferenceGraph      │   │
│  │ Reconciler       │ │ Reconciler       │ │ Reconciler          │   │
│  └────────┬─────────┘ └────────┬─────────┘ └──────────┬──────────┘   │
│           │                    │                      │              │
│  ┌────────▼────────────────────▼──────────────────────▼──────────┐   │
│  │                    KServe Controller Manager                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA PLANE                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │  Predictor   │   │ Transformer │   │  Explainer  │               │
│  │   (vLLM)     │   │  (Optional) │   │  (Optional) │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Protocols: REST v1/v2 | gRPC | OpenAI-compatible           │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Control Plane Components

| Component | Chức năng |
|-----------|-----------|
| **InferenceServiceReconciler** | Orchestrate lifecycle của inference workloads |
| **PredictorReconciler** | Quản lý lifecycle của predictor components |
| **TransformerReconciler** | Quản lý preprocessing pipeline |
| **ExplainerReconciler** | Điều khiển model explanation services |
| **Admission Webhooks** | Validate và mutate incoming resources |
| **Storage Initializer** | Chuẩn bị model artifacts từ các nguồn storage |

### 2.3 Data Plane Framework

```python
# Cấu trúc Data Plane
DataPlane
├── FastAPI/gRPC Servers    # Standardized inference protocols
├── Model Repository        # Framework-agnostic serving
├── Protocol Handlers       # KServe V1/V2, gRPC, OpenAI APIs
└── Base Model Classes      # Custom model implementations
```

### 2.4 Resource Model Hierarchy

```yaml
# Hierarchy của KServe Resources
InferenceService (primary serving resource)
├── Predictor (required - thực hiện inference)
├── Transformer (optional - preprocessing)
└── Explainer (optional - model explanations)

ServingRuntime/ClusterServingRuntime (runtime definitions)
├── Supported model formats
├── Protocol versions
└── Pod specifications

InferenceGraph (workflow orchestration)
├── Sequence routing
├── Traffic splitting
├── Ensemble aggregation
└── Conditional logic (Switch)
```

---

## 3. Các tính năng cốt lõi

### 3.1 InferenceService Component

**InferenceService** là resource chính để deploy ML models với 3 components:

#### Predictor (Required)
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: example-predictor
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
        version: "2.0"
      storageUri: "s3://bucket/model"
      protocolVersion: v2
      runtime: kserve-huggingfaceserver
```

| Field | Mô tả |
|-------|-------|
| `modelFormat.name` | Framework identifier (sklearn, pytorch, huggingface) |
| `modelFormat.version` | Framework version cho compatibility |
| `runtime` | ServingRuntime/ClusterServingRuntime name |
| `storageUri` | Vị trí model artifacts |
| `protocolVersion` | Inference protocol (v1 hoặc v2) |

#### Transformer (Optional)
```yaml
spec:
  transformer:
    containers:
    - name: preprocessing
      image: custom-transformer:latest
      resources:
        limits:
          cpu: "1"
          memory: "2Gi"
```

#### Explainer (Optional)
```yaml
spec:
  explainer:
    alibi:
      type: AnchorTabular
      storageUri: "s3://bucket/explainer"
```

### 3.2 Serving Runtimes

KServe sử dụng hệ thống runtime có thể mở rộng:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterServingRuntime
metadata:
  name: kserve-huggingfaceserver
spec:
  annotations:
    prometheus.kserve.io/scrape: "true"
  supportedModelFormats:
  - name: huggingface
    version: "1"
    autoSelect: true
  containers:
  - name: kserve-container
    image: kserve/huggingfaceserver:latest
    args:
    - --model_name={{.Name}}
    - --model_dir=/mnt/models
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
        nvidia.com/gpu: "1"
```

#### Runtime Selection Algorithm

1. **Namespace-scoped** `ServingRuntime` ưu tiên trước
2. **Format và version** compatibility checking
3. **Protocol support** validation
4. **Auto-selection** filtering
5. **Priority ordering** cho các trường hợp tie

### 3.3 Deployment Modes

#### Serverless Mode (Default)

```yaml
# Knative-based deployment
spec:
  predictor:
    minReplicas: 0  # Enable scale-to-zero
    maxReplicas: 10
    scaleTarget: 5  # Concurrent requests per pod
    scaleMetric: concurrency
```

**Đặc điểm:**
- Automatic scaling via Knative Pod Autoscaler (KPA)
- Scale-to-zero capability
- Traffic splitting cho canary deployments
- Istio service mesh integration

#### Raw Deployment Mode

```yaml
# Standard Kubernetes deployment
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5
```

**Đặc điểm:**
- Standard Kubernetes Deployment + Service
- HorizontalPodAutoscaler hoặc KEDA
- Ingress/Gateway API cho external access
- Không cần Knative dependencies

---

## 4. Scale-to-Zero Mechanism

### 4.1 Nguyên lý hoạt động

KServe sử dụng **Knative Pod Autoscaler (KPA)** để thực hiện scale-to-zero:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Scale-to-Zero Flow                           │
│                                                                 │
│  Request → Istio Gateway → Activator → Pod (scale from 0)      │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │   KPA Controller   │                       │
│                    │                    │                       │
│                    │ - Monitor metrics  │                       │
│                    │ - Scale decisions  │                       │
│                    │ - Queue requests   │                       │
│                    └───────────────────┘                       │
│                                                                 │
│  Idle → Grace Period → Scale to 0 → Request → Cold Start       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Cấu hình Scale-to-Zero

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llm-service
  annotations:
    # Knative annotations cho scale-to-zero
    autoscaling.knative.dev/class: kpa.autoscaling.knative.dev
    autoscaling.knative.dev/metric: concurrency
    autoscaling.knative.dev/target: "10"
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "5"
    # Scale-down delay
    autoscaling.knative.dev/scale-down-delay: "5m"
    # Window for averaging metrics
    autoscaling.knative.dev/window: "60s"
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 5
    containerConcurrency: 10
```

### 4.3 GPU Scale-to-Zero

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gpu-llm
  annotations:
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/target-utilization-percentage: "70"
spec:
  predictor:
    minReplicas: 0
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      resources:
        limits:
          nvidia.com/gpu: "1"
        requests:
          nvidia.com/gpu: "1"
```

### 4.4 Cold Start Optimization

| Chiến lược | Mô tả | Trade-off |
|------------|-------|-----------|
| **Model Caching** | Cache model trên node | Faster startup, storage cost |
| **Warm Pods** | Giữ minimum replicas | No cold start, resource cost |
| **Predictive Scaling** | Scale trước traffic spike | Complex setup |
| **Optimized Images** | Smaller container images | Development effort |

```yaml
# Minimum replicas để tránh cold start hoàn toàn
spec:
  predictor:
    minReplicas: 1  # Always keep at least 1 pod warm
```

---

## 5. Caching và Storage

### 5.1 Storage Initializer

Storage Initializer là webhook component xử lý việc tải model artifacts:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Storage Initializer Flow                       │
│                                                                  │
│  InferenceService → Webhook → Init Container → Model Download   │
│                                      │                           │
│                    ┌─────────────────▼──────────────────┐       │
│                    │   Supported Storage Backends        │       │
│                    │                                     │       │
│                    │  • S3 (AWS)                        │       │
│                    │  • GCS (Google Cloud)              │       │
│                    │  • Azure Blob Storage              │       │
│                    │  • Hugging Face Hub (hf://)        │       │
│                    │  • MLflow Registry                 │       │
│                    │  • PVC (Persistent Volume)         │       │
│                    │  • HTTP/HTTPS URLs                 │       │
│                    └────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 LocalModelCache

**LocalModelCache** cho phép cache model trên local node để giảm cold start:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LocalModelCache
metadata:
  name: gemma-cache
  namespace: default
spec:
  sourceModelUri: "hf://google/gemma-3-27b-it"
  modelSize: "54Gi"
  nodeGroups:
  - name: gpu-nodes
    nodeSelector:
      node.kubernetes.io/gpu: "true"
    persistentVolumeSpec:
      storageClassName: local-ssd
      resources:
        requests:
          storage: 60Gi
```

### 5.3 ClusterStorageContainer

Định nghĩa storage configuration ở cluster level:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterStorageContainer
metadata:
  name: default
spec:
  container:
    name: storage-initializer
    image: kserve/storage-initializer:latest
    resources:
      requests:
        memory: 100Mi
        cpu: 100m
      limits:
        memory: 1Gi
        cpu: "1"
  supportedUriFormats:
  - prefix: s3://
  - prefix: gs://
  - prefix: hf://
  - prefix: az://
```

### 5.4 Model Caching Strategies

```yaml
# PVC-based caching
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: cached-llm
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "pvc://model-cache-pvc/gemma-3-27b-it"
```

### 5.5 KV Cache Offloading (cho LLM)

```yaml
spec:
  predictor:
    model:
      args:
      # Enable KV cache offloading to CPU
      - --cpu-offload-gb=16
      # Swap space for KV cache
      - --swap-space=8
```

---

## 6. Routing với InferenceGraph

### 6.1 Tổng quan InferenceGraph

**InferenceGraph** cho phép orchestrate nhiều models với routing logic phức tạp:

```
┌─────────────────────────────────────────────────────────────────┐
│                    InferenceGraph Patterns                       │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ SEQUENCE │    │ SPLITTER │    │ ENSEMBLE │    ┌──────────┐   │
│  │  A → B   │    │  A ─┬─ B │    │  ┌─ A ─┐ │    │  SWITCH  │   │
│  │  → C     │    │     └─ C │    │  M     R │    │  if X: A │   │
│  └──────────┘    └──────────┘    │  └─ B ─┘ │    │  else: B │   │
│                                   └──────────┘    └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Sequence Router

Xử lý models tuần tự, output của model trước là input của model sau:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: llm-pipeline
spec:
  nodes:
    root:
      routerType: Sequence
      steps:
      - serviceName: tokenizer
        name: tokenizer-step
      - serviceName: embedding
        name: embedding-step
        data: $response
      - serviceName: llm-model
        name: inference-step
        data: $response
```

### 6.3 Splitter Router (A/B Testing)

Phân phối traffic giữa các model versions:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: canary-deployment
spec:
  nodes:
    root:
      routerType: Splitter
      steps:
      - serviceName: gemma-v1
        name: stable
        weight: 80
      - serviceName: gemma-v2
        name: canary
        weight: 20
```

### 6.4 Ensemble Router

Thực thi parallel và aggregate results:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: ensemble-models
spec:
  nodes:
    root:
      routerType: Ensemble
      steps:
      - serviceName: model-a
        name: model-a
      - serviceName: model-b
        name: model-b
      - serviceName: model-c
        name: model-c
      - serviceName: aggregator
        name: vote
        data: $response
```

### 6.5 Switch Router (Conditional)

Routing dựa trên CEL (Common Expression Language):

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: smart-router
spec:
  nodes:
    root:
      routerType: Switch
      steps:
      - serviceName: small-model
        name: small
        condition: "request.prompt.size() < 100"
      - serviceName: medium-model
        name: medium
        condition: "request.prompt.size() >= 100 && request.prompt.size() < 1000"
      - serviceName: large-model
        name: large
        condition: "request.prompt.size() >= 1000"
```

### 6.6 Complex Multi-Model Pipeline

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: rag-pipeline
spec:
  nodes:
    root:
      routerType: Sequence
      steps:
      - serviceName: query-processor
        name: process-query
      - nodeName: retriever-ensemble
        name: retrieve-docs
        data: $response
      - serviceName: reranker
        name: rerank
        data: $response
      - serviceName: llm-generator
        name: generate
        data: $response

    retriever-ensemble:
      routerType: Ensemble
      steps:
      - serviceName: dense-retriever
        name: dense
      - serviceName: sparse-retriever
        name: sparse
```

---

## 7. Autoscaling

### 7.1 Autoscaling Modes

```
┌─────────────────────────────────────────────────────────────────┐
│                    Autoscaling Architecture                      │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   Serverless    │    │ Raw Deployment  │    │    KEDA     │  │
│  │    (Knative)    │    │     (HPA)       │    │  (Optional) │  │
│  │                 │    │                 │    │             │  │
│  │ • Request-based │    │ • CPU/Memory    │    │ • Custom    │  │
│  │ • Scale-to-zero │    │ • Min replica 1 │    │   metrics   │  │
│  │ • Concurrency   │    │ • Custom metrics│    │ • Event-    │  │
│  │                 │    │                 │    │   driven    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Knative Pod Autoscaler (KPA)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: kpa-autoscale
  annotations:
    # Autoscaler class
    autoscaling.knative.dev/class: kpa.autoscaling.knative.dev
    # Metric type: concurrency, rps, cpu, memory
    autoscaling.knative.dev/metric: concurrency
    # Target value
    autoscaling.knative.dev/target: "10"
    # Scale bounds
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "20"
    # Panic mode settings
    autoscaling.knative.dev/panic-threshold-percentage: "200"
    autoscaling.knative.dev/panic-window-percentage: "10"
spec:
  predictor:
    containerConcurrency: 10
```

#### Metric Types

| Metric | Mô tả | Use Case |
|--------|-------|----------|
| `concurrency` | Số concurrent requests | Default, tốt cho LLM |
| `rps` | Requests per second | High-throughput services |
| `cpu` | CPU utilization | CPU-bound workloads |
| `memory` | Memory utilization | Memory-intensive models |

### 7.3 Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: hpa-autoscale
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/autoscalerClass: hpa
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 70  # CPU target percentage
    scaleMetric: cpu
---
# Custom HPA for advanced metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service-predictor
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"
```

### 7.4 KEDA Integration

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-keda-scaler
spec:
  scaleTargetRef:
    name: llm-service-predictor
  pollingInterval: 15
  cooldownPeriod: 300
  minReplicaCount: 0
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: kserve_request_queue_length
      query: sum(kserve_inference_request_queue_length{service="llm-service"})
      threshold: "10"
  - type: kafka
    metadata:
      bootstrapServers: kafka:9092
      consumerGroup: llm-requests
      topic: inference-requests
      lagThreshold: "100"
```

### 7.5 GPU-Aware Autoscaling

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gpu-autoscale
  annotations:
    autoscaling.knative.dev/class: kpa.autoscaling.knative.dev
    autoscaling.knative.dev/metric: concurrency
    autoscaling.knative.dev/target: "1"  # 1 request per GPU
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 4
    containerConcurrency: 1  # Single request per container
    model:
      resources:
        limits:
          nvidia.com/gpu: "1"
```

---

## 8. Fallback và Error Handling

Khi triển khai LLM trong production, việc xử lý lỗi và fallback là critical để đảm bảo high availability. KServe cung cấp nhiều cơ chế để xử lý các tình huống khi model gặp sự cố (ví dụ: lỗi 500).

### 8.1 Tổng quan các chiến lược Fallback

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fallback Strategies trong KServe                      │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  InferenceGraph  │  │  Istio Traffic   │  │  Envoy AI Gateway    │   │
│  │  Switch/Splitter │  │  Management      │  │  (KServe v0.15+)     │   │
│  │                  │  │                  │  │                      │   │
│  │ • Condition-based│  │ • Retry policy   │  │ • Provider fallback  │   │
│  │ • Soft/Hard deps │  │ • Circuit breaker│  │ • Auto failover      │   │
│  │ • Multi-model    │  │ • Timeout config │  │ • Health checks      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Fallback với InferenceGraph

#### 8.2.1 Soft vs Hard Dependencies

InferenceGraph hỗ trợ hai loại dependency cho các steps:

| Dependency Type | Behavior | Use Case |
|-----------------|----------|----------|
| **Hard** | Step failure halts execution | Critical processing steps |
| **Soft** | Step failure không block execution | Optional enhancement steps |

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: llm-with-fallback
spec:
  nodes:
    root:
      routerType: Sequence
      steps:
      - serviceName: primary-llm
        name: primary
        dependency: Hard  # Fail nếu primary fail
      - serviceName: post-processor
        name: enhance
        dependency: Soft  # Continue nếu post-processor fail
```

#### 8.2.2 Switch-based Fallback Pattern

Sử dụng Switch router với GJSON conditions để route đến fallback model:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: model-with-fallback
spec:
  nodes:
    root:
      routerType: Sequence
      steps:
      - serviceName: primary-gemma-27b
        name: try-primary
      - nodeName: fallback-checker
        name: check-response
        data: $response

    fallback-checker:
      routerType: Switch
      steps:
      # Nếu response có error field, route đến fallback
      - serviceName: fallback-gemma-9b
        name: fallback
        condition: "error"  # GJSON: check if error field exists
      # Nếu response OK, pass through
      - serviceName: response-passthrough
        name: success
        condition: "result"
```

#### 8.2.3 Splitter-based Load Balancing với Fallback

Phân tán traffic và tự động failover:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: ha-llm-service
spec:
  nodes:
    root:
      routerType: Splitter
      steps:
      - serviceName: gemma-primary-az1
        name: primary-az1
        weight: 50
      - serviceName: gemma-primary-az2
        name: primary-az2
        weight: 50
      # Fallback với weight thấp hơn
      # Traffic sẽ tự động route đến available services
```

### 8.3 Fallback với Istio Traffic Management

KServe tích hợp với Istio, cho phép sử dụng VirtualService và DestinationRule để cấu hình retry, timeout, và circuit breaker.

#### 8.3.1 VirtualService với Retry Policy

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-retry-policy
  namespace: llm-serving
spec:
  hosts:
  - gemma-3-27b-it.llm-serving.svc.cluster.local
  http:
  - route:
    - destination:
        host: gemma-3-27b-it.llm-serving.svc.cluster.local
    # Retry configuration
    retries:
      attempts: 3
      perTryTimeout: 30s
      retryOn: 5xx,reset,connect-failure,retriable-4xx
    # Overall timeout
    timeout: 120s
```

#### 8.3.2 DestinationRule với Circuit Breaker

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: llm-circuit-breaker
  namespace: llm-serving
spec:
  host: gemma-3-27b-it.llm-serving.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
        maxRetries: 3
    # Outlier detection (Circuit Breaker)
    outlierDetection:
      consecutive5xxErrors: 3        # 3 lỗi 5xx liên tiếp
      interval: 10s                  # Check mỗi 10s
      baseEjectionTime: 30s          # Thời gian eject minimum
      maxEjectionPercent: 50         # Max 50% endpoints bị eject
      minHealthPercent: 30           # Giữ ít nhất 30% healthy
```

#### 8.3.3 Kết hợp Multi-Version Fallback

```yaml
# Primary model
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-primary
  namespace: llm-serving
  labels:
    version: primary
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
---
# Fallback model (smaller, more reliable)
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-fallback
  namespace: llm-serving
  labels:
    version: fallback
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-2-9b-it"
---
# VirtualService với fallback routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: gemma-with-fallback
  namespace: llm-serving
spec:
  hosts:
  - gemma-service.llm-serving.svc.cluster.local
  http:
  - route:
    - destination:
        host: gemma-primary.llm-serving.svc.cluster.local
      weight: 100
    retries:
      attempts: 2
      perTryTimeout: 60s
      retryOn: 5xx,reset,connect-failure
    timeout: 180s
  # Fallback route khi primary không available
  - match:
    - headers:
        x-fallback:
          exact: "true"
    route:
    - destination:
        host: gemma-fallback.llm-serving.svc.cluster.local
---
# DestinationRule với circuit breaker để trigger fallback
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: gemma-primary-circuit
  namespace: llm-serving
spec:
  host: gemma-primary.llm-serving.svc.cluster.local
  trafficPolicy:
    outlierDetection:
      consecutive5xxErrors: 2
      interval: 5s
      baseEjectionTime: 60s
      maxEjectionPercent: 100
```

### 8.4 Envoy AI Gateway Fallback (KServe v0.15+)

KServe v0.15 giới thiệu tích hợp với **Envoy AI Gateway** cung cấp automatic failover cho LLM workloads.

#### 8.4.1 Provider Fallback Configuration

```yaml
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: AIGatewayRoute
metadata:
  name: llm-failover-route
spec:
  parentRefs:
  - name: ai-gateway
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /v1/chat/completions
    backendRefs:
    # Primary backend (higher priority)
    - name: gemma-primary
      namespace: llm-serving
      port: 80
      weight: 100
    # Fallback backend (lower priority, activated on primary failure)
    - name: gemma-fallback
      namespace: llm-serving
      port: 80
      weight: 0
---
# BackendTrafficPolicy với retry và health check
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: BackendTrafficPolicy
metadata:
  name: llm-failover-policy
spec:
  targetRefs:
  - group: gateway.networking.k8s.io
    kind: HTTPRoute
    name: llm-failover-route
  retry:
    numRetries: 3
    perRetry:
      backOff:
        baseInterval: 500ms
        maxInterval: 10s
      timeout: 60s
    retryOn:
      httpStatusCodes:
        - 500
        - 502
        - 503
        - 504
      triggers:
        - connect-failure
        - retriable-status-codes
        - reset
  healthCheck:
    passive:
      baseEjectionTime: 30s
      interval: 5s
      maxEjectionPercent: 100
      consecutive5XxErrors: 2
```

#### 8.4.2 Multi-Provider Failover

```yaml
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: AIGatewayRoute
metadata:
  name: multi-provider-fallback
spec:
  rules:
  - backendRefs:
    # Priority 1: Primary on-prem model
    - name: gemma-onprem
      weight: 100
    # Priority 2: Cloud backup
    - name: gemma-cloud-backup
      weight: 0
    # Priority 3: External API fallback
    - name: external-llm-api
      weight: 0
```

### 8.5 Application-Level Fallback Pattern

Ngoài infrastructure-level fallback, bạn cũng có thể implement fallback ở application level:

#### 8.5.1 Client-Side Retry với Fallback

```python
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMClient:
    def __init__(self):
        self.primary_url = "http://gemma-primary.llm-serving/v1/chat/completions"
        self.fallback_url = "http://gemma-fallback.llm-serving/v1/chat/completions"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_model(self, url: str, payload: dict) -> dict:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def generate(self, messages: list, **kwargs) -> dict:
        payload = {
            "model": "gemma-3-27b-it",
            "messages": messages,
            **kwargs
        }

        # Try primary first
        try:
            return self._call_model(self.primary_url, payload)
        except Exception as e:
            print(f"Primary failed: {e}, trying fallback...")

        # Fallback to smaller model
        payload["model"] = "gemma-2-9b-it"
        try:
            return self._call_model(self.fallback_url, payload)
        except Exception as e:
            print(f"Fallback also failed: {e}")
            raise

# Usage
client = LLMClient()
response = client.generate([
    {"role": "user", "content": "Hello, how are you?"}
])
```

#### 8.5.2 Kubernetes Service với Multiple Endpoints

```yaml
# Headless service để client tự manage endpoints
apiVersion: v1
kind: Service
metadata:
  name: llm-endpoints
  namespace: llm-serving
spec:
  clusterIP: None  # Headless
  selector:
    app: llm-service
  ports:
  - port: 80
    targetPort: 8080
---
# Endpoints pointing to multiple InferenceServices
apiVersion: v1
kind: Endpoints
metadata:
  name: llm-endpoints
  namespace: llm-serving
subsets:
- addresses:
  - ip: <gemma-primary-cluster-ip>
  - ip: <gemma-fallback-cluster-ip>
  ports:
  - port: 8080
```

### 8.6 Health Check Configuration

Cấu hình health check đúng cách là quan trọng để fallback hoạt động hiệu quả:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-with-health-checks
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      # Readiness probe - check if model ready to serve
      readinessProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 120
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 3
        successThreshold: 1
      # Liveness probe - check if container alive
      livenessProbe:
        httpGet:
          path: /v2/health/live
          port: 8080
        initialDelaySeconds: 300
        periodSeconds: 30
        timeoutSeconds: 10
        failureThreshold: 3
      # Startup probe - longer timeout for initial model loading
      startupProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 60
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 30  # 5 minutes total startup time
```

### 8.7 Monitoring và Alerting cho Fallback

```yaml
# PrometheusRule để alert khi fallback được triggered
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: llm-fallback-alerts
spec:
  groups:
  - name: llm-fallback
    rules:
    # Alert khi error rate cao
    - alert: LLMHighErrorRate
      expr: |
        sum(rate(kserve_inference_request_count{status="500"}[5m]))
        / sum(rate(kserve_inference_request_count[5m])) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High error rate on LLM service"
        description: "Error rate is above 10% for 2 minutes"

    # Alert khi primary bị ejected (circuit open)
    - alert: LLMPrimaryCircuitOpen
      expr: |
        envoy_cluster_outlier_detection_ejections_active{cluster="gemma-primary"} > 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Primary LLM model circuit breaker open"
        description: "Traffic is being routed to fallback model"

    # Alert khi tất cả models unavailable
    - alert: LLMAllModelsUnavailable
      expr: |
        sum(kube_deployment_status_replicas_available{
          deployment=~"gemma-.*"
        }) == 0
      for: 30s
      labels:
        severity: critical
      annotations:
        summary: "All LLM models unavailable"
```

### 8.8 Complete Fallback Architecture Example

```yaml
# Complete example: Gemma-3-27B với multi-layer fallback

# 1. Primary InferenceService
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-primary
  namespace: llm-serving
  labels:
    tier: primary
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 5
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      args:
      - --backend=vllm
      - --tensor-parallel-size=2
      resources:
        limits:
          nvidia.com/gpu: "2"
          memory: "80Gi"
---
# 2. Fallback InferenceService (smaller, faster)
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-fallback
  namespace: llm-serving
  labels:
    tier: fallback
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 3
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-2-9b-it"
      args:
      - --backend=vllm
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: "24Gi"
---
# 3. InferenceGraph orchestrating fallback
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: gemma-ha-service
  namespace: llm-serving
spec:
  nodes:
    root:
      routerType: Sequence
      steps:
      - serviceName: gemma-primary
        name: try-primary
        dependency: Soft  # Don't fail completely if primary fails
      - nodeName: response-handler
        name: handle-response
        data: $response

    response-handler:
      routerType: Switch
      steps:
      # If error in response, use fallback
      - serviceName: gemma-fallback
        name: use-fallback
        condition: "error"
      # If successful, return as-is
      - serviceName: passthrough
        name: return-success
        condition: "choices"
---
# 4. Istio DestinationRule for circuit breaking
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: gemma-primary-circuit
  namespace: llm-serving
spec:
  host: gemma-primary.llm-serving.svc.cluster.local
  trafficPolicy:
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
      baseEjectionTime: 60s
      maxEjectionPercent: 100
---
# 5. VirtualService with retry
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: gemma-retry-policy
  namespace: llm-serving
spec:
  hosts:
  - gemma-ha-service.llm-serving.svc.cluster.local
  http:
  - route:
    - destination:
        host: gemma-ha-service.llm-serving.svc.cluster.local
    retries:
      attempts: 2
      perTryTimeout: 60s
      retryOn: 5xx,reset,connect-failure
    timeout: 180s
```

### 8.9 Fallback Best Practices

| Practice | Mô tả |
|----------|-------|
| **Graceful Degradation** | Fallback model nên provide similar API, có thể với lower quality |
| **Fast Failure Detection** | Configure aggressive health checks để detect failures sớm |
| **Capacity Planning** | Fallback phải có đủ capacity khi primary unavailable |
| **Testing** | Regularly test fallback scenarios (Chaos Engineering) |
| **Monitoring** | Alert khi fallback triggered, track fallback usage |
| **Timeout Tuning** | Set reasonable timeouts, account for LLM latency |
| **State Management** | Ensure stateless design cho seamless failover |

---

## 9. Triển khai LLM với KServe

### 9.1 LLMInferenceService

KServe cung cấp **LLMInferenceService** cho workloads LLM chuyên biệt:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: llama-service
spec:
  model:
    uri: "hf://meta-llama/Llama-2-70b"
    modelName: llama-2-70b
  adapters:
  - name: sql-lora
    uri: "hf://sql-lora-adapter"
  parallelism:
    tensor: 4
    pipeline: 2
    data: 1
```

### 8.2 Deployment Patterns cho LLM

#### Single-Node Deployment

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: single-node-llm
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-2-9b-it"
      args:
      - --backend=vllm
      - --dtype=float16
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: "24Gi"
        requests:
          nvidia.com/gpu: "1"
          memory: "20Gi"
```

#### Multi-Node với Tensor Parallelism

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: multi-gpu-llm
spec:
  model:
    uri: "hf://meta-llama/Llama-2-70b"
  parallelism:
    tensor: 8  # Split across 8 GPUs
  workerSpec:
    replicas: 1
    template:
      spec:
        containers:
        - name: worker
          resources:
            limits:
              nvidia.com/gpu: "8"
```

#### Disaggregated Prefill/Decode

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: disaggregated-llm
spec:
  model:
    uri: "hf://meta-llama/Llama-2-70b"
  prefill:
    replicas: 2
    resources:
      limits:
        nvidia.com/gpu: "4"
  decode:
    replicas: 4
    resources:
      limits:
        nvidia.com/gpu: "2"
```

### 8.3 vLLM Backend Configuration

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: vllm-optimized
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      runtime: kserve-huggingfaceserver
      args:
      # vLLM backend
      - --backend=vllm
      # Tensor parallelism
      - --tensor-parallel-size=2
      # Memory optimization
      - --gpu-memory-utilization=0.9
      - --max-model-len=8192
      # Batching
      - --max-num-batched-tokens=32768
      - --max-num-seqs=256
      # Quantization
      - --quantization=awq
      # Speculative decoding
      - --speculative-model=google/gemma-2-2b-it
      - --num-speculative-tokens=5
```

### 8.4 OpenAI-Compatible API

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: openai-compatible-llm
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      args:
      - --backend=vllm
      env:
      - name: KSERVE_OPENAI_ROUTE_PREFIX
        value: "/v1"
```

**API Endpoints:**
- Chat Completions: `POST /v1/chat/completions`
- Completions: `POST /v1/completions`
- Embeddings: `POST /v1/embeddings`
- Models: `GET /v1/models`

---

## 10. Ví dụ thực tế: Triển khai Gemma-3-27B-IT

### 10.1 Yêu cầu hệ thống

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Memory | 48GB (FP16) | 80GB (A100) |
| System RAM | 64GB | 128GB |
| Storage | 60GB SSD | 100GB NVMe |
| GPU | 2x RTX 4090 / 1x A100 | 2x A100 80GB |
| Kubernetes | 1.24+ | 1.28+ |

### 10.2 Prerequisites Setup

```bash
# 1. Install KServe with serverless mode
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.15/hack/quick_install.sh" | bash -s -- -s

# 2. Verify installation
kubectl get pods -n kserve
kubectl get crd | grep serving.kserve.io

# 3. Create namespace for LLM workloads
kubectl create namespace llm-serving

# 4. Create Hugging Face token secret
kubectl create secret generic hf-token \
  --from-literal=HF_TOKEN=<your-huggingface-token> \
  -n llm-serving
```

### 10.3 Basic Deployment

```yaml
# gemma-3-27b-basic.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-3-27b-it
  namespace: llm-serving
  annotations:
    autoscaling.knative.dev/minScale: "1"
    autoscaling.knative.dev/maxScale: "3"
    autoscaling.knative.dev/target: "2"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 3
    containerConcurrency: 2
    timeout: 300
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      runtime: kserve-huggingfaceserver
      args:
      - --backend=vllm
      - --dtype=float16
      - --max-model-len=8192
      - --gpu-memory-utilization=0.9
      resources:
        limits:
          nvidia.com/gpu: "2"
          memory: "80Gi"
          cpu: "8"
        requests:
          nvidia.com/gpu: "2"
          memory: "64Gi"
          cpu: "4"
      env:
      - name: HF_TOKEN
        valueFrom:
          secretKeyRef:
            name: hf-token
            key: HF_TOKEN
```

### 10.4 Production-Ready Deployment

```yaml
# gemma-3-27b-production.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-3-27b-production
  namespace: llm-serving
  annotations:
    # Autoscaling
    autoscaling.knative.dev/class: kpa.autoscaling.knative.dev
    autoscaling.knative.dev/metric: concurrency
    autoscaling.knative.dev/target: "2"
    autoscaling.knative.dev/minScale: "1"
    autoscaling.knative.dev/maxScale: "5"
    autoscaling.knative.dev/scale-down-delay: "10m"
    # Prometheus metrics
    prometheus.kserve.io/scrape: "true"
    prometheus.kserve.io/port: "8080"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 5
    containerConcurrency: 2
    timeout: 600
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      runtime: kserve-huggingfaceserver
      args:
      # vLLM optimizations
      - --backend=vllm
      - --dtype=bfloat16
      - --tensor-parallel-size=2
      - --max-model-len=8192
      - --max-num-batched-tokens=32768
      - --max-num-seqs=128
      - --gpu-memory-utilization=0.92
      # Speculative decoding (optional)
      # - --speculative-model=google/gemma-2-2b-it
      # - --num-speculative-tokens=5
      # Prefix caching
      - --enable-prefix-caching
      # Chunked prefill
      - --enable-chunked-prefill
      resources:
        limits:
          nvidia.com/gpu: "2"
          memory: "96Gi"
          cpu: "16"
        requests:
          nvidia.com/gpu: "2"
          memory: "80Gi"
          cpu: "8"
      env:
      - name: HF_TOKEN
        valueFrom:
          secretKeyRef:
            name: hf-token
            key: HF_TOKEN
      - name: VLLM_LOGGING_LEVEL
        value: "INFO"
    # Node affinity for GPU nodes
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          nodeSelectorTerms:
          - matchExpressions:
            - key: nvidia.com/gpu.product
              operator: In
              values:
              - NVIDIA-A100-SXM4-80GB
              - NVIDIA-A100-PCIE-80GB
    # Tolerations for GPU taints
    tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

### 10.5 Scale-to-Zero Configuration

```yaml
# gemma-3-27b-serverless.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-3-27b-serverless
  namespace: llm-serving
  annotations:
    # Enable scale-to-zero
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "3"
    autoscaling.knative.dev/target: "1"
    # Longer scale-down delay for cold start mitigation
    autoscaling.knative.dev/scale-down-delay: "30m"
    # Initial scale
    autoscaling.knative.dev/initial-scale: "1"
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 3
    containerConcurrency: 1
    timeout: 600
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      args:
      - --backend=vllm
      - --dtype=float16
      - --tensor-parallel-size=2
      resources:
        limits:
          nvidia.com/gpu: "2"
          memory: "80Gi"
        requests:
          nvidia.com/gpu: "2"
          memory: "64Gi"
```

### 10.6 Model Caching Setup

```yaml
# Step 1: Create PVC for model caching
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gemma-model-cache
  namespace: llm-serving
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-nvme
---
# Step 2: Pre-download job
apiVersion: batch/v1
kind: Job
metadata:
  name: download-gemma
  namespace: llm-serving
spec:
  template:
    spec:
      containers:
      - name: downloader
        image: kserve/storage-initializer:latest
        command: ["/bin/sh", "-c"]
        args:
        - |
          python -c "
          from huggingface_hub import snapshot_download
          snapshot_download(
            repo_id='google/gemma-3-27b-it',
            local_dir='/mnt/models/gemma-3-27b-it',
            token='$HF_TOKEN'
          )"
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: HF_TOKEN
        volumeMounts:
        - name: model-cache
          mountPath: /mnt/models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: gemma-model-cache
      restartPolicy: Never
---
# Step 3: Deploy with cached model
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-cached
  namespace: llm-serving
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "pvc://gemma-model-cache/gemma-3-27b-it"
      args:
      - --backend=vllm
      resources:
        limits:
          nvidia.com/gpu: "2"
```

### 10.7 A/B Testing với InferenceGraph

```yaml
# gemma-ab-testing.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-v1
  namespace: llm-serving
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-2-27b-it"
      args:
      - --backend=vllm
---
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: gemma-v2
  namespace: llm-serving
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "hf://google/gemma-3-27b-it"
      args:
      - --backend=vllm
---
apiVersion: serving.kserve.io/v1alpha1
kind: InferenceGraph
metadata:
  name: gemma-canary
  namespace: llm-serving
spec:
  nodes:
    root:
      routerType: Splitter
      steps:
      - serviceName: gemma-v1
        name: stable
        weight: 90
      - serviceName: gemma-v2
        name: canary
        weight: 10
```

### 10.8 Testing và Verification

```bash
# Get service URL
export SERVICE_HOSTNAME=$(kubectl get inferenceservice gemma-3-27b-it \
  -n llm-serving -o jsonpath='{.status.url}')

# OpenAI-compatible chat request
curl -X POST "${SERVICE_HOSTNAME}/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'

# V2 inference protocol
curl -X POST "${SERVICE_HOSTNAME}/v2/models/gemma-3-27b-it/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "text_input",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["What is machine learning?"]
      }
    ]
  }'

# Check model status
curl "${SERVICE_HOSTNAME}/v2/models/gemma-3-27b-it/ready"

# Get metrics
curl "${SERVICE_HOSTNAME}/metrics"
```

### 10.9 Monitoring Dashboard

```yaml
# prometheus-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: gemma-monitor
  namespace: llm-serving
spec:
  selector:
    matchLabels:
      serving.kserve.io/inferenceservice: gemma-3-27b-it
  endpoints:
  - port: http
    interval: 15s
    path: /metrics
---
# Grafana Dashboard ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: kserve-llm-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  kserve-llm.json: |
    {
      "title": "KServe LLM Metrics",
      "panels": [
        {
          "title": "Request Latency",
          "targets": [
            {
              "expr": "histogram_quantile(0.95, sum(rate(kserve_inference_latency_bucket[5m])) by (le))"
            }
          ]
        },
        {
          "title": "Tokens per Second",
          "targets": [
            {
              "expr": "sum(rate(vllm_num_generation_tokens_total[5m]))"
            }
          ]
        },
        {
          "title": "GPU Memory Usage",
          "targets": [
            {
              "expr": "sum(vllm_gpu_cache_usage_perc) by (pod)"
            }
          ]
        }
      ]
    }
```

---

## 11. Best Practices và Troubleshooting

### 11.1 Best Practices

#### Resource Planning
```yaml
# Calculate resources based on model size
# Gemma-3-27B-IT: ~54GB in FP16, ~27GB in INT8

# For FP16 (full precision)
resources:
  limits:
    nvidia.com/gpu: "2"  # 2x 40GB GPUs or 1x 80GB GPU
    memory: "80Gi"

# For INT4 quantization (AWQ/GPTQ)
resources:
  limits:
    nvidia.com/gpu: "1"  # Single 24GB+ GPU
    memory: "32Gi"
```

#### Timeout Configuration
```yaml
spec:
  predictor:
    timeout: 600  # 10 minutes for long sequences
    containerConcurrency: 2  # Limit concurrent requests per pod
```

#### Health Checks
```yaml
spec:
  predictor:
    model:
      readinessProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 120
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 3
      livenessProbe:
        httpGet:
          path: /v2/health/live
          port: 8080
        initialDelaySeconds: 300
        periodSeconds: 30
```

### 11.2 Common Issues và Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM Killed | Model too large | Reduce `gpu-memory-utilization`, enable quantization |
| Cold Start Slow | Large model download | Use LocalModelCache, PVC caching |
| Timeout Errors | Long inference time | Increase timeout, reduce max_tokens |
| Scale Issues | Incorrect metrics | Check autoscaling config, metrics |
| Connection Refused | Pod not ready | Check readiness probe, startup time |

### 11.3 Debugging Commands

```bash
# Check pod status
kubectl get pods -n llm-serving -l serving.kserve.io/inferenceservice=gemma-3-27b-it

# View pod logs
kubectl logs -n llm-serving -l serving.kserve.io/inferenceservice=gemma-3-27b-it -c kserve-container --tail=100

# Check InferenceService status
kubectl describe inferenceservice gemma-3-27b-it -n llm-serving

# Check events
kubectl get events -n llm-serving --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n llm-serving

# Debug Knative revision
kubectl get revision -n llm-serving
kubectl describe revision <revision-name> -n llm-serving
```

### 11.4 Performance Tuning

```yaml
# Optimized configuration for maximum throughput
spec:
  predictor:
    model:
      args:
      # Batching optimization
      - --max-num-batched-tokens=65536
      - --max-num-seqs=256
      # Memory optimization
      - --gpu-memory-utilization=0.95
      - --swap-space=8
      # Prefix caching for similar prompts
      - --enable-prefix-caching
      # Chunked prefill for long contexts
      - --enable-chunked-prefill
      - --max-num-batched-tokens=32768
```

### 11.5 Security Considerations

```yaml
# Security-hardened deployment
spec:
  predictor:
    model:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        readOnlyRootFilesystem: true
        allowPrivilegeEscalation: false
        capabilities:
          drop:
          - ALL
    # Network policy
    podSpec:
      networkPolicy:
        ingress:
        - from:
          - namespaceSelector:
              matchLabels:
                name: api-gateway
```

---

## Tài liệu tham khảo

- [KServe Official Documentation](https://kserve.github.io/website/)
- [KServe GitHub Repository](https://github.com/kserve/kserve)
- [DeepWiki KServe Guide](https://deepwiki.com/kserve/kserve)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Knative Serving](https://knative.dev/docs/serving/)
- [Hugging Face Model Hub](https://huggingface.co/models)

---

*Tài liệu được tạo ngày: 2025-01-24*
*Version: 1.0*
