# DCGM (Data Center GPU Manager) - Hướng Dẫn Giám Sát GPU

## 1. Tổng Quan về DCGM

### 1.1 DCGM là gì?

**DCGM (Data Center GPU Manager)** là bộ công cụ của NVIDIA dùng để quản lý và giám sát GPU trong các môi trường cluster Linux quy mô lớn. DCGM cung cấp:

- **Active health monitoring**: Giám sát sức khỏe GPU theo thời gian thực
- **Diagnostics**: Chẩn đoán và phát hiện lỗi GPU
- **System validation**: Xác thực hệ thống GPU
- **Policy management**: Quản lý chính sách hoạt động
- **Power & clock management**: Quản lý nguồn và xung nhịp
- **Accounting**: Theo dõi và ghi nhận tài nguyên sử dụng

### 1.2 Mục Đích Chính

DCGM được thiết kế để:
- Giám sát GPU trong môi trường production
- Cung cấp metrics cho hệ thống monitoring (Prometheus, Grafana)
- Hỗ trợ troubleshooting và diagnostics
- Tích hợp với Kubernetes và container ecosystems
- Cho phép autoscaling dựa trên GPU utilization

### 1.3 Use Cases

| Use Case | Mô Tả |
|----------|-------|
| **Data Center Management** | Quản lý hàng ngàn GPU trong data center |
| **Kubernetes Monitoring** | Giám sát GPU workloads trong K8s cluster |
| **HPC Clusters** | Theo dõi GPU trong môi trường HPC |
| **Cloud GPU Instances** | Monitoring GPU trên cloud platforms |
| **ML/AI Training** | Giám sát training jobs và resource utilization |

---

## 2. Kiến Trúc DCGM

### 2.1 Tổng Quan Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Monitoring Stack                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Grafana    │◄───│  Prometheus  │◄───│dcgm-exporter │       │
│  │ (Dashboard)  │    │  (Storage)   │    │  (Metrics)   │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│                                           ┌──────────────┐       │
│                                           │     DCGM     │       │
│                                           │   (Agent)    │       │
│                                           └──────┬───────┘       │
│                                                  │               │
│                      ┌───────────────────────────┼───────────┐   │
│                      │                           │           │   │
│                      ▼                           ▼           ▼   │
│                 ┌─────────┐               ┌─────────┐   ┌─────────┐
│                 │  GPU 0  │               │  GPU 1  │   │  GPU N  │
│                 │ (A100)  │               │ (A100)  │   │ (A100)  │
│                 └─────────┘               └─────────┘   └─────────┘
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### 2.2.1 DCGM Agent/Daemon

DCGM daemon chạy như một service trên host, thu thập metrics trực tiếp từ GPU drivers:

```bash
# Start DCGM daemon
sudo systemctl start dcgm

# Check DCGM status
dcgmi discovery -l
```

#### 2.2.2 dcgm-exporter

**dcgm-exporter** là cầu nối giữa DCGM và Prometheus:
- Kết nối với DCGM để thu thập GPU telemetry
- Expose metrics qua HTTP endpoint (port 9400)
- Kết nối kubelet's pod-resources socket để mapping GPU với Pods

```
GPU Devices → DCGM → dcgm-exporter → Prometheus → Grafana
                          │
                          ▼
                 kubelet pod-resources API
                 (Pod-to-GPU correlation)
```

#### 2.2.3 DCGM APIs

DCGM cung cấp nhiều APIs:
- **C API**: Native API cho high-performance applications
- **Python bindings**: Cho scripting và automation
- **Go bindings**: Cho container ecosystem integration
- **REST API**: Cho remote monitoring

---

## 3. GPU Metrics

### 3.1 Các Metrics Chính

| Metric | Mô Tả | Đơn Vị |
|--------|-------|--------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU utilization | % |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory copy utilization | % |
| `DCGM_FI_DEV_ENC_UTIL` | Encoder utilization | % |
| `DCGM_FI_DEV_DEC_UTIL` | Decoder utilization | % |
| `DCGM_FI_DEV_FB_FREE` | Framebuffer memory free | MB |
| `DCGM_FI_DEV_FB_USED` | Framebuffer memory used | MB |
| `DCGM_FI_DEV_POWER_USAGE` | Power usage | W |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature | °C |
| `DCGM_FI_DEV_SM_CLOCK` | SM clock frequency | MHz |
| `DCGM_FI_DEV_MEM_CLOCK` | Memory clock frequency | MHz |

### 3.2 Profiling Metrics (Require elevated permissions)

```
┌─────────────────────────────────────────────────────────────┐
│                    Profiling Metrics                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GrActive (Graphics Active)                                  │
│  ├── Percentage of time GPU is actively processing          │
│  └── Key indicator for GPU workload                         │
│                                                              │
│  Tensor Core Utilization                                     │
│  ├── SM (Streaming Multiprocessor) occupancy                │
│  └── Tensor Core activity for AI/ML workloads               │
│                                                              │
│  Memory Bandwidth                                            │
│  ├── DRAM read/write bandwidth                              │
│  └── NVLink/PCIe bandwidth                                  │
│                                                              │
│  Interconnect Traffic                                        │
│  ├── NVLink traffic between GPUs                            │
│  └── PCIe traffic to host                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Prometheus Metrics Format

Khi dcgm-exporter expose metrics, format như sau:

```prometheus
# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %)
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-xxxxx",device="nvidia0",modelName="NVIDIA A100-SXM4-40GB",Hostname="node-01",container="training",namespace="ml",pod="training-job-xyz"} 95

# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MB)
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-xxxxx",...} 38000

# HELP DCGM_FI_DEV_POWER_USAGE Power draw (in W)
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-xxxxx",...} 350
```

---

## 4. Triển Khai trên Kubernetes

### 4.1 Prometheus + Grafana Stack

#### Bước 1: Cài đặt Prometheus Operator

```bash
# Add Helm repository
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts

# Update repositories
helm repo update

# Install kube-prometheus-stack
helm install prometheus-community/kube-prometheus-stack \
  --create-namespace --namespace prometheus \
  --generate-name \
  --set prometheus.service.type=NodePort \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

**Giải thích các flags:**
- `--create-namespace --namespace prometheus`: Tạo namespace riêng
- `--generate-name`: Tự động generate tên release
- `serviceMonitorSelectorNilUsesHelmValues=false`: Cho phép Prometheus tự động discover ServiceMonitors

#### Bước 2: Cài đặt dcgm-exporter

```bash
# Add NVIDIA GPU monitoring Helm repository
helm repo add gpu-helm-charts \
  https://nvidia.github.io/gpu-monitoring-tools/helm-charts

# Update repositories
helm repo update

# Install dcgm-exporter
helm install --generate-name gpu-helm-charts/dcgm-exporter
```

#### Bước 3: Verify Installation

```bash
# Check dcgm-exporter pods
kubectl get pods -l app.kubernetes.io/name=dcgm-exporter

# Check ServiceMonitor created
kubectl get servicemonitor

# Check Prometheus targets
kubectl port-forward -n prometheus svc/prometheus-operated 9090
# Access http://localhost:9090/targets
```

### 4.2 dcgm-exporter DaemonSet Configuration

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: gpu-monitoring
  labels:
    app.kubernetes.io/name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  template:
    metadata:
      labels:
        app.kubernetes.io/name: dcgm-exporter
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
        ports:
        - name: metrics
          containerPort: 9400
        env:
        - name: DCGM_EXPORTER_LISTEN
          value: ":9400"
        - name: DCGM_EXPORTER_KUBERNETES
          value: "true"
        securityContext:
          runAsNonRoot: false
          runAsUser: 0
          capabilities:
            add: ["SYS_ADMIN"]
        volumeMounts:
        - name: pod-gpu-resources
          mountPath: /var/lib/kubelet/pod-resources
          readOnly: true
      volumes:
      - name: pod-gpu-resources
        hostPath:
          path: /var/lib/kubelet/pod-resources
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### 4.3 ServiceMonitor cho Prometheus

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: gpu-monitoring
  labels:
    app.kubernetes.io/name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  namespaceSelector:
    matchNames:
    - gpu-monitoring
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

---

## 5. Thực Hành: Tạo GPU Workload và Monitoring

### 5.1 dcgmproftester - GPU Stress Test Tool

#### Dockerfile để build dcgmproftester

```dockerfile
ARG BASE_DIST
ARG CUDA_VER
FROM nvidia/cuda:${CUDA_VER}-base-${BASE_DIST}

ARG DCGM_VERSION
WORKDIR /dcgm

RUN apt-get update && apt-get install -y libgomp1 wget && \
    wget --no-check-certificate \
    https://developer.download.nvidia.com/compute/redist/dcgm/${DCGM_VERSION}/DEBS/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
    dpkg -i datacenter-gpu-manager_*.deb && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/bin/dcgmproftester11"]
```

**Build image:**

```bash
docker build \
  --build-arg BASE_DIST=ubuntu18.04 \
  --build-arg CUDA_VER=11.0 \
  --build-arg DCGM_VERSION=2.0.10 \
  -t dcgmproftester:latest .
```

### 5.2 Kubernetes Pod để chạy GPU workload

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dcgmproftester
  labels:
    app: gpu-test
spec:
  restartPolicy: OnFailure
  containers:
  - name: dcgmproftester11
    image: nvidia/samples:dcgmproftester-2.0.10-cuda11.0-ubuntu18.04
    args: ["--no-dcgm-validation", "-t 1004", "-d 120"]
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
    securityContext:
      capabilities:
        add: ["SYS_ADMIN"]
```

**Giải thích arguments:**
- `--no-dcgm-validation`: Bỏ qua DCGM validation (chạy standalone)
- `-t 1004`: Test ID 1004 (FP16 matrix multiply - Tensor Core test)
- `-d 120`: Duration 120 seconds

### 5.3 Các Test Types của dcgmproftester

| Test ID | Mô Tả | Target |
|---------|-------|--------|
| 1004 | FP16 Matrix Multiply | Tensor Cores |
| 1005 | FP32 Matrix Multiply | CUDA Cores |
| 1006 | FP64 Matrix Multiply | Double Precision |
| 1007 | INT8 Matrix Multiply | INT8 Tensor Cores |
| 1008 | Memory Bandwidth | VRAM Bandwidth |

### 5.4 Chạy Multiple GPU Workloads

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-stress-test
spec:
  parallelism: 4  # Chạy 4 pods song song
  completions: 4
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: stress-test
        image: nvidia/samples:dcgmproftester-2.0.10-cuda11.0-ubuntu18.04
        args: ["--no-dcgm-validation", "-t 1004", "-d 300"]
        resources:
          limits:
            nvidia.com/gpu: 1
        securityContext:
          capabilities:
            add: ["SYS_ADMIN"]
```

---

## 6. Grafana Dashboard

### 6.1 Import NVIDIA Dashboard

NVIDIA cung cấp official Grafana dashboard:
- **Dashboard ID**: 12239
- **Name**: NVIDIA DCGM Exporter Dashboard

**Import steps:**
1. Mở Grafana UI
2. Vào **Dashboards** → **Import**
3. Nhập ID: `12239`
4. Chọn Prometheus datasource
5. Click **Import**

### 6.2 Custom Dashboard Panels

#### Panel: GPU Utilization per Pod

```json
{
  "title": "GPU Utilization per Pod",
  "type": "timeseries",
  "targets": [
    {
      "expr": "DCGM_FI_DEV_GPU_UTIL{namespace=~\"$namespace\", pod=~\"$pod\"}",
      "legendFormat": "{{pod}} - GPU {{gpu}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "percent",
      "max": 100,
      "min": 0
    }
  }
}
```

#### Panel: GPU Memory Usage

```json
{
  "title": "GPU Memory Usage",
  "type": "gauge",
  "targets": [
    {
      "expr": "DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100",
      "legendFormat": "GPU {{gpu}}"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "percent",
      "thresholds": {
        "steps": [
          { "value": 0, "color": "green" },
          { "value": 70, "color": "yellow" },
          { "value": 90, "color": "red" }
        ]
      }
    }
  }
}
```

#### Panel: Power Consumption

```promql
# Total cluster power consumption
sum(DCGM_FI_DEV_POWER_USAGE)

# Power per node
sum by (Hostname) (DCGM_FI_DEV_POWER_USAGE)

# Power per namespace
sum by (namespace) (DCGM_FI_DEV_POWER_USAGE)
```

### 6.3 Access Grafana

```bash
# Port-forward Grafana service
kubectl port-forward -n prometheus svc/kube-prometheus-stack-grafana 3000:80

# Access via browser
# URL: http://localhost:3000
# Default credentials: admin/prom-operator
```

---

## 7. Alerting

### 7.1 PrometheusRule cho GPU Alerts

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-alerts
  namespace: prometheus
  labels:
    release: prometheus
spec:
  groups:
  - name: gpu-alerts
    rules:
    # High GPU Temperature
    - alert: GPUHighTemperature
      expr: DCGM_FI_DEV_GPU_TEMP > 80
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "GPU temperature high on {{ $labels.Hostname }}"
        description: "GPU {{ $labels.gpu }} temperature is {{ $value }}°C"

    # GPU Memory Almost Full
    - alert: GPUMemoryAlmostFull
      expr: |
        (DCGM_FI_DEV_FB_USED /
        (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)) * 100 > 90
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "GPU memory usage > 90% on {{ $labels.Hostname }}"
        description: "GPU {{ $labels.gpu }} memory usage is {{ $value }}%"

    # GPU Utilization Low (potential idle)
    - alert: GPUUnderutilized
      expr: DCGM_FI_DEV_GPU_UTIL < 10
      for: 30m
      labels:
        severity: info
      annotations:
        summary: "GPU underutilized on {{ $labels.Hostname }}"
        description: "GPU {{ $labels.gpu }} utilization is only {{ $value }}%"

    # High Power Draw
    - alert: GPUHighPowerDraw
      expr: DCGM_FI_DEV_POWER_USAGE > 400
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High power draw on GPU {{ $labels.gpu }}"
        description: "Power consumption is {{ $value }}W"
```

### 7.2 Alertmanager Configuration

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-config
  namespace: prometheus
stringData:
  alertmanager.yaml: |
    global:
      slack_api_url: 'https://hooks.slack.com/services/xxx/yyy/zzz'

    route:
      receiver: 'slack-notifications'
      group_by: ['alertname', 'severity']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      routes:
      - match:
          severity: critical
        receiver: 'slack-critical'

    receivers:
    - name: 'slack-notifications'
      slack_configs:
      - channel: '#gpu-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    - name: 'slack-critical'
      slack_configs:
      - channel: '#gpu-critical'
        title: 'CRITICAL: {{ .GroupLabels.alertname }}'
```

---

## 8. Horizontal Pod Autoscaling (HPA) với GPU Metrics

### 8.1 Prometheus Adapter Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-adapter-config
  namespace: prometheus
data:
  config.yaml: |
    rules:
    - seriesQuery: 'DCGM_FI_DEV_GPU_UTIL{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "gpu_utilization"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'

    - seriesQuery: 'DCGM_FI_DEV_FB_USED{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "gpu_memory_used"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

### 8.2 HPA sử dụng GPU Metrics

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpu-workload-hpa
  namespace: ml-training
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"  # Scale when GPU util > 80%
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

---

## 9. Custom Metrics Configuration

### 9.1 CSV Configuration cho dcgm-exporter

dcgm-exporter sử dụng CSV file để configure metrics:

```csv
# Format: DCGM_FIELD_ID, Prometheus_metric_name, type, help_text
# Standard Metrics
DCGM_FI_DEV_GPU_UTIL,     gpu_utilization,     gauge, GPU utilization (%).
DCGM_FI_DEV_MEM_COPY_UTIL,mem_copy_utilization,gauge, Memory copy utilization (%).
DCGM_FI_DEV_FB_FREE,      fb_free,             gauge, Framebuffer free (MB).
DCGM_FI_DEV_FB_USED,      fb_used,             gauge, Framebuffer used (MB).
DCGM_FI_DEV_POWER_USAGE,  power_usage,         gauge, Power usage (W).
DCGM_FI_DEV_GPU_TEMP,     gpu_temp,            gauge, GPU temperature (C).
DCGM_FI_DEV_SM_CLOCK,     sm_clock,            gauge, SM clock (MHz).
DCGM_FI_DEV_MEM_CLOCK,    memory_clock,        gauge, Memory clock (MHz).

# Profiling Metrics (require SYS_ADMIN)
DCGM_FI_PROF_GR_ENGINE_ACTIVE,    gr_engine_active,    gauge, Graphics engine active ratio.
DCGM_FI_PROF_SM_ACTIVE,           sm_active,           gauge, SM active ratio.
DCGM_FI_PROF_SM_OCCUPANCY,        sm_occupancy,        gauge, SM occupancy ratio.
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,  tensor_active,       gauge, Tensor core active ratio.
DCGM_FI_PROF_DRAM_ACTIVE,         dram_active,         gauge, DRAM active ratio.
DCGM_FI_PROF_PCIE_TX_BYTES,       pcie_tx_bytes,       counter, PCIe TX bytes.
DCGM_FI_PROF_PCIE_RX_BYTES,       pcie_rx_bytes,       counter, PCIe RX bytes.
```

### 9.2 Mount Custom Config vào dcgm-exporter

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dcgm-exporter-config
  namespace: gpu-monitoring
data:
  custom-counters.csv: |
    DCGM_FI_DEV_GPU_UTIL,     gpu_utilization,     gauge, GPU utilization (%).
    DCGM_FI_DEV_FB_USED,      fb_used,             gauge, Framebuffer used (MB).
    DCGM_FI_DEV_POWER_USAGE,  power_usage,         gauge, Power usage (W).
    DCGM_FI_DEV_GPU_TEMP,     gpu_temp,            gauge, GPU temperature (C).
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
spec:
  template:
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
        args:
        - "-f"
        - "/etc/dcgm-exporter/custom-counters.csv"
        volumeMounts:
        - name: config
          mountPath: /etc/dcgm-exporter
      volumes:
      - name: config
        configMap:
          name: dcgm-exporter-config
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Issue 1: dcgm-exporter không show metrics

```bash
# Check if dcgm-exporter pod is running
kubectl get pods -l app.kubernetes.io/name=dcgm-exporter

# Check logs
kubectl logs -l app.kubernetes.io/name=dcgm-exporter

# Test metrics endpoint directly
kubectl exec -it <dcgm-exporter-pod> -- curl localhost:9400/metrics
```

#### Issue 2: Profiling metrics không available

Profiling metrics cần `SYS_ADMIN` capability:

```yaml
securityContext:
  capabilities:
    add: ["SYS_ADMIN"]
```

#### Issue 3: Pod metrics không xuất hiện

Kiểm tra kubelet pod-resources API:

```bash
# Verify pod-resources socket exists
ls -la /var/lib/kubelet/pod-resources/

# Check DCGM_EXPORTER_KUBERNETES env variable
kubectl get pod <dcgm-exporter-pod> -o yaml | grep -A2 DCGM_EXPORTER_KUBERNETES
```

### 10.2 DCGM CLI Commands

```bash
# List all GPUs
dcgmi discovery -l

# Get GPU info
dcgmi discovery -i

# Run diagnostics
dcgmi diag -r 1  # Quick diagnostic
dcgmi diag -r 2  # Medium diagnostic
dcgmi diag -r 3  # Long diagnostic (30+ minutes)

# Get real-time stats
dcgmi dmon -e 155,150,156  # GPU util, memory, power

# Check health
dcgmi health -c
dcgmi health -g 0

# Field IDs:
# 155 = GPU Utilization
# 150 = Power Usage
# 156 = Total Memory
# 252 = GPU Temperature
```

### 10.3 Debug PromQL Queries

```promql
# Check if metrics are being scraped
up{job="dcgm-exporter"}

# List all DCGM metrics
{__name__=~"DCGM.*"}

# Check specific GPU
DCGM_FI_DEV_GPU_UTIL{gpu="0"}

# Debug pod labeling
DCGM_FI_DEV_GPU_UTIL{pod!=""}
```

---

## 11. Best Practices

### 11.1 Security

```yaml
# Use non-root where possible
securityContext:
  runAsNonRoot: true
  runAsUser: 1000

# Only add SYS_ADMIN for profiling metrics
# Standard metrics work without it
securityContext:
  capabilities:
    add: ["SYS_ADMIN"]  # Only if profiling needed
```

### 11.2 Resource Management

```yaml
# Set appropriate resource limits for dcgm-exporter
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 256Mi
```

### 11.3 Scrape Interval

```yaml
# 15s is recommended for GPU metrics
# Too frequent = overhead
# Too slow = miss spikes
endpoints:
- port: metrics
  interval: 15s
```

### 11.4 Label Cardinality

Tránh high cardinality labels để không overload Prometheus:

```yaml
# Good: Static labels
DCGM_FI_DEV_GPU_UTIL{gpu="0", Hostname="node-1", namespace="ml"}

# Bad: Dynamic labels với unique values
DCGM_FI_DEV_GPU_UTIL{request_id="uuid-xxxxx"}  # Don't do this
```

---

## 12. Tổng Kết

### 12.1 DCGM Monitoring Stack Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Complete Stack Overview                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Interface Layer                                            │
│  └── Grafana Dashboards (Dashboard ID: 12239)                   │
│                                                                  │
│  Alert Layer                                                     │
│  └── Alertmanager → Slack/PagerDuty/Email                       │
│                                                                  │
│  Storage Layer                                                   │
│  └── Prometheus (TSDB)                                          │
│                                                                  │
│  Export Layer                                                    │
│  └── dcgm-exporter (DaemonSet on GPU nodes)                     │
│                                                                  │
│  Collection Layer                                                │
│  └── DCGM Daemon (on each node)                                 │
│                                                                  │
│  Hardware Layer                                                  │
│  └── NVIDIA GPUs (A100, V100, T4, etc.)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Key Takeaways

1. **DCGM** cung cấp GPU metrics chi tiết ở hardware level
2. **dcgm-exporter** bridge DCGM với Prometheus ecosystem
3. **Pod-level metrics** cho phép tracking GPU usage per workload
4. **Alerting** cần thiết cho production GPU clusters
5. **HPA với GPU metrics** cho phép autoscaling dựa trên GPU utilization

### 12.3 Quick Start Commands

```bash
# 1. Install Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n prometheus --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# 2. Install dcgm-exporter
helm install dcgm gpu-helm-charts/dcgm-exporter

# 3. Access Grafana
kubectl port-forward -n prometheus svc/prometheus-grafana 3000:80

# 4. Import Dashboard ID: 12239

# 5. Run test workload
kubectl apply -f dcgmproftester-pod.yaml
```

---

## References

- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [dcgm-exporter GitHub](https://github.com/NVIDIA/dcgm-exporter)
- [Monitoring GPUs in Kubernetes with DCGM](https://developer.nvidia.com/blog/monitoring-gpus-in-kubernetes-with-dcgm/)
- [NVIDIA Grafana Dashboard](https://grafana.com/grafana/dashboards/12239)
- [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator)
