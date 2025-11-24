# Kong Gateway - Deep Technical Guide

## 1. Tổng quan

Kong Gateway là một **cloud-native API gateway** được xây dựng trên nền tảng **OpenResty** (NGINX + LuaJIT). Nó cung cấp các khả năng:

- **Traffic Management** - Routing, load balancing, rate limiting
- **Security** - Authentication, authorization, encryption
- **Observability** - Logging, metrics, tracing
- **Extensibility** - Plugin architecture với Lua/Go/JavaScript/Python
- **AI Gateway** - Quản lý traffic đến LLM providers

### 1.1. Kong AI Gateway

Kong AI Gateway là một **connectivity và governance layer** cho các ứng dụng AI-native:

- **Semantic routing** - Định tuyến thông minh dựa trên ngữ nghĩa
- **Security** - Bảo mật và kiểm soát truy cập
- **Observation** - Giám sát và logging
- **Acceleration** - Tăng tốc với caching
- **Governance** - Quản trị và kiểm soát chi phí

## 2. Core Architecture (Deep Dive)

### 2.1. Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     Kong Gateway                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Plugin Layer (Lua/Go/JS)               │   │
│  │   Authentication │ Rate Limiting │ Transformations  │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Kong Core (Lua)                         │   │
│  │   Router │ Balancer │ PDK │ Admin API │ Clustering  │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              OpenResty                               │   │
│  │   LuaJIT │ lua-resty-* libraries │ cosockets        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              NGINX                                   │   │
│  │   Event Loop │ HTTP/Stream │ SSL/TLS │ Upstream     │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Data Store: PostgreSQL │ DB-less (LMDB) │ Hybrid Mode    │
└─────────────────────────────────────────────────────────────┘
```

**Tại sao OpenResty + LuaJIT?**

| Component | Vai trò | Lợi ích |
|-----------|---------|---------|
| **NGINX** | HTTP server & reverse proxy | Event-driven, non-blocking I/O, C10K+ connections |
| **OpenResty** | Control flow layer cho NGINX | Cho phép extend NGINX với Lua scripts |
| **LuaJIT** | Just-In-Time compiler | Nhanh hơn V8, ~100KB footprint, tracing compiler |

### 2.2. Core Entities

```
┌──────────────────────────────────────────────────────────────────┐
│                         Consumer                                  │
│                    (API key, OAuth, JWT)                         │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                          Route                                    │
│            (paths, hosts, methods, headers)                       │
│                     ┌─────────────────┐                          │
│                     │    Plugins      │                          │
│                     └─────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Service                                   │
│              (upstream URL, protocol, timeout)                    │
│                     ┌─────────────────┐                          │
│                     │    Plugins      │                          │
│                     └─────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Upstream                                   │
│         (load balancing, health checks, circuit breaker)          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
│  │  Target 1  │  │  Target 2  │  │  Target 3  │                  │
│  │ weight=100 │  │ weight=100 │  │ weight=50  │                  │
│  └────────────┘  └────────────┘  └────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

**Entity Relationships:**

| Entity | Mô tả | Quan hệ |
|--------|-------|---------|
| **Consumer** | Client/user identity | Gắn với credentials (API key, JWT, OAuth) |
| **Route** | URL pattern matching | Thuộc về 1 Service, có nhiều Plugins |
| **Service** | Backend service abstraction | Trỏ đến URL hoặc Upstream |
| **Upstream** | Load balancer virtual host | Chứa nhiều Targets |
| **Target** | Backend server instance | IP:port với weight |
| **Plugin** | Middleware logic | Gắn vào Route/Service/Consumer/Global |

## 3. Request Lifecycle & Plugin Phases

### 3.1. Request Flow

```
Client Request
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NGINX Event Loop                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌────────┐  ┌──────────────────┐    │
│  │  init   │─▶│certificate│─▶│rewrite │─▶│     access       │    │
│  │ _worker │  │  (SSL)   │  │        │  │ (auth, rate-limit)│    │
│  └─────────┘  └─────────┘  └────────┘  └──────────────────┘    │
│                                                  │               │
│                                                  ▼               │
│                                         ┌──────────────┐        │
│                                         │   Upstream   │        │
│                                         │  (balancer)  │        │
│                                         └──────────────┘        │
│                                                  │               │
│                                                  ▼               │
│  ┌─────────┐  ┌─────────────┐  ┌─────────────┐  │               │
│  │   log   │◀─│ body_filter │◀─│header_filter│◀─┘               │
│  │         │  │  (chunks)   │  │ (response)  │                  │
│  └─────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Client Response
```

### 3.2. Plugin Phases (Chi tiết)

| Phase | Thời điểm | Use Case | PDK Available |
|-------|-----------|----------|---------------|
| **init_worker** | Worker process startup | Initialize shared resources, timers | Limited |
| **configure** | Config changes (v3.4+) | Dynamic reconfiguration | Limited |
| **certificate** | SSL handshake | Custom certificate logic, mTLS | `kong.client.tls` |
| **rewrite** | Before routing | URL rewriting, early transformations | `kong.request` |
| **access** | Before upstream | Auth, rate-limit, request transform | Full PDK |
| **response** | Full response buffered | Body transformation (single callback) | `kong.response` |
| **header_filter** | Response headers received | Modify response headers | `kong.response` |
| **body_filter** | Response body chunks | Stream processing, compression | `kong.response` |
| **log** | After response sent | Logging, metrics, analytics | Read-only |

**Lưu ý quan trọng:**
- `response` phase buffers toàn bộ response → tăng memory usage
- Không thể dùng cả `response` và `header_filter/body_filter` cùng lúc
- `log` phase không ảnh hưởng latency của client nhưng ảnh hưởng throughput

### 3.3. Plugin Priority (Execution Order)

```lua
-- Priority càng cao → chạy trước
PRIORITY = 1000000  -- Pre-function (chạy đầu tiên)
PRIORITY = 1250     -- Key Authentication
PRIORITY = 1005     -- JWT
PRIORITY = 910      -- Rate Limiting
PRIORITY = 801      -- Request Transformer
PRIORITY = 800      -- Response Transformer
PRIORITY = 12       -- File Log
PRIORITY = 9        -- HTTP Log
PRIORITY = -1000    -- Post-function (chạy cuối cùng)
```

**Dynamic Plugin Ordering (Enterprise):**

```bash
# Rate-limit TRƯỚC authentication
curl -X POST http://localhost:8001/plugins \
  --data name=rate-limiting \
  --data config.minute=5 \
  --data config.policy=local \
  --data config.limit_by=ip \
  --data ordering.before.access=key-auth
```

## 4. Database & Storage Architecture

### 4.1. Storage Options

| Mode | Database | Use Case | Pros | Cons |
|------|----------|----------|------|------|
| **Traditional** | PostgreSQL | Full features | Full CRUD, clustering | DB dependency |
| **DB-less** | LMDB (in-memory) | K8s native, GitOps | No DB, declarative | No Admin API writes |
| **Hybrid** | CP: PostgreSQL, DP: LMDB | Enterprise | Secure, scalable | Complex setup |

### 4.2. LMDB Storage Engine

Kong sử dụng **LMDB (Lightning Memory-Mapped Database)** cho DB-less và Hybrid mode:

```
┌─────────────────────────────────────────┐
│           LMDB Characteristics          │
├─────────────────────────────────────────┤
│ • Size: ~100KB compiled object code     │
│ • Multi-process: Native support         │
│ • Reads: Never blocked (even during     │
│   write transactions)                   │
│ • ACID: Full transaction support        │
│ • MVCC: Multi-version concurrency       │
│ • Persistent: Survives restarts         │
└─────────────────────────────────────────┘
```

**Cấu hình LMDB:**

```bash
# Maximum memory map size (default: 2048m)
# Có thể set lớn (vài GB) cho future growth + MVCC headroom
KONG_LMDB_MAP_SIZE=4096m
```

### 4.3. Hybrid Mode Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Control Plane (CP)                          │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────────────────┐ │
│  │  Admin API   │  │  Kong Core  │  │      PostgreSQL DB        │ │
│  │  (port 8001) │  │             │  │  (services, routes, etc.) │ │
│  └──────────────┘  └─────────────┘  └───────────────────────────┘ │
│                           │                                        │
│                    Port 8005 (cluster)                             │
└───────────────────────────┼────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   Data Plane 1    │ │   Data Plane 2    │ │   Data Plane 3    │
│  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │
│  │ Proxy       │  │ │  │ Proxy       │  │ │  │ Proxy       │  │
│  │ (port 8000) │  │ │  │ (port 8000) │  │ │  │ (port 8000) │  │
│  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │
│  ┌─────────────┐  │ │  ┌─────────────┐  │ │  ┌─────────────┐  │
│  │ LMDB Cache  │  │ │  │ LMDB Cache  │  │ │  │ LMDB Cache  │  │
│  │ (encrypted) │  │ │  │ (encrypted) │  │ │  │ (encrypted) │  │
│  └─────────────┘  │ │  └─────────────┘  │ │  └─────────────┘  │
│   Region: US-East │ │   Region: EU-West │ │   Region: AP-South│
└───────────────────┘ └───────────────────┘ └───────────────────┘
```

**Hybrid Mode Benefits:**

| Benefit | Mô tả |
|---------|-------|
| **Deployment Flexibility** | DP có thể ở nhiều regions/zones khác nhau |
| **Increased Reliability** | DP tiếp tục hoạt động khi CP down (có cached config) |
| **Traffic Reduction** | Chỉ CP cần kết nối DB |
| **Security** | DP bị compromise không ảnh hưởng cluster khác |
| **Ease of Management** | Chỉ cần tương tác với CP |

## 5. Load Balancing & Health Checks

### 5.1. Load Balancing Algorithms

```yaml
# Upstream configuration
upstreams:
  - name: my-upstream
    algorithm: round-robin  # hoặc consistent-hashing, least-connections, latency
    slots: 10000
    hash_on: none  # header, cookie, ip, path, query_arg
    hash_fallback: none
```

| Algorithm | Mô tả | Use Case |
|-----------|-------|----------|
| **round-robin** | Weighted round-robin | General purpose, default |
| **consistent-hashing** | Hash-based sticky sessions | Cache hit ratio, stateful |
| **least-connections** | Route to least busy target | Long-running connections |
| **latency** | EWMA-based lowest latency | Real-time, latency-sensitive |

### 5.2. Health Checks

```yaml
upstreams:
  - name: my-upstream
    healthchecks:
      threshold: 50  # % capacity để upstream healthy

      # Active health checks
      active:
        healthy:
          interval: 5        # Check mỗi 5s khi healthy
          successes: 2       # 2 success → mark healthy
          http_statuses: [200, 302]
        unhealthy:
          interval: 1        # Check mỗi 1s khi unhealthy
          http_failures: 5   # 5 failures → mark unhealthy
          tcp_failures: 2
          timeouts: 3
        type: http           # http, https, tcp
        http_path: /health

      # Passive health checks (Circuit Breaker)
      passive:
        healthy:
          successes: 5
          http_statuses: [200, 201, 202, 203, 204, 205, 206, 207, 208, 226]
        unhealthy:
          http_failures: 5
          http_statuses: [429, 500, 503]
          tcp_failures: 2
          timeouts: 7
```

**Active vs Passive:**

| Type | Mechanism | Pros | Cons |
|------|-----------|------|------|
| **Active** | Kong gửi probe requests | Tự động re-enable targets | Tạo extra traffic |
| **Passive** | Dựa trên real traffic | Không extra traffic | Cần manual re-enable |

## 6. Performance Optimization

### 6.1. Connection Tuning

```bash
# nginx.conf equivalents
KONG_NGINX_WORKER_PROCESSES=auto           # hoặc 4, 8 (match CPU cores)
KONG_UPSTREAM_KEEPALIVE_POOL_SIZE=512      # connections per upstream
KONG_UPSTREAM_KEEPALIVE_MAX_REQUESTS=100000  # requests per connection
KONG_UPSTREAM_KEEPALIVE_IDLE_TIMEOUT=60    # idle timeout (seconds)

# Client-side keepalive
KONG_NGINX_HTTP_KEEPALIVE_REQUESTS=100000  # requests per client connection
KONG_NGINX_HTTP_KEEPALIVE_TIMEOUT=60s

# Worker connections
KONG_NGINX_MAIN_WORKER_CONNECTIONS=16384   # max connections per worker
```

### 6.2. Performance Best Practices

```yaml
# Recommended production config
nginx_worker_processes: auto      # Hoặc CPU cores - 1
upstream_keepalive_pool_size: 512
upstream_keepalive_max_requests: 100000
nginx_http_keepalive_requests: 100000
proxy_access_log: "off"           # Disable cho high throughput
dns_stale_ttl: 3600               # Cache DNS longer
```

**Ulimit considerations:**
- Kong defaults to `ulimit` value với upper bound 16384
- Nếu `ulimit` < 16384 → tăng lên
- Check cả client và upstream server ulimit

### 6.3. Memory Optimization

```bash
# Reduce memory footprint
KONG_LUA_SOCKET_POOL_SIZE=30      # Connection pool size
KONG_MEM_CACHE_SIZE=128m          # In-memory cache for entities
KONG_LMDB_MAP_SIZE=2048m          # LMDB memory map

# Disable unused features
KONG_UNTRUSTED_LUA=off            # Disable sandboxed Lua
KONG_NGINX_HTTP_LUA_REGEX_CACHE_MAX_ENTRIES=8192
```

## 7. Plugin Development Kit (PDK)

### 7.1. Plugin Structure

```
my-plugin/
├── handler.lua      # Core logic, phase hooks
├── schema.lua       # Config validation schema
├── daos.lua         # Custom database entities (optional)
├── migrations/      # DB migrations (optional)
│   ├── init.lua
│   └── 000_base.lua
└── api.lua          # Custom Admin API endpoints (optional)
```

### 7.2. Handler Template

```lua
-- handler.lua
local MyPlugin = {
  VERSION = "1.0.0",
  PRIORITY = 1000,  -- Execution order
}

function MyPlugin:init_worker()
  -- Worker initialization
  kong.log.debug("Worker initialized")
end

function MyPlugin:certificate(conf)
  -- SSL certificate phase
end

function MyPlugin:rewrite(conf)
  -- Before routing
end

function MyPlugin:access(conf)
  -- Main plugin logic
  local headers = kong.request.get_headers()
  local path = kong.request.get_path()

  -- Example: Add custom header
  kong.service.request.set_header("X-Custom-Header", "value")

  -- Example: Rate limit check
  if should_rate_limit() then
    return kong.response.exit(429, { message = "Rate limited" })
  end
end

function MyPlugin:header_filter(conf)
  -- Modify response headers
  kong.response.set_header("X-Response-Time", ngx.now() - ngx.req.start_time())
end

function MyPlugin:body_filter(conf)
  -- Process response body chunks
end

function MyPlugin:log(conf)
  -- Logging phase
  kong.log.info("Request completed: ", kong.request.get_path())
end

return MyPlugin
```

### 7.3. PDK Modules

| Module | Mô tả | Example |
|--------|-------|---------|
| `kong.request` | Access request data | `kong.request.get_header("X-API-Key")` |
| `kong.response` | Modify response | `kong.response.exit(403, {error = "Forbidden"})` |
| `kong.service.request` | Modify upstream request | `kong.service.request.set_header()` |
| `kong.service.response` | Access upstream response | `kong.service.response.get_header()` |
| `kong.client` | Client info | `kong.client.get_ip()` |
| `kong.router` | Routing info | `kong.router.get_route()` |
| `kong.log` | Logging | `kong.log.err("Error message")` |
| `kong.cache` | Caching | `kong.cache:get("key", nil, callback)` |
| `kong.db` | Database access | `kong.db.consumers:select()` |

## 8. Yêu cầu hệ thống (AI Gateway)

| Component | Minimum Version |
|-----------|-----------------|
| Kong Gateway | 3.6+ |
| decK (CLI tool) | 1.43+ |

**Bạn cần:**
- Kong Gateway Enterprise license (hoặc Konnect account)
- API key từ LLM provider (OpenAI, Azure, AWS Bedrock, etc.)
- Konnect personal access token (nếu dùng cloud)

## 9. AI Gateway Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Client    │────▶│   Kong AI Gateway   │────▶│   LLM Provider  │
│ Application │     │   (Proxy + Plugins) │     │ (OpenAI, Azure) │
└─────────────┘     └─────────────────────┘     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼─────┐      ┌──────▼──────┐
              │  Caching  │      │   Logging   │
              │  (Redis)  │      │  (Metrics)  │
              └───────────┘      └─────────────┘
```

## 10. Quy trình Deploy (AI Gateway)

### 10.1. Quick Start với Konnect (Cloud)

```bash
# Set token
export KONNECT_TOKEN="your-personal-access-token"

# Chạy quickstart script - tự động provision Control Plane và Data Plane
curl -Ls https://get.konghq.com/quickstart | bash -s -- -k $KONNECT_TOKEN --deck-output
```

### 10.2. Cấu hình từng bước

**Bước 1: Tạo Gateway Service**

```bash
echo '_format_version: "3.0"
services:
  - name: llm-service
    url: http://localhost:32000' | deck gateway apply -
```

**Bước 2: Tạo Route**

```bash
echo '_format_version: "3.0"
routes:
  - name: openai-chat
    service:
      name: llm-service
    paths:
    - "/chat"' | deck gateway apply -
```

**Bước 3: Enable AI Proxy Plugin**

```bash
echo '_format_version: "3.0"
plugins:
  - name: ai-proxy
    config:
      route_type: llm/v1/chat
      model:
        provider: openai' | deck gateway apply -
```

### 10.3. Kiểm tra deployment

```bash
# Test endpoint
curl -X POST "$KONNECT_PROXY_URL/chat" \
  -H "Authorization: Bearer $OPENAI_KEY" \
  --json '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 11. Các Plugin AI quan trọng

### 11.1. Core Plugins

| Plugin | Mô tả |
|--------|-------|
| **AI Proxy** | Proxy cơ bản đến LLM providers |
| **AI Proxy Advanced** | Load balancing, retry, fallback giữa nhiều providers |
| **AI Semantic Cache** | Cache response dựa trên semantic similarity |
| **AI Rate Limiting** | Giới hạn request theo token/request count |

### 11.2. Security Plugins

| Plugin | Mô tả |
|--------|-------|
| **AI Prompt Guard** | Allow/deny list cho prompts |
| **AI PII Sanitizer** | Phát hiện và ẩn PII (20+ categories, 12 languages) |
| **Azure Content Safety** | Tích hợp Azure content moderation |
| **AWS Guardrails** | Tích hợp AWS Bedrock Guardrails |

### 11.3. Prompt Engineering Plugins

| Plugin | Mô tả |
|--------|-------|
| **AI Prompt Template** | Template prompts với variables |
| **AI Prompt Decorator** | Prepend/append system messages |
| **AI RAG Injector** | Tự động inject RAG data vào prompts |

## 12. Best Practices (AI Gateway)

### 12.1. Security

```yaml
# Cấu hình Prompt Guard để chặn injection
plugins:
  - name: ai-prompt-guard
    config:
      allow_patterns:
        - "^(What|How|Why|Can you)"
      deny_patterns:
        - "ignore previous instructions"
        - "system prompt"
```

**Khuyến nghị:**
- Luôn enable **AI Prompt Guard** để chặn prompt injection
- Sử dụng **PII Sanitizer** cho dữ liệu nhạy cảm
- Không hardcode API keys - dùng environment variables hoặc secrets manager

### 12.2. Cost Optimization

```yaml
# Enable semantic caching để giảm chi phí
plugins:
  - name: ai-semantic-cache
    config:
      vectordb:
        strategy: redis
        threshold: 0.85  # Similarity threshold
      cache_ttl: 3600   # 1 hour
```

**Khuyến nghị:**
- Enable **Semantic Cache** cho các queries lặp lại
- Sử dụng **Rate Limiting** theo tokens để kiểm soát chi phí
- **Prompt Compression** để giảm input tokens

### 12.3. High Availability

```yaml
# Load balancing với fallback
plugins:
  - name: ai-proxy-advanced
    config:
      targets:
        - route_type: llm/v1/chat
          model:
            name: gpt-4
            provider: openai
          weight: 70
        - route_type: llm/v1/chat
          model:
            name: claude-3-sonnet
            provider: anthropic
          weight: 30
      balancer:
        algorithm: round-robin
```

**Khuyến nghị:**
- Cấu hình **multi-provider fallback** (OpenAI → Azure → Anthropic)
- Sử dụng **lowest-latency** algorithm cho real-time apps
- Enable **retry mechanism** cho transient failures

### 12.4. Observability

```yaml
# Enable logging và metrics
plugins:
  - name: ai-audit-log
    config:
      log_prompts: true
      log_responses: true
      include_usage_stats: true
```

**Khuyến nghị:**
- Bật **audit logging** để track tất cả LLM calls
- Expose **Prometheus metrics** cho monitoring
- Tích hợp với **Konnect Analytics** để visualize usage

### 12.5. GitOps Workflow

```bash
# Lưu config vào file
deck gateway dump -o kong.yaml

# Apply từ file (version controlled)
deck gateway apply kong.yaml

# Diff trước khi apply
deck gateway diff kong.yaml
```

**Khuyến nghị:**
- Sử dụng **decK** cho declarative configuration
- Version control tất cả config files
- Dùng **Terraform** cho infrastructure as code
- Implement **CI/CD pipeline** cho config changes

## 13. Deployment Modes

Kong AI Gateway hỗ trợ nhiều deployment modes:

| Mode | Mô tả | Use Case |
|------|-------|----------|
| **Konnect (SaaS)** | Fully managed cloud | Quick start, low ops |
| **Self-hosted** | On-premise deployment | Data sovereignty |
| **Hybrid** | Control plane cloud, data plane on-prem | Best of both |
| **DB-less** | Declarative config, no database | Kubernetes native |
| **Kubernetes** | Kong Ingress Controller | Cloud-native apps |

## 14. Ví dụ Production Setup (AI Gateway)

```yaml
# kong.yaml - Production configuration
_format_version: "3.0"

services:
  - name: ai-service
    url: http://localhost:32000

routes:
  - name: ai-chat
    service:
      name: ai-service
    paths:
      - /v1/chat

plugins:
  # AI Proxy với model restriction
  - name: ai-proxy
    route: ai-chat
    config:
      route_type: llm/v1/chat
      model:
        provider: openai
        name: gpt-4  # Restrict to specific model
      auth:
        header_name: Authorization
        header_value: Bearer ${OPENAI_API_KEY}

  # Rate limiting
  - name: rate-limiting
    route: ai-chat
    config:
      minute: 60
      policy: local

  # Security - Prompt Guard
  - name: ai-prompt-guard
    route: ai-chat
    config:
      deny_patterns:
        - "ignore.*instructions"
        - "reveal.*system.*prompt"

  # Caching
  - name: ai-semantic-cache
    route: ai-chat
    config:
      vectordb:
        strategy: redis
        connection:
          host: redis.local
          port: 6379
      cache_ttl: 3600
```

## 15. Troubleshooting

### 15.1. Common Issues

| Issue | Solution |
|-------|----------|
| 401 Unauthorized | Kiểm tra API key và Authorization header |
| 429 Rate Limited | Tăng rate limit hoặc implement backoff |
| 504 Gateway Timeout | Tăng timeout config hoặc check LLM provider status |
| Cache miss rate cao | Giảm similarity threshold |

### 15.2. Debug Commands

```bash
# Check Kong status
curl -i http://localhost:8001/status

# List all services
curl http://localhost:8001/services

# Check plugin config
curl http://localhost:8001/plugins

# View logs
tail -f /usr/local/kong/logs/error.log
```

### 15.3. Performance Debugging

```bash
# Check worker connections
curl http://localhost:8001/status | jq '.server.connections_active'

# Check upstream health
curl http://localhost:8001/upstreams/my-upstream/health

# Check plugin latency (via HTTP Log)
# Enable with: "config.custom_fields_by_lua.latencies"

# Check memory usage
curl http://localhost:8001/status | jq '.memory'

# Validate configuration
kong config parse /etc/kong/kong.conf
```

## 16. Resources

**Official Documentation:**
- [Kong Gateway Docs](https://docs.konghq.com/gateway/latest/)
- [Kong AI Gateway](https://developer.konghq.com/ai-gateway/)
- [Kong Gateway Admin API](https://docs.konghq.com/gateway/latest/admin-api/)
- [PDK Reference](https://developer.konghq.com/gateway/pdk/reference/)
- [Plugin Hub](https://docs.konghq.com/hub/)

**Tools:**
- [decK CLI Reference](https://docs.konghq.com/deck/latest/)
- [Pongo - Plugin Development](https://github.com/Kong/kong-pongo)

**Architecture Deep Dives:**
- [Hybrid Mode](https://docs.konghq.com/gateway/latest/production/deployment-topologies/hybrid-mode/)
- [Performance Optimization](https://developer.konghq.com/gateway/performance/optimize/)
- [Load Balancing](https://developer.konghq.com/gateway/load-balancing/)
- [Health Checks](https://docs.konghq.com/gateway/latest/how-kong-works/health-checks/)
