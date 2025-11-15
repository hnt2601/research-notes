# Hướng Dẫn Sử Dụng Claude Skills

## Giới Thiệu

Claude Skills là hệ thống plugin chuyên biệt giúp mở rộng khả năng của Claude Code với các chuyên môn cụ thể. Repository **wshobson/agents** cung cấp **63 plugin chuyên biệt** được tổ chức thành **23 danh mục**, bao gồm:

- **85 agent chuyên biệt** trên nhiều lĩnh vực
- **55 skill module** với kiến trúc tiết kiệm token
- **44 công cụ phát triển** tự động hóa
- **15 workflow điều phối đa-agent**

---

## Cài Đặt

### Bước 1: Thêm Marketplace

```bash
/plugin marketplace add wshobson/agents
```

Lệnh này giúp bạn khám phá tất cả 63 plugin mà không tải bất kỳ agent hoặc công cụ nào vào context.

### Bước 2: Duyệt và Cài Đặt Plugin

**Xem danh sách plugin:**
```bash
/plugin
```

**Cài đặt plugin cần thiết:**
```bash
/plugin install python-development
/plugin install kubernetes-operations
/plugin install security-scanning
```

> **Nguyên tắc quan trọng:** Chỉ cài đặt những gì bạn cần. Mỗi plugin chỉ tải các agent, command và skill cụ thể của nó (khoảng 300 token/plugin), giúp tối ưu hiệu suất.

---

## Kiến Trúc Skills

### Mô Hình 3 Tầng (Progressive Disclosure)

Mỗi skill được cấu trúc theo 3 tầng để tối ưu việc sử dụng token:

1. **Metadata (YAML Frontmatter)**
   - Luôn được tải
   - Chứa tên skill và tiêu chí kích hoạt
   - Token usage tối thiểu

2. **Instructions**
   - Tải khi skill được kích hoạt
   - Chứa hướng dẫn cốt lõi và pattern triển khai
   - Cung cấp logic xử lý chính

3. **Resources**
   - Tải theo yêu cầu (on-demand)
   - Chứa ví dụ, template, và tài liệu tham khảo
   - Chỉ load khi thực sự cần thiết

### Cơ Chế Kích Hoạt Tự Động

Skills tự động kích hoạt khi Claude phát hiện các pattern liên quan:

**Ví dụ 1:** Kubernetes
```
"Set up Kubernetes deployment with Helm chart"
→ Kích hoạt: helm-chart-scaffolding + k8s-manifest-generator
```

**Ví dụ 2:** LLM Development
```
"Build a RAG system for document Q&A"
→ Kích hoạt: rag-implementation + prompt-engineering-patterns
```

**Ví dụ 3:** Backend API
```
"Create FastAPI microservice with async patterns"
→ Kích hoạt: api-design-principles + fastapi-templates + async-patterns
```

---

## Danh Mục Plugin Chính

### 1. Language Development (11 Plugin)

#### Python Development
```bash
/plugin install python-development
```
**Bao gồm:**
- 5 specialized skills
- Async patterns
- Testing frameworks (pytest)
- Package management

**Ví dụ sử dụng:**
```bash
/python-development:python-scaffold fastapi-microservice
```

#### JavaScript/TypeScript
```bash
/plugin install javascript-typescript
```
**Bao gồm:**
- 4 specialized skills
- React/Next.js patterns
- Jest testing
- TypeScript best practices

#### Systems Programming
```bash
/plugin install systems-programming
```
**Ngôn ngữ:** Rust, Go, C++

#### JVM Languages
```bash
/plugin install jvm-languages
```
**Ngôn ngữ:** Java, Scala, C#

#### Scripting Languages
```bash
/plugin install scripting-languages
```
**Ngôn ngữ:** PHP, Ruby

#### Functional Languages
```bash
/plugin install functional-languages
```
**Ngôn ngữ:** Elixir

#### Embedded Systems
```bash
/plugin install embedded-development
```
**Platform:** ARM Cortex microcontrollers

---

### 2. Infrastructure & DevOps (9 Plugin)

#### Kubernetes Operations
```bash
/plugin install kubernetes-operations
```
**Bao gồm:**
- Kubernetes architect với 4 deployment-focused skills
- Helm chart scaffolding
- K8s manifest generator
- Production deployment patterns

**Ví dụ sử dụng:**
```
"Create production Kubernetes deployment with Helm chart"
→ Tự động kích hoạt kubernetes-architect với 4 specialized skills
```

#### Cloud Infrastructure
```bash
/plugin install cloud-infrastructure
```
**Platforms:** AWS, Azure, GCP
**Bao gồm:** 4 specialized skills cho mỗi cloud provider

#### CI/CD Automation
```bash
/plugin install cicd-automation
```
**Bao gồm:**
- 4 specialized skills
- Pipeline optimization
- Automated testing integration

#### Deployment Strategies
```bash
/plugin install deployment-strategies
```
**Patterns:**
- Blue-green deployment
- Canary releases
- Rolling updates

#### Incident Response
```bash
/plugin install incident-response
```

#### Observability
```bash
/plugin install observability
```

#### Distributed Debugging
```bash
/plugin install distributed-debugging
```

---

### 3. Backend & API Development (4 Plugin)

#### Backend Architecture
```bash
/plugin install backend-architecture
```
**Bao gồm:**
- API design principles
- Microservices patterns
- Database design

#### API Scaffolding
```bash
/plugin install api-scaffolding
```
**Frameworks:**
- FastAPI
- Express.js
- Django REST Framework

#### Database Architecture
```bash
/plugin install database-architecture
```

#### Data Engineering
```bash
/plugin install data-engineering
```

---

### 4. LLM & AI Development (3 Plugin)

#### LLM Application Development
```bash
/plugin install llm-development
```
**Bao gồm:**
- 4 specialized skills
- RAG implementation
- Prompt engineering patterns
- Vector database integration
- LLM evaluation frameworks

**Skills:**
- `rag-implementation`: Hệ thống Retrieval-Augmented Generation
- `prompt-engineering-patterns`: Kỹ thuật prompt engineering
- `vector-db-integration`: Tích hợp Pinecone, Weaviate, Qdrant
- `llm-evaluation`: Framework đánh giá chất lượng LLM

#### Machine Learning Operations
```bash
/plugin install mlops
```

---

### 5. Security (4 Plugin)

#### Security Scanning
```bash
/plugin install security-scanning
```
**Bao gồm:**
- SAST vulnerability scanning
- Code audit automation
- Security best practices

**Ví dụ sử dụng:**
```bash
/security-scanning:security-hardening --level comprehensive
```

#### Compliance
```bash
/plugin install compliance
```
**Standards:** SOC2, HIPAA, GDPR

#### API Security
```bash
/plugin install api-security
```

#### Web Security
```bash
/plugin install web-security
```
**Bảo vệ:** XSS, CSRF, SQL Injection

---

### 6. Testing & Quality (5 Plugin)

#### Unit Testing
```bash
/plugin install unit-testing
```
**Frameworks:** pytest, Jest

#### Code Review
```bash
/plugin install code-review
```

#### TDD Workflows
```bash
/plugin install tdd-workflows
```

#### Performance Analysis
```bash
/plugin install performance-analysis
```

#### Test Automation
```bash
/plugin install test-automation
```

---

### 7. Specialized Domains

#### Blockchain/Web3
```bash
/plugin install blockchain-web3
```
**Bao gồm:**
- Smart contract development
- Solidity best practices
- DeFi patterns

#### Quantitative Trading
```bash
/plugin install quant-trading
```

#### Payment Processing
```bash
/plugin install payment-processing
```

#### Game Development
```bash
/plugin install game-development
```

#### Accessibility Compliance
```bash
/plugin install accessibility
```
**Standards:** WCAG, ADA

---

### 8. Business & Marketing (7 Plugin)

#### SEO Optimization
```bash
/plugin install seo-optimization
```

#### Content Marketing
```bash
/plugin install content-marketing
```

#### Business Analytics
```bash
/plugin install business-analytics
```

#### HR Compliance
```bash
/plugin install hr-compliance
```

#### Sales Automation
```bash
/plugin install sales-automation
```

---

## Workflow Điều Phối Đa-Agent

### Full-Stack Feature Development

```bash
/full-stack-orchestration:full-stack-feature "user authentication with OAuth2"
```

**Agent được điều phối:**
1. Backend architect
2. Database architect
3. Frontend specialist
4. Testing engineer
5. Security auditor
6. DevOps engineer
7. Code reviewer

**Workflow:**
1. Sonnet agent lập kế hoạch tổng thể
2. Haiku agents thực thi các task cụ thể
3. Sonnet agent review và tối ưu

---

## Mô Hình Hybrid Sonnet/Haiku

Hệ thống sử dụng **hybrid model orchestration** để tối ưu hiệu suất:

### Phân Công Model

- **47 Haiku agents**: Xử lý các task xác định, nhanh chóng
  - Code generation
  - File manipulation
  - Simple transformations

- **97 Sonnet agents**: Xử lý reasoning phức tạp
  - Architecture design
  - Problem solving
  - Code review
  - Planning

### Pattern Điều Phối

```
Sonnet (Planning)
    ↓
Haiku (Execution)
    ↓
Sonnet (Review & Optimization)
```

**Ví dụ thực tế:**
```
1. Sonnet: Phân tích yêu cầu "Build microservices architecture"
2. Haiku: Tạo boilerplate code, config files
3. Haiku: Generate tests
4. Sonnet: Review architecture, suggest optimizations
5. Haiku: Apply changes
6. Sonnet: Final validation
```

---

## Đặc Điểm Kỹ Thuật Skills

### 1. Token Efficiency (Hiệu Quả Token)

- Chỉ những skill được kích hoạt mới load vào context
- Giảm thiểu overhead tính toán
- Trung bình 300 token/plugin

### 2. Composability (Khả Năng Kết Hợp)

Nhiều skill có thể kết hợp để giải quyết task phức tạp:

```
Backend Architect Agent
├── api-design-principles
├── fastapi-templates
├── async-patterns
└── database-integration
```

### 3. Clear Scope (Phạm Vi Rõ Ràng)

Tất cả 55 skills tuân thủ chuẩn hóa:
- **Naming**: Hyphen-case (ví dụ: `rag-implementation`)
- **Description**: Dưới 1024 ký tự
- **Activation triggers**: Explicit "Use when" criteria

### 4. Single Responsibility

Mỗi plugin tập trung vào "một việc và làm tốt việc đó":
- Python Development: Chỉ về Python
- Kubernetes Operations: Chỉ về K8s
- Security Scanning: Chỉ về bảo mật

---

## Ví Dụ Thực Tế

### Ví Dụ 1: Xây Dựng Microservice Python

**Input:**
```
"Create a FastAPI microservice with async database operations,
pytest tests, and Docker deployment"
```

**Skills được kích hoạt:**
1. `fastapi-templates` - Cấu trúc project
2. `async-patterns` - Async/await patterns
3. `pytest-patterns` - Test setup
4. `docker-optimization` - Container configuration

**Output:**
- FastAPI app với async endpoints
- SQLAlchemy async session
- Pytest fixtures và test cases
- Dockerfile + docker-compose.yml

### Ví Dụ 2: RAG System

**Input:**
```
"Build a document Q&A system using RAG with vector database"
```

**Skills được kích hoạt:**
1. `rag-implementation` - RAG architecture
2. `vector-db-integration` - Pinecone/Weaviate setup
3. `prompt-engineering-patterns` - Prompt optimization
4. `llm-evaluation` - Quality metrics

**Output:**
- Document ingestion pipeline
- Vector embedding generation
- Semantic search implementation
- Context-aware response generation

### Ví Dụ 3: Kubernetes Production Deployment

**Input:**
```
"Deploy microservices to production Kubernetes with
Helm charts, monitoring, and auto-scaling"
```

**Skills được kích hoạt:**
1. `helm-chart-scaffolding` - Helm structure
2. `k8s-manifest-generator` - K8s resources
3. `production-deployment-patterns` - Best practices
4. `observability-integration` - Prometheus/Grafana

**Output:**
- Helm chart với values.yaml
- Deployment, Service, Ingress manifests
- HorizontalPodAutoscaler
- ServiceMonitor cho Prometheus

### Ví Dụ 4: Security Audit

**Input:**
```
"Perform comprehensive security audit of the API codebase"
```

**Skills được kích hoạt:**
1. `sast-scanning` - Static analysis
2. `api-security-patterns` - REST/GraphQL security
3. `owasp-top-10` - Vulnerability checks
4. `compliance-validation` - Regulatory requirements

**Output:**
- Vulnerability report
- Security recommendations
- Code fixes
- Compliance checklist

---

## Best Practices

### 1. Cài Đặt Chọn Lọc

```bash
# ✅ TốT: Chỉ cài những gì cần
/plugin install python-development
/plugin install kubernetes-operations

# ❌ TRÁNH: Cài tất cả plugins
```

### 2. Sử Dụng Descriptive Prompts

```bash
# ✅ TỐT: Mô tả rõ ràng, đầy đủ
"Create a FastAPI microservice with async PostgreSQL operations,
authentication middleware, and comprehensive pytest tests"

# ❌ TRÁNH: Mơ hồ
"Make an API"
```

### 3. Kết Hợp Skills

```bash
# Yêu cầu phức tạp tự động kết hợp nhiều skills
"Build a full-stack app with:
- Next.js frontend
- FastAPI backend
- PostgreSQL database
- Docker deployment
- CI/CD pipeline"

→ Kích hoạt 10+ skills từ 5 plugins khác nhau
```

### 4. Kiểm Tra Token Usage

```bash
# Xem skills nào đang active
/context

# Unload plugin không dùng
/plugin uninstall [plugin-name]
```

### 5. Progressive Enhancement

Bắt đầu đơn giản, sau đó mở rộng:

```bash
# Giai đoạn 1: Basic setup
/plugin install python-development
"Create basic FastAPI app"

# Giai đoạn 2: Add testing
/plugin install unit-testing
"Add pytest tests"

# Giai đoạn 3: Add deployment
/plugin install kubernetes-operations
"Create K8s deployment"
```

---

## Khắc Phục Sự Cố

### Skill Không Được Kích Hoạt

**Nguyên nhân:**
- Prompt không đủ rõ ràng
- Plugin chưa được cài đặt
- Activation trigger không match

**Giải pháp:**
```bash
# 1. Kiểm tra plugin đã cài
/plugin list

# 2. Sử dụng prompt cụ thể hơn
"Create [specific framework] app with [specific features]"

# 3. Gọi trực tiếp command
/[plugin-name]:[command-name]
```

### Token Limit Issues

**Giải pháp:**
```bash
# Unload plugins không dùng
/plugin uninstall [unused-plugin]

# Sử dụng specific commands thay vì general prompts
/python-scaffold fastapi-microservice
# thay vì
"Create a Python microservice"
```

### Conflict Giữa Skills

**Giải pháp:**
- Sử dụng explicit commands
- Specify framework/tool trong prompt
- Cài một plugin cho một lần làm việc

---

## Tài Nguyên Tham Khảo

### Repository
- **GitHub:** https://github.com/wshobson/agents
- **Documentation:** https://github.com/wshobson/agents/tree/main/docs

### Documentation Files
- `agent-skills.md` - Chi tiết về skills
- `agents.md` - Agent documentation
- `architecture.md` - System architecture
- `plugins.md` - Plugin reference
- `usage.md` - Usage guides

### Community
- Issues: https://github.com/wshobson/agents/issues
- Discussions: https://github.com/wshobson/agents/discussions

---

## Kết Luận

Claude Skills từ repository **wshobson/agents** cung cấp một hệ sinh thái plugin mạnh mẽ và linh hoạt:

✅ **63 plugin chuyên biệt** phủ 23 lĩnh vực
✅ **85 specialized agents** với expertise sâu
✅ **55 modular skills** với progressive disclosure
✅ **Hybrid Sonnet/Haiku** orchestration
✅ **Token-efficient** architecture
✅ **Composable & extensible** design

Bằng cách chỉ cài đặt những gì bạn cần và tận dụng tính năng kích hoạt tự động, bạn có thể xây dựng các ứng dụng phức tạp một cách hiệu quả với sự hỗ trợ của AI chuyên môn cao.

---

**Phiên bản:** 1.0
**Cập nhật:** 2025-11-15
**Tác giả:** Based on wshobson/agents repository
