# Helm - Hướng Dẫn Viết Cấu Hình Chi Tiết

## Mục Lục
1. [Giới Thiệu Helm](#1-giới-thiệu-helm)
2. [Cấu Trúc Helm Chart](#2-cấu-trúc-helm-chart)
3. [Làm Việc Với Templates](#3-làm-việc-với-templates)
4. [Values và Quản Lý Cấu Hình](#4-values-và-quản-lý-cấu-hình)
5. [Template Functions và Pipelines](#5-template-functions-và-pipelines)
6. [Control Structures](#6-control-structures)
7. [Named Templates](#7-named-templates)
8. [Built-in Objects](#8-built-in-objects)
9. [Helm Hooks](#9-helm-hooks)
10. [Testing Charts](#10-testing-charts)
11. [Các Lệnh Helm Quan Trọng](#11-các-lệnh-helm-quan-trọng)
12. [Best Practices](#12-best-practices)

---

## 1. Giới Thiệu Helm

**Reference:** https://helm.sh/docs/intro/using_helm

### Helm là gì?
Helm là package manager cho Kubernetes, giúp bạn định nghĩa, cài đặt và nâng cấp các ứng dụng Kubernetes phức tạp.

### Ba Khái Niệm Cơ Bản

#### **Chart**
- Là một package Helm chứa tất cả các định nghĩa tài nguyên cần thiết để chạy ứng dụng trong Kubernetes cluster
- Tương tự như Homebrew formula, APT dpkg, hoặc Yum RPM

#### **Repository**
- Nơi lưu trữ và chia sẻ các charts
- Tương tự như CPAN archive hoặc Fedora Package Database

#### **Release**
- Một instance của chart đang chạy trong cluster
- Mỗi lần cài đặt tạo ra một release mới với tên riêng
- Một chart có thể được cài đặt nhiều lần trong cùng một cluster

---

## 2. Cấu Trúc Helm Chart

**Reference:** https://helm.sh/docs/topics/charts

### Cấu Trúc Thư Mục Chuẩn

```
mychart/
├── Chart.yaml              # Metadata của chart
├── values.yaml             # Giá trị mặc định
├── values.schema.json      # (Tùy chọn) JSON Schema cho validation
├── charts/                 # Chứa các chart phụ thuộc
├── crds/                   # Custom Resource Definitions
├── templates/              # Chứa các template files
│   ├── NOTES.txt          # Hướng dẫn sau khi cài đặt
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── _helpers.tpl       # Named templates
│   └── tests/             # Test files
├── LICENSE                 # (Tùy chọn) License
└── README.md              # (Tùy chọn) Tài liệu
```

### File Chart.yaml - Các Trường Bắt Buộc

**Reference:** https://helm.sh/docs/topics/charts/#the-chartyaml-file

```yaml
apiVersion: v2              # API version (v1 hoặc v2)
name: mychart              # Tên chart
version: 1.0.0             # Phiên bản chart (SemVer format)

# Các trường tùy chọn
appVersion: "1.16.0"       # Phiên bản ứng dụng
description: A Helm chart for Kubernetes
type: application          # application hoặc library
keywords:
  - nginx
  - web
maintainers:
  - name: Developer Name
    email: dev@example.com
dependencies:              # Danh sách dependencies
  - name: apache
    version: 1.2.3
    repository: https://example.com/charts
```

### Tạo Chart Mới

```bash
helm create mychart
```

---

## 3. Làm Việc Với Templates

**Reference:** https://helm.sh/docs/chart_template_guide/getting_started

### Template Cơ Bản

Templates là các file YAML với cú pháp template được đặt trong dấu `{{ }}`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-configmap
data:
  myvalue: "Hello World"
  drink: {{ .Values.favorite.drink }}
```

### Template Directives

- `{{ }}` - Chèn giá trị
- `{{- }}` - Xóa khoảng trắng bên trái
- `{{ -}}` - Xóa khoảng trắng bên phải

**Ví dụ quản lý khoảng trắng:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{- .Release.Name }}-configmap
data:
  {{- if .Values.enabled }}
  myvalue: "enabled"
  {{- end }}
```

### Test Template Trước Khi Deploy

```bash
# Render template và hiển thị kết quả mà không cài đặt
helm install --debug --dry-run myrelease ./mychart

# Chỉ render template
helm template myrelease ./mychart
```

---

## 4. Values và Quản Lý Cấu Hình

**Reference:** https://helm.sh/docs/chart_template_guide/values_files

### Nguồn Values (Theo Thứ Tự Ưu Tiên)

1. Flag `--set` (ưu tiên cao nhất)
2. File values do user cung cấp qua `-f`
3. File `values.yaml` của parent chart
4. File `values.yaml` mặc định trong chart

### File values.yaml

```yaml
# values.yaml
replicaCount: 3

image:
  repository: nginx
  tag: "1.16.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

resources:
  limits:
    cpu: 100m
    memory: 128Mi
  requests:
    cpu: 100m
    memory: 128Mi

favorite:
  drink: coffee
  food: pizza
```

### Truy Cập Values Trong Template

```yaml
# Truy cập giá trị đơn giản
{{ .Values.replicaCount }}

# Truy cập giá trị lồng nhau
{{ .Values.image.repository }}
{{ .Values.favorite.drink }}

# Sử dụng với functions
{{ .Values.image.repository | quote }}
```

### Override Values Khi Install

```bash
# Sử dụng file values tùy chỉnh
helm install myrelease ./mychart -f custom-values.yaml

# Sử dụng --set
helm install myrelease ./mychart --set replicaCount=5

# Sử dụng nhiều files
helm install myrelease ./mychart -f values1.yaml -f values2.yaml

# Set giá trị lồng nhau
helm install myrelease ./mychart --set image.tag=1.17.0

# Set null để xóa key
helm install myrelease ./mychart --set service.type=null
```

### Best Practices

- Giữ cấu trúc values đơn giản, không lồng quá sâu
- Đặt tên rõ ràng, dễ hiểu
- Cung cấp giá trị mặc định hợp lý
- Thêm comments giải thích trong values.yaml

---

## 5. Template Functions và Pipelines

**Reference:** https://helm.sh/docs/chart_template_guide/functions_and_pipelines

Helm cung cấp hơn 60 functions từ Go template và thư viện Sprig.

### Cú Pháp Functions

```yaml
# Cú pháp thông thường
{{ quote .Values.favorite.drink }}

# Sử dụng pipeline (khuyến nghị)
{{ .Values.favorite.drink | quote }}

# Chaining nhiều functions
{{ .Values.favorite.drink | upper | quote }}
```

### Functions Quan Trọng

#### **default** - Giá Trị Mặc Định
```yaml
# Nếu .Values.favorite.drink không tồn tại, dùng "tea"
drink: {{ .Values.favorite.drink | default "tea" | quote }}
```

#### **quote** - Thêm Dấu Ngoặc Kép
```yaml
name: {{ .Values.name | quote }}
# Kết quả: name: "myapp"
```

#### **upper / lower** - Chuyển Đổi Chữ Hoa/Thường
```yaml
{{ .Values.name | upper }}  # MYAPP
{{ .Values.name | lower }}  # myapp
```

#### **trim** - Xóa Khoảng Trắng
```yaml
{{ .Values.name | trim }}
```

#### **indent / nindent** - Thụt Lề
```yaml
data:
{{ .Values.config | indent 2 }}

# nindent thêm newline trước khi indent
labels:
  {{- include "mychart.labels" . | nindent 4 }}
```

#### **toYaml / toJson** - Chuyển Đổi Format
```yaml
resources:
{{ .Values.resources | toYaml | indent 2 }}
```

#### **lookup** - Truy Vấn Kubernetes Cluster
```yaml
# Lấy thông tin ConfigMap đang tồn tại
{{ $cm := lookup "v1" "ConfigMap" "default" "myconfig" }}
{{ $cm.data.mykey }}

# Syntax: lookup apiVersion kind namespace name
```

**Lưu ý:** `lookup` chỉ hoạt động với `--dry-run=server`

#### **b64enc / b64dec** - Base64 Encoding
```yaml
data:
  password: {{ .Values.password | b64enc }}
```

### Operators

```yaml
# So sánh
{{ if eq .Values.env "production" }}
{{ if ne .Values.replicas 1 }}
{{ if lt .Values.replicas 5 }}  # less than
{{ if gt .Values.replicas 1 }}  # greater than

# Logic
{{ if and .Values.enabled .Values.production }}
{{ if or .Values.dev .Values.staging }}
{{ if not .Values.disabled }}
```

---

## 6. Control Structures

**Reference:** https://helm.sh/docs/chart_template_guide/control_structures

### If/Else Conditions

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-configmap
data:
  {{- if eq .Values.favorite.drink "coffee" }}
  mug: "true"
  {{- else if eq .Values.favorite.drink "tea" }}
  mug: "false"
  {{- else }}
  mug: "unknown"
  {{- end }}
```

**Giá trị được coi là false:**
- Boolean false
- Số 0
- Chuỗi rỗng ""
- nil (null)
- Collections rỗng (map, slice, tuple, dict, array)

### With - Thay Đổi Scope

```yaml
{{- with .Values.favorite }}
drink: {{ .drink }}
food: {{ .food }}
{{- end }}

# Truy cập root scope bên trong with block
{{- with .Values.service }}
type: {{ .type }}
releaseName: {{ $.Release.Name }}  # Sử dụng $
{{- end }}
```

### Range - Vòng Lặp

**Loop qua list:**
```yaml
# values.yaml
pizzaToppings:
  - mushrooms
  - cheese
  - peppers

# template
toppings: |-
  {{- range .Values.pizzaToppings }}
  - {{ . | quote }}
  {{- end }}

# Kết quả:
# toppings: |-
#   - "mushrooms"
#   - "cheese"
#   - "peppers"
```

**Loop qua map:**
```yaml
# values.yaml
favoriteFood:
  italian: pizza
  japanese: sushi
  vietnamese: pho

# template
{{- range $key, $val := .Values.favoriteFood }}
{{ $key }}: {{ $val }}
{{- end }}

# Kết quả:
# italian: pizza
# japanese: sushi
# vietnamese: pho
```

**Loop với index:**
```yaml
{{- range $index, $item := .Values.items }}
item{{ $index }}: {{ $item }}
{{- end }}
```

---

## 7. Named Templates

**Reference:** https://helm.sh/docs/chart_template_guide/named_templates

Named templates (còn gọi là partials hoặc subtemplates) cho phép tái sử dụng code template.

### Định Nghĩa Named Template

File `_helpers.tpl`:
```yaml
{{/*
Common labels
*/}}
{{- define "mychart.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mychart.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create chart name and version
*/}}
{{- define "mychart.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}
```

### Sử Dụng Named Templates

#### Với `template`
```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    {{- template "mychart.labels" . }}
```

#### Với `include` (Khuyến Nghị)
```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
spec:
  selector:
    {{- include "mychart.selectorLabels" . | nindent 4 }}
```

**Lưu ý:**
- `include` tốt hơn `template` vì có thể sử dụng với pipelines
- Luôn truyền scope (`.`) khi gọi template
- Template names là global, nên dùng prefix chart name

### Quy Ước Đặt Tên

```yaml
# Tốt: có prefix chart name
{{- define "mychart.labels" -}}
{{- define "mychart.fullname" -}}

# Tốt: có version cho templates khác nhau
{{- define "mychart.v1.labels" -}}
{{- define "mychart.v2.labels" -}}

# Tránh: không có prefix
{{- define "labels" -}}  # Có thể conflict
```

---

## 8. Built-in Objects

**Reference:** https://helm.sh/docs/chart_template_guide/builtin_objects

Helm cung cấp các objects có sẵn để sử dụng trong templates:

### **Release Object**
```yaml
{{ .Release.Name }}        # Tên release
{{ .Release.Namespace }}   # Namespace deploy
{{ .Release.IsUpgrade }}   # true nếu là upgrade/rollback
{{ .Release.IsInstall }}   # true nếu là install
{{ .Release.Revision }}    # Số revision (bắt đầu từ 1)
{{ .Release.Service }}     # Luôn là "Helm"
```

### **Values Object**
```yaml
{{ .Values.replicaCount }}
{{ .Values.image.repository }}
```

### **Chart Object**
```yaml
{{ .Chart.Name }}          # Tên chart
{{ .Chart.Version }}       # Phiên bản chart
{{ .Chart.AppVersion }}    # Phiên bản app
{{ .Chart.Description }}   # Mô tả
```

### **Files Object**
```yaml
# Đọc file
{{ .Files.Get "config/app.conf" }}

# Đọc file dạng bytes
{{ .Files.GetBytes "image.png" }}

# Glob pattern
{{ range .Files.Glob "configs/*.yaml" }}
{{ .Path }}: {{ .Files.Get .Path }}
{{ end }}

# Đọc từng dòng
{{ range .Files.Lines "config/data.txt" }}
{{ . }}
{{ end }}

# Base64 encode (cho Secrets)
{{ .Files.Get "config.txt" | b64enc }}

# Hoặc dùng AsSecrets
data:
{{ .Files.AsSecrets }}

# Hoặc AsConfig cho ConfigMap
data:
{{ .Files.AsConfig }}
```

### **Capabilities Object**
```yaml
# Kubernetes version
{{ .Capabilities.KubeVersion }}
{{ .Capabilities.KubeVersion.Major }}  # 1
{{ .Capabilities.KubeVersion.Minor }}  # 28

# API versions có sẵn
{{ if .Capabilities.APIVersions.Has "apps/v1" }}
# sử dụng apps/v1
{{ end }}

# Helm version
{{ .Capabilities.HelmVersion }}
```

### **Template Object**
```yaml
{{ .Template.Name }}       # Tên file template hiện tại
{{ .Template.BasePath }}   # Đường dẫn templates directory
```

### **Subcharts**
```yaml
# Truy cập values của subchart
{{ .Values.mysubchart.enabled }}
```

---

## 9. Helm Hooks

**Reference:** https://helm.sh/docs/topics/charts_hooks

Hooks cho phép can thiệp vào lifecycle của release tại các thời điểm cụ thể.

### Các Loại Hooks

| Hook | Thời Điểm Chạy |
|------|----------------|
| `pre-install` | Sau khi render template, trước khi tạo resources |
| `post-install` | Sau khi tất cả resources được tạo |
| `pre-delete` | Trước khi xóa resources |
| `post-delete` | Sau khi tất cả resources bị xóa |
| `pre-upgrade` | Sau khi render, trước khi upgrade resources |
| `post-upgrade` | Sau khi upgrade xong |
| `pre-rollback` | Sau khi render, trước khi rollback |
| `post-rollback` | Sau khi rollback xong |
| `test` | Khi chạy lệnh `helm test` |

### Cách Sử Dụng Hooks

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ .Release.Name }}-pre-install-job
  annotations:
    # Định nghĩa hook type
    "helm.sh/hook": pre-install

    # Thứ tự thực thi (số nhỏ chạy trước)
    "helm.sh/hook-weight": "-5"

    # Chính sách xóa
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  template:
    spec:
      containers:
      - name: pre-install-job
        image: busybox
        command: ['sh', '-c', 'echo Pre-install hook running']
      restartPolicy: Never
```

### Hook Weights

- Giá trị có thể âm hoặc dương (string)
- Hooks với weight thấp hơn chạy trước
- Mặc định là "0"

```yaml
annotations:
  "helm.sh/hook-weight": "-5"   # Chạy đầu tiên
  "helm.sh/hook-weight": "0"    # Chạy sau
  "helm.sh/hook-weight": "5"    # Chạy cuối
```

### Hook Deletion Policies

| Policy | Mô Tả |
|--------|-------|
| `before-hook-creation` | Xóa resource cũ trước khi tạo hook mới (mặc định) |
| `hook-succeeded` | Xóa sau khi hook thành công |
| `hook-failed` | Xóa sau khi hook thất bại |

**Ví dụ:**
```yaml
annotations:
  "helm.sh/hook-delete-policy": hook-succeeded,hook-failed
```

### Ví Dụ Thực Tế: Database Migration

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ .Release.Name }}-db-migration
  annotations:
    "helm.sh/hook": pre-upgrade,pre-install
    "helm.sh/hook-weight": "-1"
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  template:
    metadata:
      name: {{ .Release.Name }}-db-migration
    spec:
      restartPolicy: Never
      containers:
      - name: db-migration
        image: {{ .Values.migration.image }}
        command:
        - /bin/sh
        - -c
        - |
          echo "Running database migration..."
          migrate -path /migrations -database $DATABASE_URL up
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {{ .Release.Name }}-db-secret
              key: url
```

---

## 10. Testing Charts

**Reference:** https://helm.sh/docs/topics/chart_tests

### Cấu Trúc Test

Test trong Helm là một Pod/Job với annotation `helm.sh/hook: test`:

```yaml
# templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ .Release.Name }}-test-connection"
  annotations:
    "helm.sh/hook": test
spec:
  containers:
  - name: wget
    image: busybox
    command: ['wget']
    args: ['{{ .Release.Name }}:80']
  restartPolicy: Never
```

### Test Phức Tạp Hơn

```yaml
# templates/tests/test-database.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ .Release.Name }}-test-db"
  annotations:
    "helm.sh/hook": test
    "helm.sh/hook-weight": "1"
    "helm.sh/hook-delete-policy": hook-succeeded
spec:
  containers:
  - name: test-db
    image: postgres:14
    command:
    - /bin/sh
    - -c
    - |
      psql $DATABASE_URL -c "SELECT 1" || exit 1
      echo "Database connection successful"
    env:
    - name: DATABASE_URL
      valueFrom:
        secretKeyRef:
          name: {{ .Release.Name }}-db-secret
          key: url
  restartPolicy: Never
```

### Chạy Tests

```bash
# Cài đặt chart
helm install myrelease ./mychart

# Đợi pods ready
kubectl get pods

# Chạy tests
helm test myrelease

# Xem logs của test
helm test myrelease --logs

# Cleanup sau test
kubectl delete pod myrelease-test-connection
```

### Best Practices

1. **Tổ chức tests trong thư mục riêng:**
   ```
   templates/
   └── tests/
       ├── test-connection.yaml
       ├── test-database.yaml
       └── test-api.yaml
   ```

2. **Sử dụng hook annotations:**
   ```yaml
   annotations:
     "helm.sh/hook": test
     "helm.sh/hook-weight": "1"
     "helm.sh/hook-delete-policy": hook-succeeded
   ```

3. **Test các kịch bản quan trọng:**
   - Kết nối service
   - Xác thực credentials
   - Kiểm tra cấu hình từ values.yaml
   - Validation API endpoints

4. **Thêm tests/ vào .helmignore** nếu không muốn package tests

---

## 11. Các Lệnh Helm Quan Trọng

**Reference:** https://helm.sh/docs/helm

### Quản Lý Repository

```bash
# Thêm repository
helm repo add bitnami https://charts.bitnami.com/bitnami

# Liệt kê repositories
helm repo list

# Update repositories
helm repo update

# Xóa repository
helm repo remove bitnami

# Tìm kiếm charts
helm search repo nginx
helm search hub wordpress
```

### Quản Lý Charts

```bash
# Tạo chart mới
helm create mychart

# Kiểm tra syntax
helm lint mychart

# Package chart
helm package mychart

# Xem thông tin chart
helm show chart bitnami/nginx
helm show values bitnami/nginx
helm show all bitnami/nginx

# Pull chart về local
helm pull bitnami/nginx
helm pull bitnami/nginx --untar
```

### Quản Lý Releases

```bash
# Install chart
helm install myrelease ./mychart
helm install myrelease ./mychart -f custom-values.yaml
helm install myrelease ./mychart --set replicaCount=3
helm install myrelease ./mychart --namespace mynamespace --create-namespace

# Install với dry-run
helm install myrelease ./mychart --dry-run --debug

# Liệt kê releases
helm list
helm list --all-namespaces
helm list -n mynamespace

# Xem status
helm status myrelease

# Xem values đã sử dụng
helm get values myrelease
helm get values myrelease --all

# Xem manifest đã deploy
helm get manifest myrelease

# Xem tất cả thông tin
helm get all myrelease

# Xem history
helm history myrelease
```

### Upgrade và Rollback

```bash
# Upgrade release
helm upgrade myrelease ./mychart
helm upgrade myrelease ./mychart -f new-values.yaml
helm upgrade myrelease ./mychart --set image.tag=2.0.0

# Upgrade hoặc install nếu chưa tồn tại
helm upgrade --install myrelease ./mychart

# Upgrade với dry-run
helm upgrade myrelease ./mychart --dry-run --debug

# Rollback về revision trước
helm rollback myrelease

# Rollback về revision cụ thể
helm rollback myrelease 2

# Xem history để biết revision
helm history myrelease
```

### Uninstall

```bash
# Xóa release
helm uninstall myrelease

# Xóa và giữ history
helm uninstall myrelease --keep-history

# Xóa trong namespace cụ thể
helm uninstall myrelease -n mynamespace
```

### Template và Debug

```bash
# Render templates
helm template myrelease ./mychart

# Render với values
helm template myrelease ./mychart -f values.yaml

# Show chỉ một template cụ thể
helm template myrelease ./mychart -s templates/deployment.yaml

# Debug mode
helm install myrelease ./mychart --dry-run --debug

# Validate templates
helm lint ./mychart
```

### Dependencies

```bash
# Update dependencies theo Chart.yaml
helm dependency update ./mychart

# List dependencies
helm dependency list ./mychart

# Build dependencies (download vào charts/)
helm dependency build ./mychart
```

---

## 12. Best Practices

**Reference:** https://helm.sh/docs/chart_best_practices

### General Conventions

#### Đặt Tên Chart
- Sử dụng chữ thường và dấu gạch ngang: `my-app`
- Tên chart phải khớp với tên directory
- Không dùng ký tự đặc biệt

#### Version
- Tuân thủ SemVer 2: `MAJOR.MINOR.PATCH`
- Version trong Chart.yaml là version của chart
- appVersion là version của ứng dụng được deploy

### Values Best Practices

**Reference:** https://helm.sh/docs/chart_best_practices/values

#### Cấu Trúc Values

```yaml
# Tốt: Cấu trúc phẳng, rõ ràng
image:
  repository: nginx
  tag: "1.16.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

# Tránh: Lồng quá sâu
app:
  config:
    server:
      http:
        port:
          value: 80  # Quá sâu
```

#### Tên Variables

```yaml
# Tốt: camelCase
replicaCount: 3
serviceAccount:
  create: true

# Tránh: snake_case, PascalCase
replica_count: 3
ServiceAccount:
  Create: true
```

#### Giá Trị Mặc Định

- Cung cấp giá trị mặc định hợp lý
- Document tất cả values quan trọng
- Sử dụng comments để giải thích

```yaml
# Number of replicas for the deployment
# Recommended: 3 for production, 1 for development
replicaCount: 1

# Container image configuration
image:
  # Image repository
  repository: nginx
  # Image tag (use appVersion if not specified)
  tag: ""
  # Image pull policy
  pullPolicy: IfNotPresent
```

### Templates Best Practices

**Reference:** https://helm.sh/docs/chart_best_practices/templates

#### Quản Lý Khoảng Trắng

```yaml
# Tốt: Dùng {{- và -}} để kiểm soát whitespace
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-config
  {{- with .Values.labels }}
  labels:
    {{- toYaml . | nindent 4 }}
  {{- end }}
```

#### Sử Dụng include Thay Vì template

```yaml
# Tốt: Dùng include để có thể indent
metadata:
  labels:
    {{- include "mychart.labels" . | nindent 4 }}

# Tránh: template không thể indent
metadata:
  labels:
    {{- template "mychart.labels" . }}
```

#### Validation

```yaml
# Validate required values
{{- if not .Values.image.repository }}
{{- fail "image.repository is required" }}
{{- end }}

# Validate giá trị hợp lệ
{{- if not (has .Values.service.type (list "ClusterIP" "NodePort" "LoadBalancer")) }}
{{- fail "service.type must be ClusterIP, NodePort, or LoadBalancer" }}
{{- end }}
```

### Labels và Annotations

**Reference:** https://helm.sh/docs/chart_best_practices/labels

#### Standard Labels (Kubernetes Recommended)

```yaml
{{- define "mychart.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ include "mychart.chart" . }}
{{- end }}

{{- define "mychart.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
```

#### Sử Dụng Labels

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-deployment
  labels:
    {{- include "mychart.labels" . | nindent 4 }}
spec:
  selector:
    matchLabels:
      {{- include "mychart.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "mychart.selectorLabels" . | nindent 8 }}
```

### Resources Best Practices

#### Luôn Định Nghĩa Resources

```yaml
# values.yaml
resources:
  limits:
    cpu: 100m
    memory: 128Mi
  requests:
    cpu: 100m
    memory: 128Mi

# template
resources:
  {{- toYaml .Values.resources | nindent 10 }}
```

### Security Best Practices

**Reference:** https://helm.sh/docs/chart_best_practices/pods

#### Pod Security

```yaml
# Sử dụng non-root user
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

# Read-only filesystem
containers:
- name: app
  securityContext:
    readOnlyRootFilesystem: true
  volumeMounts:
  - name: tmp
    mountPath: /tmp
```

#### Secrets Management

```yaml
# Không hardcode secrets trong values.yaml
# Sử dụng external secret management hoặc --set

# Tốt: Reference existing secret
env:
- name: DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.database.existingSecret }}
      key: password

# Hoặc cho phép tạo secret từ values (với cảnh báo)
{{- if .Values.database.password }}
---
apiVersion: v1
kind: Secret
metadata:
  name: {{ .Release.Name }}-db-secret
type: Opaque
data:
  password: {{ .Values.database.password | b64enc }}
{{- end }}
```

### Dependencies Best Practices

**Reference:** https://helm.sh/docs/chart_best_practices/dependencies

```yaml
# Chart.yaml
dependencies:
- name: postgresql
  version: "~12.1.0"  # Sử dụng version constraints
  repository: https://charts.bitnami.com/bitnami
  condition: postgresql.enabled  # Cho phép disable
  tags:
    - database

- name: redis
  version: "^17.0.0"
  repository: https://charts.bitnami.com/bitnami
  condition: redis.enabled
```

```yaml
# values.yaml - Override subchart values
postgresql:
  enabled: true
  auth:
    username: myapp
    database: myappdb

redis:
  enabled: false
```

### Documentation

1. **README.md** - Hướng dẫn sử dụng chart
2. **values.yaml** - Comments chi tiết cho mỗi option
3. **NOTES.txt** - Hướng dẫn sau khi install

```yaml
# templates/NOTES.txt
Thank you for installing {{ .Chart.Name }}.

Your release is named {{ .Release.Name }}.

To learn more about the release, try:

  $ helm status {{ .Release.Name }}
  $ helm get all {{ .Release.Name }}

{{ if .Values.ingress.enabled }}
Application URL:
{{- range .Values.ingress.hosts }}
  http{{ if $.Values.ingress.tls }}s{{ end }}://{{ . }}
{{- end }}
{{ else }}
Get the application URL by running:
  export POD_NAME=$(kubectl get pods -l "app.kubernetes.io/name={{ .Chart.Name }}" -o jsonpath="{.items[0].metadata.name}")
  kubectl port-forward $POD_NAME 8080:80
  echo "Visit http://127.0.0.1:8080"
{{ end }}
```

---

## Tài Liệu Tham Khảo Chính

- **Trang chủ Helm:** https://helm.sh
- **Documentation:** https://helm.sh/docs
- **Chart Guide:** https://helm.sh/docs/topics/charts
- **Template Guide:** https://helm.sh/docs/chart_template_guide
- **Best Practices:** https://helm.sh/docs/chart_best_practices
- **Helm Commands:** https://helm.sh/docs/helm

---

## Kết Luận

Tài liệu này tổng hợp các kiến thức cơ bản và nâng cao về Helm, giúp bạn:

✅ Hiểu cấu trúc và cách hoạt động của Helm Charts
✅ Viết templates với syntax chính xác
✅ Quản lý values và configurations hiệu quả
✅ Sử dụng functions, pipelines và control structures
✅ Implement hooks và testing
✅ Tuân thủ best practices

**Lời khuyên:**
- Bắt đầu với chart đơn giản và dần mở rộng
- Luôn test với `--dry-run` trước khi deploy
- Sử dụng `helm lint` để validate
- Tham khảo các charts có sẵn từ Bitnami, Helm stable repos để học hỏi

Chúc bạn thành công! 🚀
