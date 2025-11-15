# Kubectl Cheat Sheet - Tài Liệu Tham Khảo Nhanh

## Kubectl Autocomplete

### BASH
```bash
source <(kubectl completion bash)
echo "source <(kubectl completion bash)" >> ~/.bashrc

# Nếu bạn sử dụng alias cho kubectl
alias k=kubectl
complete -F __start_kubectl k
```

### ZSH
```bash
source <(kubectl completion zsh)
echo "if [ $commands[kubectl] ]; then source <(kubectl completion zsh); fi" >> ~/.zshrc
```

## Kubectl Context và Configuration

```bash
# Hiển thị các cài đặt kubeconfig đã được merged
kubectl config view

# Hiển thị tất cả các context
kubectl config get-contexts

# Hiển thị context hiện tại
kubectl config current-context

# Đặt context mặc định
kubectl config use-context <tên-context>

# Tạo hoặc sửa đổi context
kubectl config set-context <tên-context>

# Thêm thông tin xác thực người dùng
kubectl config set-credentials <tên-user>

# Xóa người dùng
kubectl config unset users.<tên-user>
```

## Tạo các Object

```bash
# Tạo resource từ file
kubectl apply -f ./manifest.yaml

# Tạo từ nhiều file
kubectl apply -f ./dir

# Tạo từ URL
kubectl apply -f https://git.io/vPieo

# Tạo deployment
kubectl create deployment nginx --image=nginx

# Xem tài liệu về manifest
kubectl explain pods,svc

# Tạo nhiều object từ stdin
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: busybox-sleep
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - sleep
    - "1000000"
---
apiVersion: v1
kind: Pod
metadata:
  name: busybox-sleep-less
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - sleep
    - "1000"
EOF

# Tạo Secret với nhiều key
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  password: $(echo -n "s33msi4" | base64 -w0)
  username: $(echo -n "jane" | base64 -w0)
EOF
```

## Xem và Tìm Kiếm Tài Nguyên

### Các lệnh Get cơ bản
```bash
# Liệt kê tất cả services trong namespace
kubectl get services

# Liệt kê tất cả pods ở tất cả namespaces
kubectl get pods --all-namespaces

# Liệt kê tất cả pods trong namespace hiện tại với thông tin chi tiết
kubectl get pods -o wide

# Liệt kê một deployment cụ thể
kubectl get deployment my-dep

# Liệt kê tất cả pods trong namespace hiện tại
kubectl get pods

# Lấy thông tin pod ở định dạng YAML
kubectl get pod my-pod -o yaml
```

### Các lệnh Describe chi tiết
```bash
# Mô tả thông tin chi tiết của một node
kubectl describe nodes my-node

# Mô tả thông tin chi tiết của một pod
kubectl describe pods my-pod
```

### Sắp xếp theo các trường
```bash
# Liệt kê Services được sắp xếp theo tên
kubectl get services --sort-by=.metadata.name

# Liệt kê pods được sắp xếp theo số lần restart
kubectl get pods --sort-by='.status.containerStatuses[0].restartCount'

# Liệt kê PersistentVolumes được sắp xếp theo dung lượng
kubectl get pv --sort-by=.spec.capacity.storage
```

### Sử dụng Selectors
```bash
# Lấy tất cả worker nodes (sử dụng selector để loại trừ các node có label 'node-role.kubernetes.io/control-plane')
kubectl get node --selector='!node-role.kubernetes.io/control-plane'

# Lấy tất cả pods đang chạy trong namespace hiện tại
kubectl get pods --field-selector=status.phase=Running

# Lấy ExternalIPs của tất cả nodes
kubectl get nodes -o jsonpath='{.items[*].status.addresses[?(@.type=="ExternalIP")].address}'

# Liệt kê tên các pods thuộc về một RC cụ thể
sel=${$(kubectl get rc my-rc --output=json | jq -j '.spec.selector | to_entries | .[] | "\(.key)=\(.value),"')%?}
echo $(kubectl get pods --selector=$sel --output=jsonpath={.items..metadata.name})

# Hiển thị labels của tất cả pods
kubectl get pods --show-labels

# Kiểm tra những node nào đã sẵn sàng
JSONPATH='{range .items[*]}{@.metadata.name}:{range @.status.conditions[*]}{@.type}={@.status};{end}{end}' \
 && kubectl get nodes -o jsonpath="$JSONPATH" | grep "Ready=True"

# Liệt kê tất cả Secrets hiện đang được sử dụng bởi một pod
kubectl get pods -o json | jq '.items[].spec.containers[].env[]?.valueFrom.secretKeyRef.name' | grep -v null | sort | uniq

# Liệt kê tất cả containerID của initContainer trong tất cả pods
kubectl get pods --all-namespaces -o jsonpath='{range .items[*].status.initContainerStatuses[*]}{.containerID}{"\n"}{end}' | cut -d/ -f3

# Liệt kê Events được sắp xếp theo timestamp
kubectl get events --sort-by=.metadata.creationTimestamp

# Liệt kê tất cả cảnh báo
kubectl get events --field-selector type=Warning

# So sánh trạng thái hiện tại và mong muốn của cluster
kubectl diff -f ./my-manifest.yaml

# Tạo chuỗi phân tách bởi dấu chấm của tất cả các key được trả về bởi jsonpath
kubectl get nodes -o json | jq -c 'paths|join(".")'

# Tạo chuỗi phân tách bởi dấu chấm của tất cả các key với giá trị là một mảng
kubectl get nodes -o json | jq -c 'paths(type!="object")'

# Tạo danh sách tất cả container images đang chạy trong một cluster
kubectl get pods -A -o=custom-columns='DATA:spec.containers[*].image'
```

## Cập Nhật Tài Nguyên

```bash
# Rolling update "www" containers của deployment "frontend", cập nhật image
kubectl set image deployment/frontend www=image:v2

# Kiểm tra lịch sử deployment bao gồm các revisions
kubectl rollout history deployment/frontend

# Rollback về deployment trước đó
kubectl rollout undo deployment/frontend

# Rollback về một revision cụ thể
kubectl rollout undo deployment/frontend --to-revision=2

# Theo dõi trạng thái rolling update của deployment "frontend" cho đến khi hoàn thành
kubectl rollout status -w deployment/frontend

# Rolling restart của deployment "frontend"
kubectl rollout restart deployment/frontend
```

**Lưu ý:** Lệnh `kubectl rolling-update` đã deprecated kể từ phiên bản 1.11, sử dụng `kubectl rollout` thay thế.

## Patch Tài Nguyên

```bash
# Cập nhật một phần một node
kubectl patch node k8s-node-1 -p '{"spec":{"unschedulable":true}}'

# Cập nhật image của container; spec.containers[*].name là bắt buộc vì đó là merge key
kubectl patch pod valid-pod -p '{"spec":{"containers":[{"name":"kubernetes-serve-hostname","image":"new image"}]}}'

# Cập nhật image của container sử dụng json patch với mảng vị trí
kubectl patch pod valid-pod --type='json' -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"new image"}]'

# Disable livenessProbe của deployment sử dụng json patch với mảng vị trí
kubectl patch deployment valid-deployment  --type json   -p='[{"op": "remove", "path": "/spec/template/spec/containers/0/livenessProbe"}]'

# Thêm một phần tử mới vào mảng vị trí
kubectl patch sa default --type='json' -p='[{"op": "add", "path": "/secrets/1", "value": {"name": "whatever" } }]'
```

## Chỉnh Sửa Tài Nguyên

```bash
# Chỉnh sửa service có tên 'docker-registry'
kubectl edit svc/docker-registry

# Sử dụng editor thay thế
KUBE_EDITOR="nano" kubectl edit svc/docker-registry
```

## Scale Tài Nguyên

```bash
# Scale một replicaset có tên 'foo' thành 3
kubectl scale --replicas=3 rs/foo

# Scale một resource được chỉ định trong "foo.yaml" thành 3
kubectl scale --replicas=3 -f foo.yaml

# Nếu kích thước hiện tại của deployment là 2, scale thành 3
kubectl scale --current-replicas=2 --replicas=3 deployment/mysql

# Scale nhiều replication controllers
kubectl scale --replicas=5 rc/foo rc/bar rc/baz

# Auto scale deployment "nginx"
kubectl autoscale deployment nginx --min=10 --max=15
```

## Xóa Tài Nguyên

```bash
# Xóa một pod sử dụng loại và tên được chỉ định trong pod.json
kubectl delete -f ./pod.json

# Xóa tất cả các pods và services có label name=myLabel
kubectl delete pods,services -l name=myLabel

# Xóa tất cả pods và services trong namespace my-ns
kubectl -n my-ns delete po,svc --all

# Xóa tất cả pods phù hợp với pattern1 hoặc pattern2
kubectl get pods  -n mynamespace --no-headers=true | awk '/pattern1|pattern2/{print $1}' | xargs  kubectl delete -n mynamespace pod

# Xóa pod với grace period 0 (xóa ngay lập tức)
kubectl delete pod my-pod --grace-period=0 --force

# Xóa namespace test
kubectl delete namespaces test
```

## Tương Tác với Pods Đang Chạy

```bash
# Lấy output từ pod đang chạy
kubectl logs my-pod

# Lấy output từ một container cụ thể trong pod (hữu ích cho multi-container pods)
kubectl logs my-pod -c my-container

# Theo dõi logs theo thời gian thực
kubectl logs -f my-pod

# Lấy logs từ container trước đó của pod (hữu ích khi pod bị crash)
kubectl logs my-pod --previous

# Chạy pod tương tác (hữu ích cho debugging)
kubectl run -i --tty busybox --image=busybox --restart=Never -- sh

# Chạy pod trong namespace cụ thể
kubectl run nginx --image=nginx -n mynamespace

# Chạy pod và ghi manifest vào file
kubectl run nginx --image=nginx --dry-run=client -o yaml > pod.yaml

# Attach vào một container đang chạy
kubectl attach my-pod -i

# Lắng nghe trên cổng local 5000 và forward tới port 6000 trên pod
kubectl port-forward my-pod 5000:6000

# Chạy lệnh trong container đang tồn tại - chỉ 1 container trong pod
kubectl exec my-pod -- ls /

# Chạy lệnh trong container cụ thể của pod - multi container
kubectl exec my-pod -c my-container -- ls /

# Mở shell tương tác trong pod (1 container)
kubectl exec -it my-pod -- /bin/bash

# Mở shell tương tác trong container cụ thể của pod
kubectl exec -it my-pod -c my-container -- /bin/bash

# Hiển thị metrics cho một node cụ thể
kubectl top node my-node

# Hiển thị metrics cho tất cả nodes
kubectl top node

# Hiển thị metrics cho một pod cụ thể
kubectl top pod my-pod

# Hiển thị metrics cho tất cả pods trong namespace mặc định
kubectl top pod

# Hiển thị metrics cho tất cả pods với labels
kubectl top pod --all-namespaces
```

## Copy Files và Directories

```bash
# Copy /tmp/foo từ local vào /tmp/bar trong pod trong namespace mặc định
kubectl cp /tmp/foo my-pod:/tmp/bar

# Copy /tmp/foo từ local vào /tmp/bar trong container cụ thể
kubectl cp /tmp/foo my-pod:/tmp/bar -c my-container

# Copy /tmp/foo từ pod vào /tmp/bar trên local
kubectl cp my-pod:/tmp/foo /tmp/bar

# Copy /tmp/foo từ container cụ thể về local
kubectl cp my-pod:/tmp/foo -c my-container /tmp/bar
```

## Tương Tác với Deployments và Services

```bash
# Dump trạng thái hiện tại của cluster ra stdout
kubectl cluster-info dump

# Dump trạng thái hiện tại của cluster vào /path/to/cluster-state
kubectl cluster-info dump --output-directory=/path/to/cluster-state

# Nếu một taint với key đó và effect NoSchedule đã tồn tại, giá trị của nó được thay thế
kubectl taint nodes foo dedicated=special-user:NoSchedule

# Đánh dấu node "foo" là unschedulable
kubectl cordon foo

# Drain node "foo", sẵn sàng cho maintenance
kubectl drain foo

# Đánh dấu node "foo" là schedulable
kubectl uncordon foo
```

## Tương Tác với Nodes và Cluster

```bash
# Đánh dấu node là unschedulable
kubectl cordon my-node

# Drain node để chuẩn bị cho maintenance
kubectl drain my-node

# Đánh dấu node là schedulable
kubectl uncordon my-node

# Hiển thị metrics cho node
kubectl top node my-node

# Hiển thị địa chỉ của master và services
kubectl cluster-info

# Dump trạng thái cluster hiện tại ra stdout
kubectl cluster-info dump

# Dump trạng thái cluster hiện tại vào /path/to/cluster-state
kubectl cluster-info dump --output-directory=/path/to/cluster-state

# Xem các loại resource có sẵn
kubectl api-resources

# Xem các API versions có sẵn
kubectl api-versions

# Xem cấu hình kubeconfig
kubectl config view

# Lấy danh sách các contexts
kubectl config get-contexts

# Hiển thị context hiện tại
kubectl config current-context

# Chuyển sang context khác
kubectl config use-context my-cluster-name

# Thêm user mới vào kubeconfig
kubectl config set-credentials kubeuser/foo.kubernetes.com --username=kubeuser --password=kubepassword

# Đặt namespace mặc định cho context hiện tại
kubectl config set-context --current --namespace=ggckad-s2

# Đặt context sử dụng username và namespace cụ thể
kubectl config set-context gce --user=cluster-admin --namespace=foo \
  && kubectl config use-context gce

# Xóa user khỏi kubeconfig
kubectl config unset users.foo
```

## Các Loại Resource

```bash
# Liệt kê tất cả các loại API resources được hỗ trợ
kubectl api-resources

# Liệt kê các API resources với thông tin chi tiết hơn
kubectl api-resources -o wide

# Liệt kê tất cả các API resources với namespace
kubectl api-resources --namespaced=true

# Liệt kê tất cả các API resources không có namespace
kubectl api-resources --namespaced=false

# Liệt kê các API versions
kubectl api-versions
```

## Format Output

```bash
# Output trong plain-text format
kubectl get pods

# Output trong plain-text format với thông tin bổ sung
kubectl get pods -o wide

# Output dưới dạng YAML
kubectl get pod my-pod -o yaml

# Output dưới dạng JSON
kubectl get pod my-pod -o json

# Output với custom-columns
kubectl get pods -o custom-columns=NAME:.metadata.name,RSRC:.metadata.resourceVersion

# Chỉ lấy tên của pod
kubectl get pods -o name

# Output với template
kubectl get pods -o template --template='{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}'
```

## Kubectl Verbosity và Debugging

```bash
# Kubectl verbosity được điều khiển với -v hoặc --v flags theo sau là một số nguyên
# đại diện cho log level. Các quy ước logging chung và các log level liên quan được mô tả ở đây.

# Verbosity 0: Luôn hiển thị cho operator
kubectl get pods -v=0

# Verbosity 1: Mức log mặc định hợp lý
kubectl get pods -v=1

# Verbosity 2: Thông tin trạng thái hữu ích về service
kubectl get pods -v=2

# Verbosity 3: Thông tin mở rộng về các thay đổi
kubectl get pods -v=3

# Verbosity 4: Verbosity dạng debug
kubectl get pods -v=4

# Verbosity 5: Trace level verbosity
kubectl get pods -v=5

# Verbosity 6: Hiển thị các HTTP requests
kubectl get pods -v=6

# Verbosity 7: Hiển thị HTTP request headers
kubectl get pods -v=7

# Verbosity 8: Hiển thị HTTP request contents
kubectl get pods -v=8

# Verbosity 9: Hiển thị HTTP request contents mà không cắt bớt
kubectl get pods -v=9
```

## Các Lệnh Khác

```bash
# Tạo namespace
kubectl create namespace my-namespace

# Tạo ConfigMap từ file
kubectl create configmap my-config --from-file=path/to/file

# Tạo ConfigMap từ literal values
kubectl create configmap my-config --from-literal=key1=value1 --from-literal=key2=value2

# Tạo Secret từ file
kubectl create secret generic my-secret --from-file=path/to/file

# Tạo Secret từ literal values
kubectl create secret generic my-secret --from-literal=username=admin --from-literal=password=secret

# Tạo ServiceAccount
kubectl create serviceaccount my-service-account

# Expose deployment dưới dạng service
kubectl expose deployment nginx --port=80 --type=LoadBalancer

# Tạo job
kubectl create job my-job --image=busybox -- echo "Hello World"

# Tạo cronjob
kubectl create cronjob my-cronjob --image=busybox --schedule="*/1 * * * *" -- echo "Hello World"

# Annotate resource
kubectl annotate pods my-pod description='my description'

# Label resource
kubectl label pods my-pod env=production

# Xem API resources và các thuộc tính của chúng
kubectl explain pods
kubectl explain pods.spec
kubectl explain pods.spec.containers
```

---

**Nguồn:** [Kubernetes Official Documentation](https://kubernetes.io/vi/docs/reference/kubectl/cheatsheet/)

**Ghi chú:**
- Sử dụng `kubectl apply` là cách được khuyến nghị để quản lý các ứng dụng trên Kubernetes trong môi trường production
- Tham khảo thêm tại [Kubectl Book](https://kubectl.docs.kubernetes.io) để có hướng dẫn chi tiết hơn
