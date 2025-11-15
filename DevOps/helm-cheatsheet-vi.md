# Helm Cheat Sheet - Tài Liệu Tra Cứu Nhanh

## Khái Niệm Cơ Bản

**Chart**: Tên chart, tham chiếu repository (`<tên_repo>/<tên_chart>`), hoặc URL/đường dẫn đến chart.

**Name**: Định danh được gán cho một bản cài đặt Helm chart.

**Release**: Tên phiên bản cài đặt.

**Revision**: Giá trị từ lệnh Helm history.

**Repo-name**: Định danh repository.

**DIR**: Tên thư mục hoặc đường dẫn.

---

## Quản Lý Chart

| Lệnh | Mục Đích |
|------|----------|
| `helm create <tên>` | Tạo một thư mục chart cùng với các file và thư mục phổ biến được sử dụng trong chart |
| `helm package <đường-dẫn-chart>` | Đóng gói chart thành file archive có phiên bản |
| `helm lint <chart>` | Chạy kiểm tra để xem xét chart và xác định các vấn đề có thể có |
| `helm show all <chart>` | Kiểm tra chart và liệt kê nội dung của nó |
| `helm show values <chart>` | Hiển thị nội dung của file values.yaml |
| `helm pull <chart>` | Tải xuống hoặc kéo một chart |
| `helm pull <chart> --untar=true` | Tải xuống và giải nén chart sau khi tải |
| `helm pull <chart> --verify` | Xác minh gói trước khi sử dụng |
| `helm pull <chart> --version <số>` | Chỉ định ràng buộc phiên bản chart |
| `helm dependency list <chart>` | Hiển thị danh sách các phụ thuộc của chart |

---

## Cài Đặt và Gỡ Bỏ Ứng Dụng

| Lệnh | Mục Đích |
|------|----------|
| `helm install <tên> <chart>` | Cài đặt chart với một tên |
| `helm install <tên> <chart> --namespace <namespace>` | Cài đặt trong một namespace cụ thể |
| `helm install <tên> <chart> --set key1=val1,key2=val2` | Thiết lập giá trị trên dòng lệnh |
| `helm install <tên> <chart> --values <file-yaml/url>` | Cài đặt với file giá trị được chỉ định |
| `helm install <tên> <chart> --dry-run --debug` | Chạy thử cài đặt để xác thực chart |
| `helm install <tên> <chart> --verify` | Xác minh gói trước khi sử dụng |
| `helm install <tên> <chart> --dependency-update` | Cập nhật phụ thuộc trước khi cài đặt |
| `helm uninstall <tên>` | Gỡ bỏ một release khỏi namespace hiện tại |
| `helm uninstall <tên-release> --namespace <namespace>` | Gỡ bỏ một release khỏi namespace cụ thể |

---

## Nâng Cấp và Rollback Ứng Dụng

| Lệnh | Mục Đích |
|------|----------|
| `helm upgrade <release> <chart>` | Nâng cấp một release |
| `helm upgrade <release> <chart> --rollback-on-failure` | Rollback khi nâng cấp thất bại |
| `helm upgrade <release> <chart> --dependency-update` | Cập nhật phụ thuộc trước khi nâng cấp |
| `helm upgrade <release> <chart> --version <số_phiên_bản>` | Chỉ định ràng buộc phiên bản chart |
| `helm upgrade <release> <chart> --values` | Chỉ định giá trị trong file YAML hoặc URL |
| `helm upgrade <release> <chart> --set key1=val1,key2=val2` | Thiết lập giá trị trên dòng lệnh |
| `helm upgrade <release> <chart> --force` | Buộc cập nhật tài nguyên thông qua chiến lược thay thế |
| `helm rollback <release> <revision>` | Rollback một release về một revision cụ thể |
| `helm rollback <release> <revision> --cleanup-on-fail` | Cho phép xóa tài nguyên mới khi rollback thất bại |

---

## Liệt Kê, Thêm, Xóa và Cập Nhật Repository

| Lệnh | Mục Đích |
|------|----------|
| `helm repo add <tên-repo> <url>` | Thêm một repository từ internet |
| `helm repo list` | Liệt kê các chart repository đã thêm |
| `helm repo update` | Cập nhật thông tin chart có sẵn cục bộ |
| `helm repo remove <tên_repo>` | Xóa một hoặc nhiều chart repository |
| `helm repo index <DIR>` | Tạo file index dựa trên các chart tìm thấy |
| `helm repo index <DIR> --merge` | Gộp index được tạo với file index hiện có |
| `helm search repo <từ-khóa>` | Tìm kiếm repository cho từ khóa trong charts |
| `helm search hub <từ-khóa>` | Tìm kiếm Artifact Hub cho charts |

---

## Giám Sát Helm Release

| Lệnh | Mục Đích |
|------|----------|
| `helm list` | Liệt kê tất cả releases cho một namespace |
| `helm list --all` | Hiển thị tất cả releases không có bộ lọc |
| `helm list --all-namespaces` | Liệt kê releases trên tất cả namespaces |
| `helm list -l key1=value1,key2=value2` | Lọc releases theo nhãn |
| `helm list --date` | Sắp xếp releases theo ngày |
| `helm list --deployed` | Hiển thị releases đã triển khai |
| `helm list --pending` | Hiển thị releases đang chờ |
| `helm list --failed` | Hiển thị releases thất bại |
| `helm list --uninstalled` | Hiển thị releases đã gỡ bỏ (nếu lịch sử được giữ lại) |
| `helm list --superseded` | Hiển thị releases đã bị thay thế |
| `helm list -o yaml` | Xuất ra định dạng YAML (cũng hỗ trợ table, json) |
| `helm status <release>` | Hiển thị trạng thái của một release được đặt tên |
| `helm status <release> --revision <số>` | Hiển thị trạng thái với một revision cụ thể |
| `helm history <release>` | Hiển thị các revision lịch sử cho một release |
| `helm env` | In tất cả thông tin môi trường mà Helm sử dụng |

---

## Tải Thông Tin Release

| Lệnh | Mục Đích |
|------|----------|
| `helm get all <release>` | Tập hợp thông tin dễ đọc về notes, hooks, giá trị được cung cấp |
| `helm get hooks <release>` | Tải hooks cho một release (định dạng YAML) |
| `helm get manifest <release>` | Lấy các tài nguyên Kubernetes được tạo từ một release |
| `helm get notes <release>` | Hiển thị notes được cung cấp bởi chart |
| `helm get values <release>` | Tải file values cho một release |

---

## Quản Lý Plugin

| Lệnh | Mục Đích |
|------|----------|
| `helm plugin install <đường-dẫn/url>` | Cài đặt plugins |
| `helm plugin list` | Xem tất cả plugins đã cài đặt |
| `helm plugin update <plugin>` | Cập nhật plugins |
| `helm plugin uninstall <plugin>` | Gỡ bỏ một plugin |

---

## Lưu Ý Quan Trọng

- Sử dụng `--dry-run --debug` để kiểm tra trước khi thực sự cài đặt
- Luôn backup trước khi thực hiện rollback
- Sử dụng `--namespace` để quản lý releases trong các namespace khác nhau
- Có thể kết hợp nhiều flags trong một lệnh
- Sử dụng `helm get values <release>` để xem cấu hình hiện tại trước khi nâng cấp
