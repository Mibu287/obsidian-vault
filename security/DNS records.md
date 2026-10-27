#oscp #offsec #dns

CNAME is a field in zone file which instruct DNS server to get A record from other domain name.

i.e
```plaintext
CNAME: awesome.example.com -> others.github.io
```

Khi đến trang `awesome.example.com`, DNS Server tìm đến bản ghi `awesome.example.com` nhận được `CNAME`, DNS Server tìm tiếp `others.github.io để lấy được bản ghi A.

Giả sử `others.github.io` đã bị xoá, tin tặc có thể gửi yêu cầu github cấp cho domain cũ. Khi người dùng truy cập sẽ nhận được mã độc.


MX là bản ghi mail ==> có thể tìm CVE của server email để khai thác
NS là bản ghi name server ==> Authoritative DNS server
TXT là bản ghi dạng free text, người dùng có thể đưa thông tin tuỳ ý vào trường này.
Tin tặc có thể dùng bản ghi này để truyền thông tin điều khiển virus trong hệ thống mạng nội bộ (kể cả khi mạng đã chặn requests ra ngoài)
