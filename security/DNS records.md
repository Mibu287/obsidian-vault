#oscp #offsec #dns

CNAME is a field in zone file which instruct DNS server to get A record from other domain name.

i.e
```plaintext
CNAME: awesome.example.com -> others.github.io
```

Khi đến trang `awesome.example.com`, DNS Server tìm đến bản ghi `awesome.example.com` nhận được `CNAME`, DNS Server tìm tiếp `others.github.io để lấy được bản ghi A.

Giả sử `others.github.io` đã bị xoá, tin tặc có thể gửi yêu cầu github cấp cho domain cũ. Khi người dùng truy cập sẽ nhận được mã độc.



