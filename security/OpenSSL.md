
#security #openssl #cryptography

# 1. What is OpenSSL?

OpenSSL is a cryptography library and tool tin work with cryptographic jobs like:
- encryption/decryption
- Hash
- sign/verify messages

It support a wide range of cryptography algorithms like RSA, Elliptic Curves, SHA256, AES, ...

# 2. Using OpenSSL as CLI tool

## 2.1. Overview

`openssl` command has subcommands to do specific job. E.g. `openssl rsa` can do works related to RSA algorithm like: sign/verify, encryption/decryption, ...

## 2.2. Generate private key

`openssl genrsa -out private-key.pem 2048`
`openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out private-key.pem`

This command generate a new RSA private key with 2048 key. The first command is traditional command which generate private key with RSA algorithm only. The second command is algorithm-agnostic.

Public key can be derived from private key, using this command:
`openssl rsa -in test.priv -pubout -out test.pub`

Analysis of the command:
- `rsa` is sub-command
- `-in test.priv`: Using `test.priv` file as input. This file is private key file.
- `-pubout`: What to generate from the input. In this case, generate associated public key.
- `-out test.pub`: Path to newly generated public key file.

The private key can be used to sign/verify, encrypt/decrypt.

# 2.3. Request for signing

Given private/public key pair, only the public key and some metadata (organization name, domain name, ...) are sent to some trusted authority to be cryptographically signed. Those trusted authority is called Certificate Authority (CA). After the certificate is signed, a chain of trust is established, any devices which trust the CA can verify that the signature is valid and can trust the new device.

`openssl req -new -key test.pem -out req_test.csr -subj "/C=US/ST=California/O=MyCompany/CN=myserver.example.com"`

Analysis of command:

- `req` is sub-command which create a request-for-signing file.
- `-new` option to create new request
- `-key test.pem` specify input private key file
- `-out req_test.csr` path to new request file
- `-subj "/C=US/ST=California/O=MyCompany/CN=myserver.example.com"`: metadata for request file

View request file `openssl req -in req_test.csr -text -noout`

```plaintext
Certificate Request:
    Data:
        Version: 1 (0x0)
        Subject: C=US, ST=California, O=MyCompany, CN=myserver.example.com
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
                    00:b5:a0:52:e9:a0:9b:10:f7:a8:04:0a:29:41:66:
                    11:a3:c0:12:2e:58:e1:ac:6d:2e:8d:15:06:0a:92:
                    9d:6d:28:08:bb:db:d4:2e:a2:1d:0a:29:72:c7:89:
                    35:56:58:51:71:a3:9d:94:d9:82:f3:b1:9d:0b:85:
                    8c:15:b7:48:36:a8:94:f2:6c:43:12:5e:3a:8f:74:
                    16:a2:3f:fc:33:19:52:21:3e:bc:c8:5c:13:f3:e9:
                    b3:e6:b7:bc:99:43:1d:bb:8a:13:29:94:78:f3:87:
                    6c:a5:47:e9:a4:b4:5b:20:8d:e0:1a:14:e0:c7:64:
                    6d:7f:64:3b:24:c3:72:66:a7:17:6b:93:0f:b3:9a:
                    2f:89:53:ef:9f:f6:f1:43:8d:00:e8:ab:b5:a2:12:
                    69:75:a1:2a:d9:aa:9b:7c:14:10:c8:6e:a5:5f:04:
                    db:c4:a7:d7:e4:6b:36:dc:7f:a5:49:b7:35:78:13:
                    f1:1a:70:35:37:4c:99:4c:85:95:14:6c:1b:9c:ef:
                    0b:67:d3:ec:64:02:b1:81:0a:87:10:38:a3:bf:a7:
                    0a:97:56:a7:6c:4f:31:7b:2f:bf:10:98:b5:9d:a1:
                    cb:ea:6c:ac:4f:e2:33:f2:cc:ae:87:48:03:e7:af:
                    a0:a6:5c:8f:78:e2:fe:ed:4a:5d:82:bc:a4:97:1a:
                    dd:25
                Exponent: 65537 (0x10001)
        Attributes:
            (none)
            Requested Extensions:
    Signature Algorithm: sha256WithRSAEncryption
    Signature Value:
        26:c5:01:38:ea:a8:45:d0:cc:be:95:10:7f:c1:b3:c4:4c:9c:
        80:66:aa:40:03:cc:ca:db:44:ef:6a:84:6e:ac:34:59:7f:42:
        73:1c:d1:af:d7:3d:3f:76:7b:66:08:fb:9d:40:f1:30:af:e0:
        af:fe:73:99:76:f3:43:c8:39:d7:c6:3f:a2:0c:99:24:d4:e0:
        a5:b1:f3:f8:75:98:52:9e:1d:6d:01:43:ad:e4:e9:6a:a0:fc:
        bb:67:0e:86:98:ab:e4:3f:b1:61:25:07:a6:4c:3b:3d:8c:9e:
        76:a7:92:a8:e6:05:a9:5f:30:a1:70:7a:f5:77:d1:15:35:3e:
        ff:d6:5d:41:63:e7:24:9e:2c:bf:3d:99:ed:bc:87:c9:12:b0:
        32:3e:9d:ad:e2:22:8c:78:30:50:fe:6b:45:ea:20:93:2d:25:
        0b:87:46:26:e0:1f:6c:43:85:81:d4:19:da:36:f7:d0:f7:03:
        26:ee:aa:35:1d:9f:76:47:9b:da:96:cc:54:b3:d6:fd:03:d4:
        78:36:a3:4c:d6:cc:ab:1c:45:1e:d7:b3:2a:be:51:2f:b5:00:
        00:ec:da:0c:fc:29:31:e2:88:4d:22:b0:57:7b:f2:e7:40:7f:
        f8:57:94:6e:8b:f7:0a:86:16:84:ee:7c:b6:98:db:51:f0:2d:
        dd:e8:fa:3b
```

The request file contains:
- Public key
- Metadata
- Signature (signed by the private key)

**NOTE**: no private key is sent.

# 2.4. Sign the requests

Receiver can sign requests using `openssl`

`openssl x509 -req -in req_test.csr -signkey test.pem -out cert_test.crt -days 365`

Analysis of command:
- `x509` is sub-command which work with X509 certificate. X509 is standard certificate format.
- `-req -in req_test.csr` specify request file to be signed
- `-signkey test.pem` is private key file used to sign request file
- `-days 365` number of days the certificate valid

After signing, the signer generate a new certificate. The new certificate can be viewed with the command `openssl x509 -in ./cert_test.crt -text -noout`

```plaintext
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number:
            79:dd:7d:cf:50:3d:23:71:50:01:cf:54:5a:58:32:24:72:14:c7:1e
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: C=US, ST=California, O=MyCompany, CN=myserver.example.com
        Validity
            Not Before: Mar 12 06:46:49 2026 GMT
            Not After : Mar 12 06:46:49 2027 GMT
        Subject: C=US, ST=California, O=MyCompany, CN=myserver.example.com
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
                    00:b5:a0:52:e9:a0:9b:10:f7:a8:04:0a:29:41:66:
                    11:a3:c0:12:2e:58:e1:ac:6d:2e:8d:15:06:0a:92:
                    9d:6d:28:08:bb:db:d4:2e:a2:1d:0a:29:72:c7:89:
                    35:56:58:51:71:a3:9d:94:d9:82:f3:b1:9d:0b:85:
                    8c:15:b7:48:36:a8:94:f2:6c:43:12:5e:3a:8f:74:
                    16:a2:3f:fc:33:19:52:21:3e:bc:c8:5c:13:f3:e9:
                    b3:e6:b7:bc:99:43:1d:bb:8a:13:29:94:78:f3:87:
                    6c:a5:47:e9:a4:b4:5b:20:8d:e0:1a:14:e0:c7:64:
                    6d:7f:64:3b:24:c3:72:66:a7:17:6b:93:0f:b3:9a:
                    2f:89:53:ef:9f:f6:f1:43:8d:00:e8:ab:b5:a2:12:
                    69:75:a1:2a:d9:aa:9b:7c:14:10:c8:6e:a5:5f:04:
                    db:c4:a7:d7:e4:6b:36:dc:7f:a5:49:b7:35:78:13:
                    f1:1a:70:35:37:4c:99:4c:85:95:14:6c:1b:9c:ef:
                    0b:67:d3:ec:64:02:b1:81:0a:87:10:38:a3:bf:a7:
                    0a:97:56:a7:6c:4f:31:7b:2f:bf:10:98:b5:9d:a1:
                    cb:ea:6c:ac:4f:e2:33:f2:cc:ae:87:48:03:e7:af:
                    a0:a6:5c:8f:78:e2:fe:ed:4a:5d:82:bc:a4:97:1a:
                    dd:25
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Subject Key Identifier:
                A7:2E:ED:46:71:A6:E8:DD:C3:5A:C6:53:3F:BB:13:9B:D9:30:03:57
    Signature Algorithm: sha256WithRSAEncryption
    Signature Value:
        b4:2a:1e:35:13:7f:f3:3c:57:81:e0:e2:3a:8a:55:6b:03:47:
        f7:25:7b:c6:be:08:14:c9:0f:08:2e:f9:30:44:7c:38:c1:9e:
        0c:c6:49:8f:e5:4e:2d:da:52:be:13:dd:d3:09:1e:4b:7d:6a:
        d8:10:38:56:77:1d:cc:b3:df:8d:b1:f5:54:97:8c:29:ba:a3:
        bb:1d:1b:52:d2:4d:1c:cb:f4:47:a6:4d:6e:81:34:9a:61:37:
        f5:55:0f:e9:93:5b:50:54:e6:0d:4f:94:25:9a:75:6c:6b:99:
        56:7b:1a:7b:5f:4c:7a:e3:d3:fa:21:d9:26:08:44:88:f8:7f:
        fa:8d:60:89:49:20:e4:fe:3c:1b:43:97:fd:70:64:59:1e:60:
        21:f0:58:79:0a:e8:02:ca:0a:43:92:4d:fc:2a:64:7f:be:ec:
        33:5e:e4:86:99:7f:ed:39:3a:4a:46:ee:dd:f0:77:e0:c5:2e:
        53:59:97:9d:b3:cf:b3:de:72:3d:4c:d4:a6:8d:da:9c:67:50:
        99:db:4d:f7:56:7b:8d:3b:10:68:88:02:a3:c0:8e:4b:b5:6c:
        99:70:e3:b3:ae:60:66:3f:88:24:65:30:3d:3d:04:d7:02:96:
        8b:ed:65:72:a0:8d:35:ec:c6:19:cb:66:9e:fd:2e:ae:02:16:
        c9:82:17:45
```

The certificate contains:
- Public key
- Metadata
- Signature (signed by the signer - Certificate Authority)

The signed certificate can be used as TLS certificate, Active Directory, ...
