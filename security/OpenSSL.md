
#security #openssl #cryptography

# 1. What is OpenSSL?

OpenSSL is a cryptography library and tool tin work with cryptographic jobs like:
- encryption/decryption
- Hash
- sign/verify messages

It support a wide range of cryptography algorithms like RSA, Elliptic Curves, SHA256, AES, ...

# 2. Using OpenSSL as CLI tool

## 1. Overview

`openssl` command has subcommands to do specific job. E.g. `openssl rsa` can do works related to RSA algorithm like: sign/verify, encryption/decryption, ...

## 2. Generate private key

`openssl genrsa -out private-key.pem 2048`
`openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out private-key.pem`

This command generate a new RSA private key with 2048 key. The first command is traditional command which generate private key with RSA algo