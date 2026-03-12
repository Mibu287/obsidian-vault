
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

This command generate a new RSA private key with 2048 key. The first command is traditional command which generate private key with RSA algorithm only. The second command is algorithm-agnostic.

Public key can be derived from private key, using this command:
`openssl rsa -in test.priv -pubout -out test.pub`

Analysis of the command:
- `rsa` is sub-command
- `-in test.priv`: Using `test.priv` file as input. This file is private key file.
- `-pubout`: What to generate from the input. In this case, generate associated public key.
- `-out test.pub`: Path to newly generated public key file.

The private key can be used to sign/verify, encrypt/decrypt.

# 3. Request for signing

Given private/public key pair, only the public key and some metadata (organization name, domain name, ...) are sent to some trusted authority to be cryptographically signed. Those trusted authority is called Certificate Authority (CA). After the certificate is signed, a chain of trust is established, any devices which trust the CA can verify that the signature is valid and can trust