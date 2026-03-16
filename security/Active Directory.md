
#security #activedirectory #windows


# 1. What is Active Directory (AD) ?

- Directory service i.e. software system which manage users, devices and other resources on a network. In AD term, a network is call a domain. A domain contains users, devices (computers, printers, ...), organization units (OUs) (like departments, branches, ...)
- Administrator can set policies for OUs, users, devices. E.g. users in accounting department can access accounting software but not customer relation management system (CRM).

# 2. Terminology

- **Object**: Any resources managed by active directory. E.g. users, computers, printers, ...
- **Schema**: Blueprint for AD object. In object-oriented terminology, schema is class which defined which attributes objects can have. E.g:
```
Schema defines the CLASS:          AD stores the INSTANCES:
─────────────────────────          ───────────────────────
User (class)                       John Smith (object)
  ├── firstName (attribute)          ├── firstName = "John"
  ├── lastName (attribute)           ├── lastName = "Smith"
  ├── email (attribute)              ├── email = "john@corp.com"
  └── password (attribute)           └── password = [hash]

Computer (class)                   DESKTOP-001 (object)
  ├── name (attribute)               ├── name = "DESKTOP-001"
  ├── OS (attribute)                 ├── OS = "Windows 11"
  └── SID (attribute)                └── SID = "S-1-5-21-..."
```

- **Attribute**: An attribute is a data field in schema. An object can has multiple attributes which define its properties. E.g: an user may has attributes like: name, email, password, ...
- **Domain**: Logical collection of AD object. E.g. in a large corporation, each department/branch can be a separated domain: branch1.corp, branch2.corp, ... Domains can operate independently or can be connected with each other via trust relationship.
- **Tree**: A collection of AD domain which begins with the same root domain. A domain can be added to be a child-domain of an existing domain.
- **Forrest**: At the top level, AD server manage a forest of domains i.e collection of all trees.
- **Global Unique Identifier**: when a object is added to a domain, it is assigned a GUID which stored in its `objectGUID` attribute. The `objectGUID` attribute never change as long as the object exist in the domain.
- **Security principal**: is the term is used in the context of authentication. When a user, computer attempt to authenticate themselves to the AD server, they are considered security principals.
- **Security Identifier (SID)**: is an unique ID assigned to each security principal which can be used for authentication.
- **Global catalog**: is a domain controller which contains all objects in an AD forest.
- **LDAP**: i.e. Lightweight Directory Access Protocol is an open-standard protocol to talk to directory service server. LDAP can be used for authenticate, add/modify/delete/read entry.
- **NTDS.DIT**: The NTDS.DIT file can be considered the heart of Active Directory. It is stored on a Domain Controller at `C:\Windows\NTDS\` and is a database that stores AD data such as information about user and group objects, group membership, and, most important to attackers and penetration testers, the password hashes for all users in the domain.

# 3. Authentication protocol

## 3.1. NTLM

NTLM (NT Lan Manager) is an protocol for authenticating users. It's often used as the fallback protocol in case Kerberos not available.

How it works?

- A client with attempt to login with username / password pair.
- The server send some challenge bytes to the client
- Client compute response: Function(NTLM_HASH(password), server_challenge, client_challenge)
- Server recompute the response and compare with client's. If matched, the client is authenticated.

**NOTE**:
A man-in-the-middle does not need clear text password. Only the NTLM_HASH of the password is sufficient to authenticate. If the server database is breached and hashed password is leaked, any attackers can use these hashes to login. This type of attack is called pass-the-hash.