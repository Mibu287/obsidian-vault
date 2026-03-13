
#network #protocol #hacking 


# 1. What is Server Message Block (SMB)?

- Communication protocol. i.e: defined how 2 entities talks to each other. In this cases, 2 computers/processes can talk to each other using set of rules defined by SMB.
- Can be used to share files/printers across networks. E.g. on Windows, users can map network drives and access them like a local drives. Under the hood, the local computer connect to the remote computer - which has the network drives using SMB protocol.
- Can be used for inter-process communication. E.g. DCE/RPC (Distributed computing environment / Remote procedure call) protocol mainly use SMB as its transport protocol.
- Support Kerberos in authentication

# 2. SMB named pipe

SMB protocol support named pipe in which 2 processes (either in the same or different machines) can communicate using read/write file API. In which, each process open the named pipe URL to get a file handle. Then, the 2 processes can read/write from the handle.
Under the hood, 