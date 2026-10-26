#tor #security #privacy

# 1. Connect to a website outside of the onion network

- The browser picks 3 nodes from the onion network to create a circuit (guard, middle and exit nodes).
- The client does a key exchange with the guard node to get K<sub>g</sub>. Then, it uses the guard node to connect to the middle node and does a key exchange to get K<sub>m</sub>. Finally, it uses the connection client -> guard -> middle to connect to the exit node and does a key exchange to get K<sub>e</sub>. Because Diffie-Hellman Key Exchange can share a secret key between 2 parties over an unsecured network ==> K<sub>g</sub>, K<sub>m</sub> and K<sub>e</sub> are secure (i.e. only known by the client and the corresponding node).
- The client sets up a TLS handshake with the website over the circuit set up above. The TLS protocol also performs DHKE between the client and the web server ==> the session key is only known by the client and the web server (none of the relay nodes has the session key).
- The client prepares the HTTP request, encrypts it with the TLS session key, then encrypts it with K<sub>g</sub>, then K<sub>m</sub>, then K<sub>e</sub>. Each relay node uses the corresponding key to decrypt the request and forwards it to the next.
- The web server receives the HTTPS request from the exit node, decrypts it to get the plain text request and replies back to the exit node.
- The exit node receives the HTTPS response. It cannot understand the response because it does not have the TLS session key. It only encrypts the HTTPS response using K<sub>e</sub> and forwards it to the middle and then the guard node. Finally, the guard forwards the response to the client. Now the response is encrypted in layers by K<sub>HTTPS</sub>, K<sub>e</sub>, K<sub>m</sub>, K<sub>g</sub>. The client has all the keys to decrypt and read the response.

So, to summarize, the TOR protocol uses TCP as the transport layer and, in turn, it acts as the transport layer for higher-level protocols like HTTP. In this protocol, only the client knows about the whole circuit; other parties only know about the nodes above and/or below them.

# 2. Connect to hidden services

## 2.1. How do hidden services come to be?

- A hidden service is just a web service connected to the TOR network. Note: the web service listens and responds to requests just like a normal web service. The only difference is how the service is discovered and connected to.
- Firstly, the web admin picks a private key and derives the corresponding public key and the onion address (the onion address is basically the public key with a checksum and version). Note: the web admin cannot pick the onion address like buying a domain name on the surface web. If they want the address to have a certain form (e.g. prefixed with their business name), they must try multiple keys until they find a match. The computation effort grows exponentially with the number of characters in the address required to match.
- The web admin uses the public key and the current date to derive a blinded key for each day.
- The web admin chooses a number of introduction nodes from the TOR network. They create a random AUTH_KEY for each introduction node and sign the AUTH_KEY with the blinded key above. Then, they connect to each introduction node using a TOR circuit and ask the introduction node to hold a record of (AUTH_KEY, signature and some metadata) to respond to requests. The introduction node knows neither the service's IP address nor the onion address. It only knows about the middle node to relay introduction requests from clients when presented with the AUTH_KEY.
- The web admin chooses some nodes to be directory services (HSDirs). Each HSDir is asked to hold a record of (blinded key, encrypted data). The encrypted data is encrypted by the public key (not the blinded key) ==> the HSDir cannot decrypt it to get information from it.
- The client must know the onion address beforehand from other channels (the TOR network itself contains no information about the hidden services inside it). It computes the public key and blinded key from the address.
- The client uses the blinded key to query HSDirs and gets back the encrypted data, which is then decrypted using the public key. Now, the client has learnt about the introduction nodes and the corresponding AUTH_KEYs.
- The client picks a node to be the rendezvous node and sends the node random data as a cookie. The node waits for another connection with the same cookie to arrive and bridges the 2 connections.
- The client connects to an introduction node, presents the AUTH_KEY and asks the node to send some encrypted data to the service (using the public key ==> only the service, which has the private key, can decrypt it). The encrypted data contains the rendezvous node, the cookie and the client's ephemeral key to do the key exchange.
- The service connects to the rendezvous node with the cookie. The node bridges the 2 connections and lets the client & server do the key exchange. Now a connection is established between the 2 and both of them have the shared end-to-end key.
- The request/response flow is the same as in the case of connecting to a website above, with 1 note: in this case, the client and server are connected by 2 TOR circuits bridged at the rendezvous node.