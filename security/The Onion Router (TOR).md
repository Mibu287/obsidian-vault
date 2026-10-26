
#tor #security #privacy

# 1. Connect to website out of onion network

- The browser pick 3 nodes from the onion network to create a circuit (guard, middle and exit nodes)
- The client do key exchange with guard node to get K<sub>g</sub>. Then, using guard node to connect to middle node to do key exchange to get K<sub>m</sub>. Finally, using the connection client -> guard -> middle to connect to exit node to do key exchange to get K<sub>e</sub>. Because Diffie Hellman Key Exchange can share a secret key between 2 parties over unsecured network ==> K<sub>g</sub>, K<sub>m</sub> and K<sub>e</sub> are secured (i.e only known by the client and the corresponding node)
- The client setup TLS handshake with the website over the circuit setup above. TLS protocol also perform DHKE between the client and the web server ==> the session key is only known by the client and the web server (None of the relay node has the session key)
- The client prepare the HTTP request, encrypt it with TLS session key, then encrypt with K<sub>g</sub>, then K<sub>m</sub>, then K<sub>e</sub>. Each relay node use the corresponding key to decrypt the request and forward to the next.
- The web server receive HTTPS request from the exit node, decrypt to get plain text request and reply back to the exit node.
- The exit node receive the HTTPS response. It can not understand the response because it does not have the TLS session key. In only encrypt the HTTPS response using K<sub>e</sub> and forward to middle and then guard node. Finally, the guard forward the response to the client. Now the requests is encrypted in layer by K<sub>HTTPS</sub>, K<sub>e</sub>, K<sub>m</sub>, K<sub>g</sub>. The client has all the keys to decrypt and read the response.

So, to summarize, TOR protocol use TCP as transport layer and in turn, it act as transport layer for higher-level protocol like HTTP. In this protocol, only the client know about the whole circuit, other parties only know about the node above and/or below them.

# 2. Connect to hidden services

## 2.1. How hidden services come to be?

- A hidden service is just a web service connected by the TOR network. Note: the web service listen and response to request just like a normal web service. The only difference is how the service is discovered and connected.
- Firstly, the web admin pick a private key and derive corresponding public key and the onion address (the onion address is basically the public key with checksum and version). Note: the web admin can not pick the onion address like buying domain name for surface web. If they want the address to has certain form (e.g prefix with their business name), they must try multiple keys until find a match. The computation effort grow exponentially with the number of characters in the address required to match.
- The web admin use the public key and the current date to derive blinded key for each day.
- The web admin choose a number of introduction nodes from the TOR network. They create random AUTH_KEY for each introduction node, sign the AUTH_KEY with the blinded key above. Then, connect to introduction node using TOR circuit and ask the introduction node to hold a record of (AUTH_KEY, signature and some meta data) to response to request. The introduction node know neither the service's IP address nor the onion address. It only know about the middle node to relay introduction request from client when presented with AUTH_KEY.
- The web admin choose some nodes to be directory service (HSDirs). Each HSDir is asked to hold a record of (blinded key, encrypted data). Encrypted data is encrypted by the public key (not the blinded key) ==> the HSDir can not decrypt to get information from this.
- The client must know the onion address before hand from other channels (the TOR network itself contains no information about hidden services inside it). It compute public key and blinded key from the address.
- The client use the blinded key to query HSDirs and get back encrypted data which then be decrypted using the public key. Now, the client learnt about introduction nodes and corresponding AUTH_KEY.
- The client pick a node to be rendezvous, it send the node a random data as cookie. The node wait for other connection with the same cookie arrive and bridge the 2 connections.
- The client connect to introduction node, present AUTH_KEY and ask the node to send to the service some encrypted data (using the public key ==> only the service who have the private key can decrypt). Inside the encrypted data contains rendezvous node, cookie, the client's ephemeral key to do key exchange.
- The service connect to rendezvous node with the cookie. The node bridge the 2 connections and let the client & server do key exchange. Now connection is established between the 2 and both of them has the shared end-to-end key.
- The request/response flow same as the case connect to website above with 1 note that in this case the client and server is connected by 2 TOR circuits bridged at rendezvous node.