
#tor #security #privacy

# 1. Connect to website out of onion network

- The browser pick 3 nodes from the onion network to create a circuit (guard, middle and exit nodes)
- The client do key exchange with guard node to get K<sub>g</sub>. Then, using guard node to connect to middle node to do key exchange to get K<sub>m</sub>. Finally, using the connection client -> guard -> middle to connect to exit node to do key exchange to get K<sub>e</sub>. Because Diffie Hellman Key Exchange can share a secret key between 2 parties over unsecured network ==> K<sub>g</sub>, K<sub>m</sub> and K<sub>e</sub> are secured (i.e only known by the client and the corresponding node)
- The client setup TLS handshake with the website over the circuit setup above. TLS protocol also perform DHKE between the client and the web server ==> the session key is only known by the client and the web server (None of the relay node has the session key)
- The client prepare the HTTP request, encrypt it with TLS session key, then encrypt with K<sub>g</sub>, then K<sub>m</sub>, then K<sub>e</sub>. Each relay node use the corresponding key to decrypt the request and forward to the next.
- The web server receive HTTPS request from the exit node, decrypt to get plain text request and reply back to the exit node.
- The exit node receive the HTTPS response. It can not understand the response because it does not have the TLS session key. In only encrypt the HTTPS response using K<sub>e</sub> and forward to middle and then guard node. Finally, the guard forward the response to the client. Now the requests is encrypted in layer by K<sub>HTTPS</sub>, K<sub>e</sub>, K<sub>m</sub>, K<sub>g</sub>. The client has all the keys to decrypt and read the response.

So, to summarize, TOR 
