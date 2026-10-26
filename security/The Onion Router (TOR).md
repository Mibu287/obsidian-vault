
#tor #security #privacy

# 1. Connect to website out of onion network

- The browser pick 3 nodes from the onion network to create a circuit (guard, middle and exit nodes)
- The client do key exchange with guard node to get K<sub>g</sub>. Then, using guard node to connect to middle node to do key exchange to get Km. Finally, using the connection client -> guard -> middle to connect to exit node to do key exchange to get Ke. Because Diffie Hellman Key Exchange can share a secret key between 2 parties over unsecured network ==> Kg, Km and Ke are secured (i.e only known by the client and the corresponding node)
- The client setup 
