#offsec #mssql
## 1. Authentication

SQL Server uses **logins** to authenticate connections to a SQL Server instance.

| Login type           | How it works                                                                                                                  | Examples                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **Windows login**    | Uses a Windows account, such as an Active Directory domain account. SQL Server relies on Windows to authenticate the account. | `john.doe@company.example.com` |
| **SQL Server login** | SQL Server stores and validates the credentials at the instance level.                                                        | `sa`, `jdoe`                   |

In corporate environments, SQL Server may be configured for **Windows Authentication only**, with access granted to specific Active Directory users or groups. SQL Server then determines what those users can do based on their roles and permissions.

## 2. Tools

### 2.1. `impacket mssqlclient.py`

Use `mssqlclient.py` to connect to a SQL Server instance and run T-SQL queries interactively.

```shell
impacket-mssqlclient $USER:'$PASSWD'@$SERVER_ADDR -port $SERVER_PORT
```

> [!tip] Built-in enumeration commands
> Once connected, you can run these commands at the `mssqlclient.py` prompt.
>
> | Command | Lists |
> | --- | --- |
> | `enum_db` | Databases |
> | `enum_logins` | Server-level logins |
> | `enum_users` | Users in the current database |
> | `enum_impersonate` | Impersonation permissions |
Use Impacket's `mssqlclient.py` to connect to a SQL Server instance and run T-SQL queries interactively.

### 2.2. `NetExec mssql`

NetExec's `mssql` module can connect to SQL Server and run queries.

#### Enumerate account names by RID

```shell
netexec mssql $SERVER_ADDR -u $USER -p $PASSWD --local-auth --rid-brute
```

RID enumeration varies the **relative identifier (RID)**, the final component of a Windows security identifier (SID). It then uses a T-SQL expression such as `SELECT SUSER_SNAME(SID_BINARY(...))` to resolve each candidate SID to an account name.

### Execute query


```shell
netexec mssql $SERVER_ADDR -u $USER -p $PASSWD --local-auth --rid-brute \
    -q 'SELECT * FROM sys.sql_logins;'
```