#offsec #mssql 

# 1. Authentication

- SQL Server manage logins which are credentials to authenticate with the SQL Server instance. The logins can be either: 
    - Windows-based logins: which use username / passwords for the Windows / Active Directory accounts to login to SQL Server (e.g john.doe@company.example.com)
    - SQL logins: the login is created in SQL Server instance only (e.g: `sa`, `jdoe`)
- In corporate environment, it's often restricted access to SQL Server for AD accounts only. SQL login is often disabled on the SQL server instance. By using AD accounts, the instance can rely on AD for authentication and authorization.

# 2. Tools

## 2.1. 