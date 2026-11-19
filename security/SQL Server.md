#offsec #mssql 

# 1. Authentication

- SQL Server manage logins which are credentials to authenticate with the SQL Server instance. The logins can be either: 
    - Windows-based logins: which use username / passwords for the Windows / Active Directory accounts to login to SQL Server (e.g john.doe@company.example.com)
    - SQL logins: the login is created in SQL Server instance only (e.g: sa, )