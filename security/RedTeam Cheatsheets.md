---
title: RedTeam Toolbox
aliases:
  - RedTeam Toolbox
  - Red Team Arsenal
  - Offensive Security Toolbox
tags:
  - redteam
  - cybersecurity
  - pentesting
  - toolbox
  - offsec
  - cheatsheets
type: moc
status: active
---

# RedTeam Toolbox

> [!tip] Purpose
> Map of content for authorized security assessments, defensive validation, and lab notes.

> [!important] Quick access
> [[#01. Cheatsheets|Saved cheatsheets]] · [[#11. Defense & Detection|Detection references]] · [[#16. Reporting & Documentation|Finding template]]

## Contents

- **Scope and review** · [[#02. Reconnaissance & OSINT|Reconnaissance & OSINT]] · [[#03. Enumeration|Enumeration]] · [[#04. Web Application Security|Web application security]]
- **Host and network controls** · [[#05. Shells & Payload References|Shell telemetry]] · [[#06. Active Directory Security|Active Directory]] · [[#07. Privilege Escalation|Privilege escalation]] · [[#08. Pivoting & Tunneling|Segmentation]] · [[#09. Credential Security|Credentials]]
- **Detection and infrastructure** · [[#10. Command & Control (C2)|Command & control]] · [[#12. Cloud & Container Security|Cloud & containers]] · [[#13. Utilities & Automation|Utilities]]
- **Practice and methods** · [[#14. Labs & Practice|Labs]] · [[#15. Research & Methodologies|Methodologies]]

## 01. Cheatsheets

| Resource                                                                                                                                                                         | Category         | Type       | Description                         |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------- | :--------- | :---------------------------------- |
| [PayloadsAllTheThings - Reverse Shell Cheatsheet](https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Methodology%20and%20Resources/Reverse%20Shell%20Cheatsheet.md) | Reverse Shell    | Cheatsheet | Reverse shell reference             |
| [InternalAllTheThings](https://swisskyrepo.github.io/InternalAllTheThings/)                                                                                                      | Internal Pentest | Cheatsheet | Internal security testing reference |
| [RevShells](https://www.revshells.com)                                                                                                                                           | Reverse Shell    | Generator  | Interactive reverse shell generator |
| [Awesome cURL Reverse Shell Oneliners](https://github.com/blue0x1/awesome-curl-reverseshell-oneliners)                                                                           | Reverse Shell    | Collection | cURL reverse shell references       |

For remediation guidance on common application security issues, see the [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/).

## 02. Reconnaissance & OSINT

- Review the approved asset inventory against public DNS records for owned domains. Record asset owner, scope status, source, and last verified date.
- Review alias records and ownership changes with the vault’s [[security/DNS#CNAME|DNS notes]].

## 03. Enumeration

- Reconcile approved service inventories with firewall rules and existing telemetry. Record the service, expected exposure, owner, and any discrepancy.
- Review current file-sharing requirements using [[network/Server Message Block|Server Message Block]]. Use the [Wireshark User’s Guide](https://www.wireshark.org/docs/wsug_html_chunked/) to interpret approved packet captures.

## 04. Web Application Security

- Use the [OWASP Application Security Verification Standard](https://owasp.org/projects/asvs) to define acceptance criteria for authentication, authorization, session handling, and APIs.
- Use the [OWASP Web Security Testing Guide](https://owasp.org/projects/web-security-testing-guide) to organize assessment coverage. Record the version and control IDs alongside the results.

## 05. Shells & Payload References

- Keep the saved references under [[#01. Cheatsheets|Cheatsheets]]. Use benign shell activity to confirm collection of process creation and PowerShell events. For logging configuration and event coverage, see the [Microsoft PowerShell logging reference](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_logging_windows) and [Sysmon documentation](https://learn.microsoft.com/en-us/sysinternals/downloads/sysmon).

## 06. Active Directory Security

- Use [[security/Active Directory|Active Directory]] for terminology and authentication background.
- Review privileged groups, service account ownership, and separation of administrative duties with [Microsoft’s Active Directory security guidance](https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/plan/security-best-practices/best-practices-for-securing-active-directory).

## 07. Privilege Escalation

- Review local administrators, privileged service accounts, and approved elevation policies. Record the business justification and owner for each exception.
- Compare host settings with the relevant [CIS Benchmark](https://www.cisecurity.org/cis-benchmarks). For Windows elevation prompts and policy, see [Microsoft’s User Account Control guidance](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/user-account-control/).

## 08. Pivoting & Tunneling

- Validate network segmentation against approved flows. Record each permitted or denied path, its expected policy, and the observed result.
- For AWS VPCs, use [VPC Reachability Analyzer](https://docs.aws.amazon.com/vpc/latest/reachability/what-is-reachability-analyzer.html) to review whether the network configuration allows a path.

## 09. Credential Security

- Review MFA coverage, recovery flows, service account ownership, and the handling of legacy authentication exceptions. Document affected accounts and the responsible owner without storing credentials in the vault.
- Use [NIST SP 800-63B-4](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf) for authenticator and authentication assurance requirements.

## 10. Command & Control (C2)

- Use MITRE’s [Command and Control tactic](https://attack.mitre.org/tactics/TA0011/) to map detection coverage. Note which endpoint, DNS, proxy, and firewall logs support each detection and where coverage is missing.

## 11. Defense & Detection

- Use [Sysmon](https://learn.microsoft.com/en-us/sysinternals/downloads/sysmon) to collect Windows process and network telemetry, and consult [Sigma documentation](https://sigmahq.io/docs/) for portable detection rule syntax.
- Record the expected event, required fields, observed alert, log source, and gaps for each detection check.

## 12. Cloud & Container Security

- Use [AWS Security Hub CSPM](https://docs.aws.amazon.com/securityhub/latest/userguide/what-is-securityhub.html) to review findings and configuration controls across AWS accounts.
- Review RBAC, workload restrictions, network policies, and audit logging with the [Kubernetes Security Checklist](https://kubernetes.io/docs/concepts/security/security-checklist/).

## 13. Utilities & Automation

- Use [[security/OpenSSL|OpenSSL]] for certificate and TLS reference material.
- Record the tool version, input source, and evidence location for inventory reconciliation and log analysis so you can repeat the check.

## 14. Labs & Practice

- Use [OWASP Juice Shop](https://owasp.org/projects/juice-shop) to practice writing findings and identifying the relevant application security controls. Capture the expected behavior, observed result, and proposed fix in [[#16. Reporting & Documentation|the finding template]].

## 15. Research & Methodologies

- Use [NIST SP 800-115](https://csrc.nist.gov/pubs/sp/800/115/final) to structure assessment planning, evidence collection, and reporting.
- Map observed behavior to [MITRE ATT&CK](https://attack.mitre.org/) when it helps explain detection coverage. Save the technique ID and evidence supporting the mapping.

## 16. Reporting & Documentation

Use a consistent set of fields for each finding.

| Field | What to capture |
| --- | --- |
| Title, scope, and evidence | Affected asset, expected control, observed behavior, UTC timestamp, and a reference to redacted evidence. |
| Impact and severity | Who or what is affected, relevant preconditions, and the rationale for severity. Include a [CVSS v4.0 vector](https://www.first.org/cvss/v4.0/) if used. |
| Remediation and owner | Recommended change and the team responsible for it. |
| Retest | Acceptance criteria, date, observed result, and evidence reference. |

> [!todo] To Explore
> - [ ] Map endpoint and DNS log sources to the coverage gaps in [[#11. Defense & Detection|Defense & Detection]].
> - [ ] Write a sample finding in [[#14. Labs & Practice|the lab]] and use it to check the reporting fields.
