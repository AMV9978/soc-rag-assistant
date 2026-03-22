#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("data/knowledge/packs")

def safe_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")

def main():
    packs = {
        "fundamentals": {
            "00_what_is_cybersecurity.md": "# What is Cybersecurity?\n\nCybersecurity protects systems, networks, and data from unauthorized access.\n\n## CIA Triad\n- Confidentiality: prevent unauthorized disclosure\n- Integrity: prevent unauthorized modification\n- Availability: ensure systems are accessible\n\n## Common threats\n- Phishing & social engineering\n- Malware/ransomware\n- Credential attacks\n- Misconfiguration",
            "01_core_security_concepts.md": "# Core Security Concepts\n\n## Risk basics\n- Asset: something valuable\n- Threat: potential cause of harm\n- Vulnerability: weakness that can be exploited\n- Control: safeguard that reduces risk\n\n## Control types\n- Preventive: stops an event (MFA, patching)\n- Detective: finds it (SIEM alerts)\n- Corrective: recovers (restore backups)\n\n## AuthN vs AuthZ\n- Authentication: who you are\n- Authorization: what you can do",
            "02_logging_basics.md": "# Logging Basics\n\n## Why logging matters\nLogs are evidence for detection, investigations, and compliance.\n\n## What to log\n- Authentication events (success/fail, MFA, resets)\n- Admin actions (group changes, policy edits)\n- Endpoint activity (process creation, PowerShell)\n- Network/DNS/proxy traffic\n- Cloud control plane logs (AWS CloudTrail)\n\n## Good hygiene\n- Centralize in SIEM\n- Normalize fields\n- Time sync (NTP)\n- Retention policy",
        },
        "soc_playbooks": {
            "00_triage_framework.md": "# SOC Triage Framework\n\n## Minimum steps\n1) Validate (false positive vs true positive)\n2) Identify entity (host/user/app)\n3) Collect evidence (logs/process/network/auth)\n4) Scope (how many systems/users?)\n5) Contain if risk of spread\n6) Escalate per severity\n7) Document timeline\n\n## Evidence checklist\n- Who / What / Where / When / How\n- IOCs: hashes, domains, IPs, filenames",
            "01_phishing_playbook.md": "# Phishing Playbook\n\n## Key indicators\n- Lookalike domains, urgent language, credential harvest links\n- New mailbox rules/forwarding, OAuth consent to unknown apps\n\n## Triage\n1) Collect headers, URLs, attachment hashes\n2) Confirm user interaction (clicked? entered creds?)\n3) Check sign-in logs (new geo, impossible travel)\n4) Check mailbox rules and OAuth grants\n5) Search org-wide for same IOCs\n\n## Actions\n- Quarantine message, block URL/domain\n- Reset password + revoke sessions if compromise suspected\n- Remove rules, revoke OAuth grants",
            "02_ransomware_playbook.md": "# Ransomware Playbook\n\n## Indicators\n- Mass file rename/encryption, ransom note creation\n- Unusual SMB write volume, admin share access\n- Lateral movement preceding impact\n\n## Triage\n1) Isolate endpoints/servers\n2) Identify patient zero + initial access vector\n3) Determine blast radius\n4) Preserve evidence\n5) Extract IOCs\n\n## Actions\n- Block IOCs, disable compromised accounts\n- Validate backups are clean before restore\n- Coordinate with IR/IT/legal; document everything",
            "03_account_compromise_playbook.md": "# Account Compromise Playbook\n\n## Indicators\n- Many failed logins, unusual geo, impossible travel\n- MFA reset events, password reset from rare IP\n- New inbox forwarding rules, OAuth grants, privilege changes\n\n## Triage\n1) Confirm user + device context (VPN? travel?)\n2) Review sign-in logs and session activity\n3) Check mailbox rules / OAuth consent\n4) Identify impacted apps (SSO logs)\n\n## Actions\n- Reset password, enforce MFA, revoke sessions/tokens\n- Remove malicious rules and OAuth grants\n- Monitor for re-compromise",
            "04_malware_alert_playbook.md": "# Malware Alert Playbook\n\n## Indicators\n- EDR detections, suspicious process tree\n- Persistence (scheduled tasks, registry run keys)\n- C2 beaconing to suspicious domains/IPs\n\n## Triage\n1) Isolate host (if high confidence)\n2) Capture process tree, parent/child, command line\n3) Identify persistence and network connections\n4) Search across fleet for same IOCs\n\n## Actions\n- Block IOCs, run EDR remediation\n- Validate no lateral movement; document findings",
        },
        "splunk": {
            "00_spl_basics.md": "# SPL Basics\n\n## Building blocks\n- index=... sourcetype=... host=... src=... dest=...\n- earliest= latest= for time range\n\n## Core commands\n- stats / timechart / where / eval / rex / lookup\n\n## Example: failed logins\nindex=auth \"Failed password\"\n| stats count by user src\n\n## Example: top domains\nindex=proxy\n| stats count by dest_domain\n| sort - count\n| head 20",
            "01_auth_hunting.md": "# Auth Hunting SPL\n\n## Password spraying\nindex=auth (\"Failed password\" OR \"invalid password\")\n| stats dc(user) as unique_users, count as failures by src_ip\n| where unique_users>=10 AND failures>=50\n| sort - failures\n\n## Brute force\nindex=auth (\"Failed password\" OR \"invalid password\")\n| stats count as failures, values(src_ip) as src_ips by user\n| where failures>=30\n| sort - failures",
            "02_dns_tunneling_hunting.md": "# DNS Tunneling Hunting SPL\n\n## Long DNS queries\nindex=dns\n| eval qlen=len(query)\n| where qlen>60\n| stats count avg(qlen) max(qlen) by src_ip query\n| sort - count\n\n## NXDOMAIN spikes\nindex=dns response_code=NXDOMAIN\n| stats count by src_ip query\n| sort - count",
            "03_beaconing_hunting.md": "# Beaconing Hunting SPL\n\nindex=proxy\n| bin _time span=5m\n| stats count as hits by src_ip dest_domain _time\n| stats dc(_time) as time_bins, sum(hits) as total_hits by src_ip dest_domain\n| where time_bins>=12 AND total_hits>=12\n| sort - total_hits",
            "04_powershell_hunting.md": "# PowerShell Hunting SPL\n\nindex=windows EventCode=4688 (New_Process_Name=\"*powershell.exe\")\n| eval cmd=coalesce(CommandLine, Process_Command_Line)\n| where like(cmd,\"% -enc %\") OR like(cmd,\"%IEX%\") OR like(cmd,\"%downloadstring%\")\n| table _time host user ParentProcessName cmd\n| sort - _time",
            "05_ransomware_signals.md": "# Ransomware Signals SPL\n\n## File spike\nindex=edr (event_type=file_write OR event_type=file_rename)\n| timechart span=5m count by host\n| where count>200\n\n## SMB admin share\nindex=windows EventCode=5140\n| stats count by AccountName IpAddress host\n| sort - count",
        },
        "aws_security": {
            "00_aws_logging_overview.md": "# AWS Logging Overview\n\n## Core logs\n- CloudTrail: API calls (control plane)\n- VPC Flow Logs: network traffic metadata\n- CloudWatch Logs: service/app logs\n- GuardDuty: threat detection findings\n\n## SOC approach\n1) Confirm finding and affected resource\n2) Pull CloudTrail around the time window\n3) Check IAM changes and unusual API calls\n4) Contain (disable keys, lock down policies)",
            "01_cloudtrail_triage.md": "# CloudTrail Triage\n\n## High-risk events\n- iam:CreateAccessKey / UpdateAssumeRolePolicy\n- iam:AttachUserPolicy / PutUserPolicy\n- sts:AssumeRole from unusual principal\n- ec2:AuthorizeSecurityGroupIngress (0.0.0.0/0)\n- s3:PutBucketPolicy / PutPublicAccessBlock\n\n## Triage steps\n1) Identify actor, src IP, MFA used?\n2) What changed? Which resource?\n3) Was it expected?\n4) Contain: revoke keys, detach policies",
            "02_s3_exposure.md": "# S3 Exposure Basics\n\n## Indicators\n- Bucket policy allows public access\n- ACL grants READ/WRITE to Everyone\n- PublicAccessBlock disabled\n\n## Actions\n- Enable PublicAccessBlock\n- Fix bucket policy/ACL\n- Review CloudTrail for data access\n- Rotate any exposed credentials",
            "03_security_group_risk.md": "# Security Group Risk\n\n## Risk patterns\n- 0.0.0.0/0 open to SSH(22), RDP(3389), DB ports\n- Wide inbound rules without justification\n\n## Actions\n- Restrict to known IP ranges/VPN\n- Use SSM Session Manager instead of SSH/RDP\n- Add detection on risky ingress changes via CloudTrail",
            "04_iam_key_compromise.md": "# IAM Access Key Compromise\n\n## Indicators\n- API calls from unusual geography/IP\n- High-volume enumeration (List*, Describe*)\n- Rapid creation of new users/keys\n- Failed auth followed by success\n\n## Response\n- Disable key immediately\n- Investigate CloudTrail for actions\n- Rotate credentials, audit permissions",
        },
        "iam": {
            "00_iam_fundamentals.md": "# IAM Fundamentals\n\n## What is IAM?\nIdentity and Access Management ensures the right people have the right access at the right time.\n\n## Core pieces\n- Identity (user/service account)\n- Authentication (password, MFA)\n- Authorization (roles, policies)\n- Provisioning/Deprovisioning\n\n## Best practices\n- Least privilege\n- Strong MFA\n- Conditional access\n- Regular access reviews",
            "01_mfa_events.md": "# MFA Events and Triage\n\n## Indicators\n- MFA disabled or reset\n- New device enrollment\n- Many MFA prompts (push fatigue)\n- Auth success without MFA where expected\n\n## Actions\n- Confirm user intent\n- Enforce MFA policies\n- Revoke sessions/tokens after suspicious resets\n- Add alerting for MFA changes",
            "02_oauth_consent_abuse.md": "# OAuth Consent Abuse\n\n## What happens\nUser grants a malicious app access. Attacker gains persistence without password.\n\n## Indicators\n- New OAuth app consent\n- Unusual scopes (mail.read, offline_access)\n- Mailbox forwarding + token reuse\n\n## Actions\n- Revoke app consent\n- Review audit logs\n- Reset creds + revoke sessions",
            "03_access_reviews.md": "# Access Reviews\n\n## Goal\nEnsure users have only the access they need.\n\n## What to review\n- Admin roles\n- Privileged groups\n- Service accounts\n- Dormant accounts\n- External collaborators\n\n## Output\n- Remove unnecessary access\n- Document approvals and exceptions",
            "04_privileged_access.md": "# Privileged Access Management\n\n## Why it is risky\nPrivileged accounts can change security controls and access sensitive data.\n\n## Controls\n- Just-in-time access\n- Approval workflows\n- Session recording\n- Credential rotation / vaulting\n- Break-glass accounts with tight monitoring",
        },
        "mitre_attack": {
            "00_mapping_basics.md": "# MITRE ATT&CK Mapping Basics\n\n## Tactic vs Technique\n- Tactic = adversary goal (e.g., Initial Access)\n- Technique = how (e.g., Phishing T1566)\n\n## Why it helps\n- Common language, coverage tracking, reporting\n\n## Examples\n- Phishing: T1566 (Initial Access)\n- Remote Services: T1021 (Lateral Movement)\n- Credential Dumping: T1003 (Credential Access)",
            "01_example_mappings.md": "# Example MITRE Mappings\n\n## Phishing\n- Tactic: Initial Access\n- Technique: T1566\n\n## DNS Tunneling\n- Tactic: Exfiltration / Command and Control\n- Technique: DNS-based C2 or exfil patterns\n\n## Lateral Movement\n- Tactic: Lateral Movement\n- Technique: T1021 (Remote Services)",
        },
    }

    total = 0
    for pack_name, files in packs.items():
        for fname, content in files.items():
            fp = BASE_DIR / pack_name / fname
            safe_write(fp, content)
            total += 1
            print(f"  ✓ {pack_name}/{fname}")

    print(f"\n✅ Done! {total} files written to {BASE_DIR}")

if __name__ == "__main__":
    main()
