# Ransomware Playbook

## Indicators
- Mass file rename/encryption, ransom note creation
- Unusual SMB write volume, admin share access
- Lateral movement preceding impact

## Triage
1) Isolate endpoints/servers
2) Identify patient zero + initial access vector
3) Determine blast radius
4) Preserve evidence
5) Extract IOCs

## Actions
- Block IOCs, disable compromised accounts
- Validate backups are clean before restore
- Coordinate with IR/IT/legal; document everything
