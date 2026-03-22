# Phishing Playbook

## Key indicators
- Lookalike domains, urgent language, credential harvest links
- New mailbox rules/forwarding, OAuth consent to unknown apps

## Triage
1) Collect headers, URLs, attachment hashes
2) Confirm user interaction (clicked? entered creds?)
3) Check sign-in logs (new geo, impossible travel)
4) Check mailbox rules and OAuth grants
5) Search org-wide for same IOCs

## Actions
- Quarantine message, block URL/domain
- Reset password + revoke sessions if compromise suspected
- Remove rules, revoke OAuth grants
