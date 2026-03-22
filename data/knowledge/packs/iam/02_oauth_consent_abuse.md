# OAuth Consent Abuse

## What happens
User grants a malicious app access. Attacker gains persistence without password.

## Indicators
- New OAuth app consent
- Unusual scopes (mail.read, offline_access)
- Mailbox forwarding + token reuse

## Actions
- Revoke app consent
- Review audit logs
- Reset creds + revoke sessions
