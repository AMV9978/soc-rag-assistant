# Account Compromise Playbook

## Indicators
- Many failed logins, unusual geo, impossible travel
- MFA reset events, password reset from rare IP
- New inbox forwarding rules, OAuth grants, privilege changes

## Triage
1) Confirm user + device context (VPN? travel?)
2) Review sign-in logs and session activity
3) Check mailbox rules / OAuth consent
4) Identify impacted apps (SSO logs)

## Actions
- Reset password, enforce MFA, revoke sessions/tokens
- Remove malicious rules and OAuth grants
- Monitor for re-compromise
