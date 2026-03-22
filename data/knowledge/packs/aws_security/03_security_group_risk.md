# Security Group Risk

## Risk patterns
- 0.0.0.0/0 open to SSH(22), RDP(3389), DB ports
- Wide inbound rules without justification

## Actions
- Restrict to known IP ranges/VPN
- Use SSM Session Manager instead of SSH/RDP
- Add detection on risky ingress changes via CloudTrail
