# CloudTrail Triage

## High-risk events
- iam:CreateAccessKey / UpdateAssumeRolePolicy
- iam:AttachUserPolicy / PutUserPolicy
- sts:AssumeRole from unusual principal
- ec2:AuthorizeSecurityGroupIngress (0.0.0.0/0)
- s3:PutBucketPolicy / PutPublicAccessBlock

## Triage steps
1) Identify actor, src IP, MFA used?
2) What changed? Which resource?
3) Was it expected?
4) Contain: revoke keys, detach policies
