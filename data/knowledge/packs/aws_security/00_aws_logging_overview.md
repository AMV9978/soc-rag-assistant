# AWS Logging Overview

## Core logs
- CloudTrail: API calls (control plane)
- VPC Flow Logs: network traffic metadata
- CloudWatch Logs: service/app logs
- GuardDuty: threat detection findings

## SOC approach
1) Confirm finding and affected resource
2) Pull CloudTrail around the time window
3) Check IAM changes and unusual API calls
4) Contain (disable keys, lock down policies)
