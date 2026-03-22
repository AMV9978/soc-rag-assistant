# IAM Access Key Compromise

## Indicators
- API calls from unusual geography/IP
- High-volume enumeration (List*, Describe*)
- Rapid creation of new users/keys
- Failed auth followed by success

## Response
- Disable key immediately
- Investigate CloudTrail for actions
- Rotate credentials, audit permissions
