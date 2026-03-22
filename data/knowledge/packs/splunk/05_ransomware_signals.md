# Ransomware Signals SPL

## File spike
index=edr (event_type=file_write OR event_type=file_rename)
| timechart span=5m count by host
| where count>200

## SMB admin share
index=windows EventCode=5140
| stats count by AccountName IpAddress host
| sort - count
