# Auth Hunting SPL

## Password spraying
index=auth ("Failed password" OR "invalid password")
| stats dc(user) as unique_users, count as failures by src_ip
| where unique_users>=10 AND failures>=50
| sort - failures

## Brute force
index=auth ("Failed password" OR "invalid password")
| stats count as failures, values(src_ip) as src_ips by user
| where failures>=30
| sort - failures
