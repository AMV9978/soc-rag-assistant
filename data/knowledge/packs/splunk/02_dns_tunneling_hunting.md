# DNS Tunneling Hunting SPL

## Long DNS queries
index=dns
| eval qlen=len(query)
| where qlen>60
| stats count avg(qlen) max(qlen) by src_ip query
| sort - count

## NXDOMAIN spikes
index=dns response_code=NXDOMAIN
| stats count by src_ip query
| sort - count
