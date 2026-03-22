# Beaconing Hunting SPL

index=proxy
| bin _time span=5m
| stats count as hits by src_ip dest_domain _time
| stats dc(_time) as time_bins, sum(hits) as total_hits by src_ip dest_domain
| where time_bins>=12 AND total_hits>=12
| sort - total_hits
