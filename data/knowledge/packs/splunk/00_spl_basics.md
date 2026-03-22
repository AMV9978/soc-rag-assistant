# SPL Basics

## Building blocks
- index=... sourcetype=... host=... src=... dest=...
- earliest= latest= for time range

## Core commands
- stats / timechart / where / eval / rex / lookup

## Example: failed logins
index=auth "Failed password"
| stats count by user src

## Example: top domains
index=proxy
| stats count by dest_domain
| sort - count
| head 20
