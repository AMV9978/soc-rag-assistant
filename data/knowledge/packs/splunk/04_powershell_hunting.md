# PowerShell Hunting SPL

index=windows EventCode=4688 (New_Process_Name="*powershell.exe")
| eval cmd=coalesce(CommandLine, Process_Command_Line)
| where like(cmd,"% -enc %") OR like(cmd,"%IEX%") OR like(cmd,"%downloadstring%")
| table _time host user ParentProcessName cmd
| sort - _time
