#!/usr/bin/env python3
"""Check named dynamic exports against the checked-in map, not vendor behavior."""
import argparse, pathlib, re, subprocess
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--map',required=True);p.add_argument('--nm',default='nm');a=p.parse_args()
text=pathlib.Path(a.map).read_text();section=text.split('global:',1)[1].split('local:',1)[0]
expected=set(re.findall(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*;',section,re.M))
result=subprocess.run([a.nm,'-D','--defined-only','--format=posix',a.library],capture_output=True,text=True,check=True)
actual={line.split()[0].split('@')[0] for line in result.stdout.splitlines() if line.split()}
missing=sorted(expected-actual);extra=sorted(actual-expected)
if missing or extra: raise SystemExit(f'export mismatch missing={missing} extra={extra}')
print(f'EXPORT_NAMES_PASS count={len(actual)} behavior=NOT_TESTED')
