#!/usr/bin/env python3
"""Reject checked-in software binaries; PDFs are intentionally permitted documents."""
import argparse, pathlib, subprocess
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=pathlib.Path(a.root).resolve()
r=subprocess.run(['git','-C',str(root),'ls-files','-z'],capture_output=True,check=True)
paths=[root/s for s in r.stdout.decode().split('\0') if s]
if not paths: raise SystemExit('no tracked source files; checkout evidence required')
bad=[]
for path in paths:
 if not path.is_file(): continue
 with path.open('rb') as f: prefix=f.read(8)
 if path.suffix.lower() in {'.so','.a','.o','.obj','.dll','.exe','.lib'} or prefix.startswith((b'\x7fELF',b'!<arch>\n',b'MZ')): bad.append(str(path.relative_to(root)))
if bad: raise SystemExit('software binaries tracked: '+repr(bad))
print(f'SOURCE_ONLY_PASS tracked_files={len(paths)}; historical Git objects not inspected')
