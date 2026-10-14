#!/usr/bin/env python3
"""Strict comparison of the five valid-MPI records, preserving full-log hashes.
Scope: the 1920x1080 YUV420 calibration + 4096-byte MB case only.
No timing, pointer normalization, stderr suppression, or hardware-equivalence claim.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED = {
    'cal_rc': {'cal_rc':0,'width':1920,'height':1088,'bytes':3133440,'stride':1920},
    'init_rc': {'init_rc':0},
    'alloc_rc': {'alloc_rc':0},
    'mb_size': {'mb_size':4096,'bytes_ok':1,'before':1,'add_result':0,'after':2},
    'release1': {'release1':0,'release2':0,'exit_rc':0},
}

def parse(text):
    records={}
    for line in text.splitlines():
        prefix=line.partition('=')[0]
        if prefix not in EXPECTED:
            continue  # Vendor diagnostics remain in the original hashed file.
        if prefix in records:
            raise ValueError('duplicate record: '+prefix)
        record={}
        for token in line.split():
            match=re.fullmatch(r'([a-z][a-z0-9_]*)=(-?[0-9]+)',token)
            if match is None:raise ValueError('malformed token: '+token)
            key,value=match.groups()
            if key in record:raise ValueError('duplicate field: '+key)
            record[key]=int(value)
        if set(record)!=set(EXPECTED[prefix]):raise ValueError('field set mismatch: '+prefix)
        records[prefix]=record
    if set(records)!=set(EXPECTED):raise ValueError('missing required record')
    return records

def compare(vendor_text,candidate_text,vendor_rc,candidate_rc):
    if vendor_rc!=0 or candidate_rc!=0:raise ValueError('nonzero process exit')
    v=parse(vendor_text);c=parse(candidate_text)
    differences=[]
    for record,fields in EXPECTED.items():
        for field,expected in fields.items():
            left,right=v[record][field],c[record][field]
            if left!=right or left!=expected:
                differences.append({'record':record,'field':field,'vendor':left,'candidate':right,'expected_valid_case':expected})
    return {'scope':'valid_mpi_calibration_and_mb_only','pass':not differences,'differences':differences,'vendor_records':v,'candidate_records':c}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('vendor-stdout','candidate-stdout','vendor-rc','candidate-rc'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); paths={k:v for k,v in vars(args).items() if k!='output'}
    try:
        result=compare(args.vendor_stdout.read_text(),args.candidate_stdout.read_text(),int(args.vendor_rc.read_text().strip()),int(args.candidate_rc.read_text().strip()))
    except (OSError,UnicodeError,ValueError) as exc:
        result={'scope':'valid_mpi_calibration_and_mb_only','pass':False,'error':str(exc)}
    result['input_hashes']={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items() if v.is_file()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('VALID_MPI_COMPARATOR_PASS' if result['pass'] else 'VALID_MPI_COMPARATOR_FAIL')
    return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
