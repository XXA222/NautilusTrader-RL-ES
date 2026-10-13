"""Dual-adapter comparison, dependency regression and explicit rollback plans."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose, isfinite
from hashlib import sha256
import json


class UpgradeBlocked(RuntimeError):
    pass


def _sha(o:object)->str:
    return sha256(json.dumps(o,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def compare_adapter_traces(old:Sequence[Mapping[str,object]],new:Sequence[Mapping[str,object]],*,
                           numeric_tolerances:Mapping[str,float], expected_fix_paths:set[str],
                           reviewer_approved_paths:set[str]) -> dict:
    """Records ALL paths. A/B never auto-accept; C requires rollback."""
    if any(type(v) not in (int,float) or v<0 or not isfinite(v) for v in numeric_tolerances.values()):
        raise ValueError('invalid preregistered tolerances')
    if len(old)!=len(new):
        return {'status':'BLOCKED','breaking':['event_count'],
                'requires_reviewer_approval':[],
                'differences':[{'path':'event_count','kind':'C','old':len(old),'new':len(new)}],
                'old_trace_sha256':_sha(list(old)),'new_trace_sha256':_sha(list(new)),
                'tolerance_sha256':_sha(dict(numeric_tolerances))}
    differences=[]
    def compare(a,b,path):
        if isinstance(a,Mapping) and isinstance(b,Mapping):
            for k in sorted(set(a)|set(b)):
                name=f'{path}.{k}'
                if k not in a or k not in b:
                    differences.append({'path':name,'kind':'C','reason':'missing_field'})
                else:compare(a[k],b[k],name)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):
                differences.append({'path':path,'kind':'C','reason':'array_length'})
            else:
                for j,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{j}]')
        elif isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
            if a!=b:
                tolerance=numeric_tolerances.get(path)
                if path in expected_fix_paths:kind='A'
                elif tolerance is not None and isclose(a,b,rel_tol=0,abs_tol=tolerance):kind='B'
                else:kind='C'
                differences.append({'path':path,'kind':kind,'old':a,'new':b})
        elif a!=b:
            kind='A' if path in expected_fix_paths else 'C'
            differences.append({'path':path,'kind':kind,'old':a,'new':b})
    for i,(a,b) in enumerate(zip(old,new)):
        compare(a,b,f'event[{i}]')
    unapproved=[d['path'] for d in differences if d['kind'] in {'A','B'} and d['path'] not in reviewer_approved_paths]
    breaking=[d['path'] for d in differences if d['kind']=='C']
    return {'status':'BLOCKED' if breaking or unapproved else 'APPROVED',
            'breaking':breaking,'requires_reviewer_approval':unapproved,
            'differences':differences,'old_trace_sha256':_sha(list(old)),'new_trace_sha256':_sha(list(new)),
            'tolerance_sha256':_sha(dict(numeric_tolerances))}


def dependency_regression(versions:Mapping[str,str], checks:Mapping[str,bool],
                          evidence_hashes:Mapping[str,str])->dict:
    required_versions={'python','torch','stable_baselines3','evotorch','nautilus_trader'}
    required_checks={'train','save','load','parity','engine_golden','restore'}
    if set(versions)!=required_versions or set(checks)!=required_checks or set(evidence_hashes)!=required_checks:
        raise UpgradeBlocked('incomplete dependency/ABI acceptance matrix')
    if any(not isinstance(x,bool) for x in checks.values()) or any(not x for x in versions.values()):
        raise UpgradeBlocked('malformed dependency evidence')
    if any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in evidence_hashes.values()):
        raise UpgradeBlocked('evidence hash missing/malformed')
    return {'status':'PASS' if all(checks.values()) else 'BLOCKED','versions':dict(versions),
            'checks':dict(checks),'evidence_hashes':dict(evidence_hashes)}


def rollback_plan(*, old_lock_sha:str,old_adapter_sha:str,old_artifact_sha:str,
                  replay_old_report_sha:str,drift_report:Mapping[str,object],
                  reviewer_approved:bool)->dict:
    values=(old_lock_sha,old_adapter_sha,old_artifact_sha,replay_old_report_sha)
    if any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in values):
        raise UpgradeBlocked('rollback requires pinned old inputs and reproduced old report')
    return {'status':'READY_FOR_REVIEWER' if reviewer_approved and drift_report.get('status')=='BLOCKED' else 'BLOCKED',
            'old_lock_sha256':old_lock_sha,'old_adapter_sha256':old_adapter_sha,
            'old_artifact_sha256':old_artifact_sha,'replayed_old_report_sha256':replay_old_report_sha,
            'requires_external_rollback_execution':True}
