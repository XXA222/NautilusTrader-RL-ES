"""Reverify locally supplied Q0-Q10 reports before release preparation.

L6 does not sign or manufacture upstream evidence. Producer allowlists and
reviewer-approved booleans are trust assertions; a separate external signing
policy is needed when the local filesystem / reviewers are not trusted.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .gates import EvidenceRef, GATE_CHECKS, GATES, PREREQUISITES, GateBlocked, verify_evidence


def _digest(payload: object) -> str:
    try:
        raw=json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise GateBlocked('noncanonical gate report') from exc
    return hashlib.sha256(raw).hexdigest()


def verify_gate_reports(reports: Mapping[str, Mapping[str, object]], *,
                        trusted_producers: set[str], require_paper: bool = False) -> dict[str, str]:
    """Require an unaltered PASS report and one verified evidence per frozen check.

    A report is checked against the exact L6 gate catalog. The dependency graph
    is not caller-supplied and Q10 is never a prerequisite for research release.
    """
    if not trusted_producers or not isinstance(reports, Mapping):
        raise GateBlocked('trusted evidence producers and report mapping required')
    required={f'Q{i}' for i in range(11 if require_paper else 10)}
    if set(reports) != required:
        raise GateBlocked('missing/extra research/paper gate reports')
    accepted={}
    for gate in (f'Q{i}' for i in range(11 if require_paper else 10)):
        report=reports[gate]
        if not isinstance(report,Mapping):
            raise GateBlocked('gate report must be a mapping')
        fields={'schema_version','gate','title','status','prerequisites','blocked_prerequisites',
                'checks','evidence','reviewer_approved','report_sha256'}
        if set(report)!=fields or type(report['schema_version']) is not int or report['schema_version']!=1:
            raise GateBlocked(f'{gate} schema drift')
        actual=_digest({k:v for k,v in report.items() if k!='report_sha256'})
        if report['report_sha256']!=actual:
            raise GateBlocked(f'{gate} report content/hash mismatch')
        if (report['gate']!=gate or report['title']!=GATES[gate] or report['status']!='PASS'
            or report['reviewer_approved'] is not True
            or report['prerequisites']!=list(PREREQUISITES[gate])
            or report['blocked_prerequisites']!=[]):
            raise GateBlocked(f'{gate} not explicitly approved or dependency graph drift')
        if any(dependency not in accepted for dependency in PREREQUISITES[gate]):
            raise GateBlocked(f'{gate} incomplete gate prerequisites')
        checks=report['checks']
        if not isinstance(checks, Mapping) or set(checks)!=GATE_CHECKS[gate] or any(v is not True for v in checks.values()):
            raise GateBlocked(f'{gate} missing/failed required checks')
        items=report['evidence']
        if not isinstance(items,list) or len(items)!=len(GATE_CHECKS[gate]):
            raise GateBlocked(f'{gate} needs exactly one evidence item per check')
        try:
            refs=[EvidenceRef(**item) for item in items]
        except (TypeError, KeyError) as exc:
            raise GateBlocked(f'{gate} invalid evidence schema') from exc
        if {ref.id for ref in refs}!=GATE_CHECKS[gate]:
            raise GateBlocked(f'{gate} each frozen check needs its own evidence')
        for ref in refs:
            verify_evidence(ref,trusted_producers=trusted_producers)
        accepted[gate]=actual
    return accepted


def load_gate_reports(directory: Path, *, trusted_producers: set[str],
                      require_paper: bool = False) -> dict[str,str]:
    """Read externally produced reports; does not fabricate missing evidence."""
    count=11 if require_paper else 10
    reports={}
    for n in range(count):
        gate=f'Q{n}'
        path=Path(directory)/f'{gate}.json'
        if not path.is_file() or path.is_symlink() or path.stat().st_size>2_000_000:
            raise GateBlocked(f'{gate} report missing/unsafe or oversized')
        try:
            reports[gate]=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:
            raise GateBlocked(f'{gate} report unreadable') from exc
    return verify_gate_reports(reports,trusted_producers=trusted_producers,require_paper=require_paper)


def main() -> int:
    """Reverify externally generated Q0–Q9 evidence; never create a PASS report."""
    import argparse
    parser=argparse.ArgumentParser(description='Fail-closed L6 gate evidence verifier')
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--trusted-producers',type=Path,required=True)
    args=parser.parse_args()
    src=args.trusted_producers
    if not src.is_file() or src.is_symlink() or src.stat().st_size>100_000:
        raise GateBlocked('reviewer trusted producer registry missing/unsafe')
    try:
        producers=json.loads(src.read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        raise GateBlocked('invalid reviewer producer registry') from exc
    if not isinstance(producers,list) or not producers or any(not isinstance(x,str) or not x for x in producers):
        raise GateBlocked('expected nonempty JSON list of approved evidence producers')
    result=load_gate_reports(args.reports,trusted_producers=set(producers))
    print('Q0-Q9 report/evidence content verified; still requires external signed review')
    print('Gate hashes:', ', '.join(f'{k}={v}' for k,v in sorted(result.items())))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
