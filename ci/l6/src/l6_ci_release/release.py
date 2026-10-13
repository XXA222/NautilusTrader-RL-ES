"""L6 release manifest assembly ONLY, never merge/tag/redistribute upstream source."""
from __future__ import annotations

from pathlib import Path
import hashlib
from collections.abc import Mapping
from .gates import GateBlocked
from .attestation import verify_gate_reports


class ReleaseManifestBuilder:
    def __init__(self, *, reviewer_approved:bool, gates:Mapping[str,Mapping[str,object]],
                 trusted_producers: set[str]):
        if not reviewer_approved:
            raise GateBlocked('user release authorization required')
        self.gate_hashes = verify_gate_reports(gates, trusted_producers=trusted_producers)
        self.gates = {key: dict(value) for key,value in gates.items()}

    def build(self, *, source_refs:Mapping[str,Path],wheel_refs:Mapping[str,Path],
              lock_refs:Mapping[str,Path],sbom:Path,notice:Path,docs:Mapping[str,Path],
              cards:Mapping[str,Path],reproducibility_statement:str,source_tag:str)->dict:
        if not source_tag or not reproducibility_statement:
            raise GateBlocked('tag proposal and honest reproducibility statement required')
        if any(not d for d in (source_refs,wheel_refs,lock_refs,docs,cards)):
            raise GateBlocked('source, wheel, lock, docs and model/data cards required')
        inventory={}
        for category,files in {'source':source_refs,'wheels':wheel_refs,'locks':lock_refs,
                               'documents':docs,'cards':cards,'legal':{'SBOM':sbom,'NOTICE':notice}}.items():
            entries={}
            for name,path in files.items():
                p=Path(path)
                if not p.is_file() or p.is_symlink():
                    raise GateBlocked('release file missing/unsafe')
                digest=hashlib.sha256()
                with p.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(block)
                entries[name]={'source_path':str(p),'sha256':digest.hexdigest(),
                               'size_bytes':p.stat().st_size}
            inventory[category]=entries
        return {'schema_version':1,'status':'READY_FOR_USER_ASSEMBLY_NOT_RELEASED',
                'proposed_tag':source_tag,'inventory':inventory,
                'gate_report_hashes':dict(self.gate_hashes),
                'reproducibility_statement':reproducibility_statement,
                'no_automatic_tag_or_merge':True}


def evidence_bundle_index(*, reports:Mapping[str,Path],reviewer_approved:bool)->dict:
    needed={'research','compute','artifact','upgrade','release'}
    if not reviewer_approved or set(reports)!=needed:
        raise GateBlocked('Q6–Q9 evidence index requires complete reports and human approval')
    entries={}
    for k,v in reports.items():
        p=Path(v)
        if not p.is_file() or p.is_symlink():raise GateBlocked('evidence report missing')
        entries[k]=hashlib.sha256(p.read_bytes()).hexdigest()
    return {'status':'READY_FOR_USER_REVIEW','evidence_sha256':entries}
