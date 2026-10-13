"""Machine-readable Q0-Q10 gate orchestration; evidence originates from other owners."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import copy
from collections.abc import Mapping


class GateBlocked(RuntimeError):
    pass


GATES={
 'Q0':'Runtime', 'Q1':'Data', 'Q2':'Engine', 'Q3':'Env Contract',
 'Q4':'Algorithms','Q5':'Hybrid','Q6':'Research Validity','Q7':'Compute',
 'Q8':'Artifact','Q9':'Upgrade','Q10':'Paper Ready',
}
# Explicit prerequisite graph. Reporting Q10 never changes research release Q0–Q9.
PREREQUISITES={
 'Q0':(), 'Q1':('Q0',), 'Q2':('Q0','Q1'), 'Q3':('Q2',),
 'Q4':('Q3',), 'Q5':('Q4',), 'Q6':('Q1','Q2','Q3','Q4','Q5'),
 'Q7':('Q4','Q5'), 'Q8':('Q6','Q7'), 'Q9':('Q2','Q4','Q8'),
 'Q10':tuple(f'Q{i}' for i in range(10)),
}


# Mandatory check names are part of the CI/release policy, not caller-defined.
GATE_CHECKS={
 'Q0': frozenset({'cp314_abi','cpu_train_save_load','cuda_train_save_load','dependency_hashes'}),
 'Q1': frozenset({'data_quality','available_at','fold_purge_embargo','future_leakage'}),
 'Q2': frozenset({'event_order','fills_fees_funding_margin','terminal_golden'}),
 'Q3': frozenset({'observation_action_execution_reward','normalization_freeze','property_checks'}),
 'Q4': frozenset({'ppo','sac','td3','ddpg','pgpe','cma_es','restore_eval'}),
 'Q5': frozenset({'actor_genome_parity','e1_r1_e2_fold'}),
 'Q6': frozenset({'wfo','baselines','bootstrap_dsr_pbo','selection_ledger','one_shot_holdout'}),
 'Q7': frozenset({'cpu_gpu_hybrid_e2e','rl_sync_barrier','es_async_candidates','oom_timeout_retry'}),
 'Q8': frozenset({'seed_tree','atomic_store','l0_l1_restore','cache_integrity'}),
 'Q9': frozenset({'dual_adapter_golden','dependency_regression','rollback_drill','drift_approved'}),
 'Q10': frozenset({'latency_sla','disconnect','kill_switch','secret_isolation'}),
}


@dataclass(frozen=True)
class EvidenceRef:
    id: str
    path: str
    sha256: str
    producer: str


def _json(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)


def _sha(raw:bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_evidence(item: EvidenceRef, *, trusted_producers:set[str]) -> None:
    if not item.id or not item.producer or item.producer not in trusted_producers:
        raise GateBlocked('unknown or untrusted evidence producer')
    if len(item.sha256)!=64 or any(x not in '0123456789abcdef' for x in item.sha256):
        raise GateBlocked('bad evidence sha256')
    p=Path(item.path)
    if not p.is_file() or p.is_symlink():
        raise GateBlocked('evidence file missing or unsafe')
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    if digest.hexdigest()!=item.sha256:
        raise GateBlocked('evidence file hash mismatch')


class GateRunner:
    """Gate pass requires witnessed evidence; no tests are bundled with L6 deliverable."""

    def __init__(self, *, trusted_producers:set[str]):
        if not trusted_producers:
            raise ValueError('trusted_producers required')
        self.trusted=frozenset(trusted_producers)
        self._reports={}

    def evaluate(self, gate:str, *, evidence:list[EvidenceRef], checks:Mapping[str,bool], reviewer_approved:bool=False) -> dict:
        if gate not in GATES or gate in self._reports:
            raise GateBlocked('unknown gate or gate already evaluated')
        prerequisites=PREREQUISITES[gate]
        blocked=[x for x in prerequisites if self._reports.get(x,{}).get('status')!='PASS']
        if set(checks)!=GATE_CHECKS[gate] or any(not isinstance(v,bool) for v in checks.values()):
            raise GateBlocked('check set drift; missing/extra or nonboolean check')
        for item in evidence:
            verify_evidence(item,trusted_producers=set(self.trusted))
        if len(evidence)!=len(GATE_CHECKS[gate]) or {e.id for e in evidence}!=GATE_CHECKS[gate]:
            raise GateBlocked('each frozen check requires one uniquely keyed evidence item')
        # Evidence files are authored/verified outside this module; a boolean alone cannot PASS.
        status='PASS' if not blocked and evidence and all(checks.values()) and reviewer_approved is True else 'BLOCKED'
        report={'schema_version':1,'gate':gate,'title':GATES[gate], 'status':status,
                'prerequisites':list(prerequisites),'blocked_prerequisites':blocked,
                'checks':dict(checks),'evidence':[e.__dict__ if hasattr(e,'__dict__') else
                   {f:getattr(e,f) for f in ('id','path','sha256','producer')} for e in evidence],
                'reviewer_approved':reviewer_approved}
        report['report_sha256']=_sha(_json(report).encode())
        self._reports[gate]=report
        return copy.deepcopy(report)

    def research_release_ready(self) -> bool:
        return all(self._reports.get(f'Q{i}',{}).get('status')=='PASS' for i in range(10))

    def reports(self) -> dict:
        return copy.deepcopy(self._reports)
