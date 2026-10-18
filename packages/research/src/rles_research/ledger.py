"""Persistent, fail-closed research selection accounting and one-shot holdout state."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast


class ResearchLockError(RuntimeError):
    pass


def _json(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)


def _sha(s: str):
    return hashlib.sha256(s.encode()).hexdigest()


class ResearchLedger:
    """All choices use a frozen protocol_id/run_id; external access must be serialized."""

    VALID_QUERIES=frozenset({'validation','hpo_trial','es_evaluation','test','final_holdout'})

    def __init__(self, database: str|Path):
        self.path=Path(database);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._tx() as c:
            c.execute('CREATE TABLE IF NOT EXISTS protocol (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, state TEXT NOT NULL, recipe_sha TEXT, code_sha TEXT, retrain_sha TEXT, holdout_sha TEXT, selection_spec_sha TEXT, validation_limit INTEGER NOT NULL, hpo_limit INTEGER NOT NULL, es_limit INTEGER NOT NULL, resume_state TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS query (query_id TEXT PRIMARY KEY, protocol_id TEXT NOT NULL, kind TEXT NOT NULL, candidate TEXT NOT NULL, fold_id TEXT NOT NULL, recorded REAL NOT NULL, FOREIGN KEY(protocol_id) REFERENCES protocol(id))')
            c.execute('CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, protocol_id TEXT NOT NULL, action TEXT NOT NULL, payload_sha TEXT NOT NULL, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS stage_receipt (protocol_id TEXT NOT NULL, run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, stage TEXT NOT NULL, artifact_id TEXT NOT NULL, payload_hash TEXT NOT NULL, run_fingerprint TEXT NOT NULL, usage_json TEXT NOT NULL, receipt_hash TEXT NOT NULL UNIQUE, recorded REAL NOT NULL, PRIMARY KEY(protocol_id,run_id,attempt_id,stage), UNIQUE(protocol_id,artifact_id), FOREIGN KEY(protocol_id) REFERENCES protocol(id))')
            c.execute('CREATE TABLE IF NOT EXISTS es_lane_request (protocol_id TEXT NOT NULL, run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, stage TEXT NOT NULL, request_key TEXT NOT NULL, request_hash TEXT NOT NULL, recorded REAL NOT NULL, PRIMARY KEY(protocol_id,run_id,attempt_id,stage,request_key), UNIQUE(protocol_id,request_hash), FOREIGN KEY(protocol_id) REFERENCES protocol(id))')
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_shot_fold_test ON query(protocol_id, fold_id) WHERE kind='test'")
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_shot_dataset_holdout ON protocol(holdout_sha) WHERE holdout_sha IS NOT NULL")
            columns={row[1] for row in c.execute('PRAGMA table_info(protocol)')}
            if 'resume_state' not in columns:
                c.execute('ALTER TABLE protocol ADD COLUMN resume_state TEXT')
            if 'pre_holdout_end_ns' not in columns:
                c.execute('ALTER TABLE protocol ADD COLUMN pre_holdout_end_ns INTEGER')
            if 'holdout_start_ns' not in columns:
                c.execute('ALTER TABLE protocol ADD COLUMN holdout_start_ns INTEGER')

    @contextmanager
    def _tx(self):
        c=sqlite3.connect(self.path,timeout=15,isolation_level=None)
        try:
            c.execute('PRAGMA foreign_keys=ON'); c.execute('BEGIN IMMEDIATE')
            yield c; c.execute('COMMIT')
        except BaseException:
            if c.in_transaction:c.execute('ROLLBACK')
            raise
        finally:c.close()

    def _audit(self,c,pid,action,payload):
        last=c.execute('SELECT row_hash FROM audit ORDER BY seq DESC LIMIT 1').fetchone()
        prev=last[0] if last else '0'*64
        psha=_sha(_json(payload))
        h=_sha(_json([pid,action,psha,prev]))
        c.execute('INSERT INTO audit(protocol_id,action,payload_sha,prev_hash,row_hash) VALUES(?,?,?,?,?)',(pid,action,psha,prev,h))

    def create(self, protocol_id: str, run_id: str, *, validation_budget:int, hpo_budget:int, es_evaluation_budget:int) -> None:
        limits=(validation_budget,hpo_budget,es_evaluation_budget)
        if (not all(isinstance(x,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',x)
                    and x not in {'.','..'} for x in (protocol_id,run_id))
            or any(type(v) is not int or v<0 for v in limits) or validation_budget<1):
            raise ValueError('valid IDs, positive validation budget, nonnegative HPO/ES budgets required')
        with self._tx() as c:
            try:
                c.execute('INSERT INTO protocol(id,run_id,state,validation_limit,hpo_limit,es_limit) VALUES(?,?,?,?,?,?)',
                          (protocol_id,run_id,'DEVELOPMENT',*limits))
            except sqlite3.IntegrityError as exc:
                raise ResearchLockError('protocol ID already exists') from exc
            self._audit(c,protocol_id,'CREATE',{'run_id':run_id,'query_budgets':limits})

    def record_query(self, protocol_id: str, query_id: str, kind: str, candidate: str, fold_id: str) -> None:
        if kind not in self.VALID_QUERIES or not all(isinstance(x,str) and 0<len(x)<=200 and not any(ord(c)<32 for c in x)
                                                         for x in (query_id,candidate,fold_id)):
            raise ValueError('invalid research query')
        with self._tx() as c:
            row=c.execute('SELECT state FROM protocol WHERE id=?',(protocol_id,)).fetchone()
            if not row:raise ResearchLockError('unknown protocol')
            state=row[0]
            budget_column={'validation':'validation_limit','hpo_trial':'hpo_limit','es_evaluation':'es_limit'}.get(kind)
            if budget_column:
                limit=c.execute(f'SELECT {budget_column} FROM protocol WHERE id=?',(protocol_id,)).fetchone()[0]
                used=c.execute('SELECT COUNT(*) FROM query WHERE protocol_id=? AND kind=?',(protocol_id,kind)).fetchone()[0]
                if kind=='es_evaluation':
                    used+=c.execute('SELECT COUNT(*) FROM es_lane_request WHERE protocol_id=?',
                                    (protocol_id,)).fetchone()[0]
                if used>=limit:
                    raise ResearchLockError('frozen research query budget exhausted')
            if kind in {'validation','hpo_trial','es_evaluation'} and state!='DEVELOPMENT':
                raise ResearchLockError('protocol frozen; additional tuning prohibited')
            if kind=='test' and state!='RECIPE_FROZEN':
                raise ResearchLockError('fold Test requires frozen recipe')
            if kind=='final_holdout' or state in {'HOLDOUT_OPENED','CLOSED'}:
                raise ResearchLockError('holdout must be opened via one-shot transaction')
            try:
                c.execute('INSERT INTO query(query_id,protocol_id,kind,candidate,fold_id,recorded) VALUES(?,?,?,?,?,?)',
                          (query_id,protocol_id,kind,candidate,fold_id,time.time()))
            except sqlite3.IntegrityError as exc:
                raise ResearchLockError('duplicate query ID or one-shot fold Test already consumed') from exc
            self._audit(c,protocol_id,'QUERY',{'query_id':query_id,'kind':kind,'candidate':candidate,'fold':fold_id})

    def freeze_recipe(self, protocol_id: str, *, recipe_sha: str, code_sha: str, selection_spec_sha: str) -> None:
        if not all(len(x)==64 and all(ch in '0123456789abcdef' for ch in x) for x in (recipe_sha,code_sha,selection_spec_sha)):
            raise ValueError('expected lowercase SHA256')
        with self._tx() as c:
            cur=c.execute("UPDATE protocol SET state='RECIPE_FROZEN',recipe_sha=?,code_sha=?,selection_spec_sha=? WHERE id=? AND state='DEVELOPMENT'",
                (recipe_sha,code_sha,selection_spec_sha,protocol_id))
            if cur.rowcount!=1:raise ResearchLockError('recipe cannot be refrozen')
            self._audit(c,protocol_id,'FREEZE',{'recipe':recipe_sha,'code':code_sha,'spec':selection_spec_sha})

    def register_retrain(self, protocol_id: str, retrain_sha: str, *, pre_holdout_end_ns: int) -> None:
        if not isinstance(retrain_sha,str) or len(retrain_sha)!=64 or any(c not in '0123456789abcdef' for c in retrain_sha) or isinstance(pre_holdout_end_ns,bool) or not isinstance(pre_holdout_end_ns,int) or pre_holdout_end_ns<1:
            raise ValueError('retrain evidence and nanosecond cutoff required')
        with self._tx() as c:
            cur=c.execute("UPDATE protocol SET state='RETRAINED',retrain_sha=?,pre_holdout_end_ns=? WHERE id=? AND state='RECIPE_FROZEN'",
                          (retrain_sha,pre_holdout_end_ns,protocol_id))
            if cur.rowcount!=1:raise ResearchLockError('full pre-holdout retraining requires frozen recipe')
            self._audit(c,protocol_id,'RETRAIN',{'hash':retrain_sha,'pre_holdout_end_ns':pre_holdout_end_ns})

    def open_holdout(self, protocol_id: str, *, holdout_dataset_sha: str, reviewer_approval_sha: str,
                     holdout_start_ns: int) -> str:
        if any(not isinstance(x,str) or len(x)!=64 or any(ch not in '0123456789abcdef' for ch in x)
               for x in (holdout_dataset_sha,reviewer_approval_sha)):
            raise ValueError('holdout/reviewer fingerprint required')
        if isinstance(holdout_start_ns,bool) or not isinstance(holdout_start_ns,int) or holdout_start_ns<1:
            raise ValueError('holdout start UTC nanoseconds required')
        with self._tx() as c:
            try:
                cur=c.execute("UPDATE protocol SET state='HOLDOUT_OPENED',holdout_sha=?,holdout_start_ns=? WHERE id=? AND state='RETRAINED' AND pre_holdout_end_ns IS NOT NULL AND pre_holdout_end_ns<=?",
                              (holdout_dataset_sha,holdout_start_ns,protocol_id,holdout_start_ns))
            except sqlite3.IntegrityError as exc:
                raise ResearchLockError('holdout dataset has already been opened by another protocol') from exc
            if cur.rowcount!=1:raise ResearchLockError('one-shot holdout already opened, protocol not ready, or retrain crosses holdout start')
            self._audit(c,protocol_id,'OPEN_HOLDOUT',{'dataset':holdout_dataset_sha,'approval':reviewer_approval_sha,'holdout_start_ns':holdout_start_ns})
        return 'HOLDOUT_OPENED'

    def close_holdout(self, protocol_id: str, *, report_sha: str, conclusion: str) -> None:
        if conclusion not in {'PASS','FAIL'} or len(report_sha)!=64 or any(c not in '0123456789abcdef' for c in report_sha):
            raise ValueError('invalid final research conclusion')
        with self._tx() as c:
            cur=c.execute("UPDATE protocol SET state='CLOSED' WHERE id=? AND state='HOLDOUT_OPENED'",(protocol_id,))
            if cur.rowcount!=1:raise ResearchLockError('holdout not open or already closed')
            self._audit(c,protocol_id,'CLOSE_HOLDOUT',{'report':report_sha,'conclusion':conclusion})

    def record_holdout_report(self, protocol_id: str, report_sha: str) -> None:
        """Durably bind the one-shot Holdout report before closing its ledger stage."""
        if (not isinstance(report_sha, str) or re.fullmatch(r'[0-9a-f]{64}', report_sha) is None):
            raise ValueError('Holdout report SHA-256 required')
        payload = {'report_sha256': report_sha}
        payload_sha = _sha(_json(payload))
        with self._tx() as c:
            row = c.execute('SELECT state FROM protocol WHERE id=?', (protocol_id,)).fetchone()
            if row is None:
                raise ResearchLockError('unknown protocol')
            existing = c.execute(
                "SELECT payload_sha FROM audit WHERE protocol_id=? AND action='HOLDOUT_REPORT'",
                (protocol_id,),
            ).fetchall()
            if existing:
                if len(existing) == 1 and existing[0][0] == payload_sha:
                    return
                raise ResearchLockError('Holdout report differs from its durable ledger receipt')
            if row[0] != 'HOLDOUT_OPENED':
                raise ResearchLockError('Holdout report can only be recorded while the one-shot stage is open')
            self._audit(c, protocol_id, 'HOLDOUT_REPORT', payload)

    def abort(self, protocol_id: str, *, error_type: str, detail: str) -> None:
        """Terminate an incomplete protocol without consuming unopened holdouts."""
        if (not isinstance(error_type, str) or not error_type or len(error_type) > 128
                or not isinstance(detail, str) or not detail or len(detail) > 2000):
            raise ValueError('bounded failure type and detail required')
        with self._tx() as c:
            cur=c.execute(
                "UPDATE protocol SET state='ABORTED' WHERE id=? AND state IN "
                "('DEVELOPMENT','RECIPE_FROZEN','RETRAINED','HOLDOUT_OPENED')",
                (protocol_id,),
            )
            if cur.rowcount != 1:
                exists=c.execute('SELECT state FROM protocol WHERE id=?',(protocol_id,)).fetchone()
                if exists is None:
                    raise ResearchLockError('unknown protocol')
                raise ResearchLockError('protocol is already terminal and cannot be aborted')
            self._audit(c,protocol_id,'ABORT',{'error_type':error_type,'detail':detail})

    def suspend(self, protocol_id: str, *, detail: str) -> None:
        """Mark a supported WFO stage interruption without refunding queries."""
        if not isinstance(detail, str) or not detail or len(detail) > 2000:
            raise ValueError('bounded interruption detail required')
        with self._tx() as c:
            row=c.execute('SELECT state FROM protocol WHERE id=?',(protocol_id,)).fetchone()
            if row is None:
                raise ResearchLockError('unknown protocol')
            prior_state=row[0]
            if prior_state not in {'DEVELOPMENT','RECIPE_FROZEN','RETRAINED','HOLDOUT_OPENED'}:
                raise ResearchLockError('protocol state is not resumable at this WFO boundary')
            cur=c.execute("UPDATE protocol SET state='SUSPENDED',resume_state=? WHERE id=? AND state=?",
                          (prior_state,protocol_id,prior_state))
            if cur.rowcount != 1:
                raise ResearchLockError('protocol state changed while suspending')
            self._audit(c,protocol_id,'SUSPEND',{'detail':detail,'resume_state':prior_state})

    def resume_suspended(self, protocol_id: str) -> str:
        """Restore a supported WFO stage while preserving query reservations."""
        with self._tx() as c:
            row=c.execute('SELECT resume_state FROM protocol WHERE id=? AND state=?',
                          (protocol_id,'SUSPENDED')).fetchone()
            if row is None or row[0] not in {'DEVELOPMENT','RECIPE_FROZEN','RETRAINED','HOLDOUT_OPENED'}:
                raise ResearchLockError('suspended protocol has no supported resume stage')
            resume_state=row[0]
            cur=c.execute('UPDATE protocol SET state=?,resume_state=NULL WHERE id=? AND state=?',
                          (resume_state,protocol_id,'SUSPENDED'))
            if cur.rowcount != 1:
                raise ResearchLockError('protocol state changed while resuming')
            self._audit(c,protocol_id,'RESUME',{'resume_state':resume_state})
            return resume_state

    def query(self, protocol_id: str, query_id: str) -> dict[str, str] | None:
        """Return a durable research query reservation, if it exists."""
        with self._tx() as c:
            row=c.execute('SELECT query_id,kind,candidate,fold_id FROM query '
                          'WHERE protocol_id=? AND query_id=?',(protocol_id,query_id)).fetchone()
            if row is None:
                return None
            return dict(zip(('query_id','kind','candidate','fold_id'),row))

    def state(self, protocol_id: str) -> dict:
        with self._tx() as c:
            fields=('id','run_id','state','recipe_sha','code_sha','retrain_sha','holdout_sha','selection_spec_sha',
                    'pre_holdout_end_ns','holdout_start_ns','resume_state')
            row=c.execute(f'SELECT {",".join(fields)} FROM protocol WHERE id=?',(protocol_id,)).fetchone()
            if not row:raise ResearchLockError('unknown protocol')
            return dict(zip(fields,row))

    def query_counts(self, protocol_id: str) -> dict[str,int]:
        with self._tx() as c:
            return {kind:n for kind,n in c.execute('SELECT kind,COUNT(*) FROM query WHERE protocol_id=? GROUP BY kind',(protocol_id,))}

    def verify_audit_chain(self) -> bool:
        with self._tx() as c:
            prev='0'*64
            for pid,action,payload,prior,digest in c.execute('SELECT protocol_id,action,payload_sha,prev_hash,row_hash FROM audit ORDER BY seq'):
                if prior!=prev or digest!=_sha(_json([pid,action,payload,prior])):
                    raise ResearchLockError('audit journal chain broken')
                prev=digest
        return True

    def attest_stage_usage(self, protocol_id: str, spec: object, artifact: object) -> str:
        """Persist an L5 stage's bounded usage and return its deterministic receipt.

        This is structural by design: L6 does not import the L5 package. The
        caller must precompute ``artifact.ledger_receipt_hash`` with
        ``SelectionLedger.receipt_hash_for`` so the receipt is part of the
        immutable StageArtifact before the orchestrator accepts it.
        """
        payload, usage = _stage_receipt_payload(protocol_id, spec, artifact)
        receipt = _sha(_json(payload))
        if getattr(artifact, 'ledger_receipt_hash', None) != receipt:
            raise ResearchLockError('stage artifact receipt does not match its frozen usage')
        with self._tx() as c:
            row=c.execute('SELECT run_id,state FROM protocol WHERE id=?',(protocol_id,)).fetchone()
            if row is None or row[0] != payload['run_id']:
                raise ResearchLockError('stage run does not match the configured research protocol')
            if row[1] != 'DEVELOPMENT':
                raise ResearchLockError('stage training is only allowed while protocol is in DEVELOPMENT')
            existing=c.execute('SELECT artifact_id,payload_hash,run_fingerprint,usage_json,receipt_hash FROM stage_receipt WHERE protocol_id=? AND run_id=? AND attempt_id=? AND stage=?',
                               (protocol_id,payload['run_id'],payload['attempt_id'],payload['stage'])).fetchone()
            if existing is not None:
                if existing == (payload['artifact_id'],payload['payload_hash'],payload['run_fingerprint'],_json(usage),receipt):
                    return receipt
                raise ResearchLockError('stage usage already attested for a different artifact')
            reserved=c.execute('SELECT COUNT(*) FROM es_lane_request WHERE protocol_id=? AND run_id=? AND attempt_id=? AND stage=?',
                               (protocol_id,payload['run_id'],payload['attempt_id'],payload['stage'])).fetchone()[0]
            if reserved != usage['lane_requests']:
                raise ResearchLockError('stage lane usage differs from durable pre-dispatch reservations')
            try:
                c.execute('INSERT INTO stage_receipt(protocol_id,run_id,attempt_id,stage,artifact_id,payload_hash,run_fingerprint,usage_json,receipt_hash,recorded) VALUES(?,?,?,?,?,?,?,?,?,?)',
                          (protocol_id,payload['run_id'],payload['attempt_id'],payload['stage'],
                           payload['artifact_id'],payload['payload_hash'],payload['run_fingerprint'],
                           _json(usage),receipt,time.time()))
            except sqlite3.IntegrityError as exc:
                raise ResearchLockError('stage artifact or receipt identity has already been used') from exc
            self._audit(c,protocol_id,'STAGE_USAGE',payload | {'receipt_hash':receipt})
        return receipt

    def reserve_es_lane_requests(self, protocol_id: str, spec: object, *, stage: str,
                                 requests) -> int:
        """Durably charge candidate×episode requests before any ES worker executes.

        Reservations are irreversible within a research protocol, including
        requests whose worker later crashes. A replacement attempt therefore
        consumes fresh protocol budget instead of obtaining free retries.
        """
        if stage not in {'E1', 'E2'}:
            raise ValueError('ES lane reservations require E1 or E2')
        run_id=getattr(spec,'run_id',None);attempt_id=getattr(spec,'attempt_id',None)
        stage_budgets=getattr(spec,'stage_budget',None)
        budget=(stage_budgets.get(stage) if isinstance(stage_budgets,Mapping) else None)
        cap=getattr(budget,'max_lane_requests',None)
        if (not isinstance(run_id,str) or not run_id or not isinstance(attempt_id,str)
                or not attempt_id or type(cap) is not int or cap < 1):
            raise ResearchLockError('frozen ES lane cap and run/attempt identity required')
        batch=tuple(requests)
        if not batch:
            raise ValueError('cannot reserve an empty ES lane batch')
        expected_kind='actor' if stage=='E1' else 'soft_risk'
        entries=[]
        for request in batch:
            key=getattr(request,'key',None)
            request_hash=getattr(request,'request_hash',None)
            if (not isinstance(key,tuple) or len(key)!=6
                    or key[0]!=run_id or key[1]!=attempt_id
                    or type(key[2]) is not int or key[2]<0
                    or not isinstance(key[3],str) or not key[3]
                    or not isinstance(key[4],str) or not key[4]
                    or type(key[5]) is not int or not 0<=key[5]<2**32
                    or not isinstance(request_hash,str) or not re.fullmatch(r'[0-9a-f]{64}',request_hash)
                    or getattr(request,'candidate_kind',None)!=expected_kind
                    or getattr(request,'split',None)!='train_inner'):
                raise ResearchLockError('invalid or mismatched Train-inner ES lane request')
            entries.append((_json(key),request_hash))
        if len({entry[0] for entry in entries})!=len(entries):
            raise ResearchLockError('duplicate logical ES lane request in reservation batch')
        with self._tx() as c:
            row=c.execute('SELECT run_id,state,es_limit FROM protocol WHERE id=?',(protocol_id,)).fetchone()
            if row is None or row[0]!=run_id:
                raise ResearchLockError('ES lane request does not match configured research protocol')
            if row[1]!='DEVELOPMENT':
                raise ResearchLockError('ES lane dispatch is only allowed during DEVELOPMENT')
            attempt_used=c.execute('SELECT COUNT(*) FROM es_lane_request WHERE protocol_id=? AND run_id=? AND attempt_id=? AND stage=?',
                                   (protocol_id,run_id,attempt_id,stage)).fetchone()[0]
            if attempt_used+len(entries)>cap:
                raise ResearchLockError('frozen stage lane request budget exhausted')
            query_used=c.execute("SELECT COUNT(*) FROM query WHERE protocol_id=? AND kind='es_evaluation'",
                                 (protocol_id,)).fetchone()[0]
            total_used=c.execute('SELECT COUNT(*) FROM es_lane_request WHERE protocol_id=?',
                                 (protocol_id,)).fetchone()[0]
            if query_used+total_used+len(entries)>row[2]:
                raise ResearchLockError('frozen research ES evaluation budget exhausted')
            try:
                c.executemany('INSERT INTO es_lane_request(protocol_id,run_id,attempt_id,stage,request_key,request_hash,recorded) VALUES(?,?,?,?,?,?,?)',
                              ((protocol_id,run_id,attempt_id,stage,key,request_hash,time.time())
                               for key,request_hash in entries))
            except sqlite3.IntegrityError as exc:
                raise ResearchLockError('ES lane request was already reserved') from exc
            self._audit(c,protocol_id,'ES_LANE_RESERVE',{
                'run_id':run_id,'attempt_id':attempt_id,'stage':stage,
                'count':len(entries),'request_manifest_sha256':_sha(_json(entries)),
            })
            return attempt_used+len(entries)

    def es_lane_request_count(self, protocol_id: str, run_id: str, attempt_id: str,
                              stage: str) -> int:
        if stage not in {'E1','E2'}:
            raise ValueError('ES lane request count requires E1 or E2')
        with self._tx() as c:
            row=c.execute('SELECT COUNT(*) FROM es_lane_request WHERE protocol_id=? AND run_id=? AND attempt_id=? AND stage=?',
                          (protocol_id,run_id,attempt_id,stage)).fetchone()
            return int(row[0])

    def verify_es_lane_reservations(self, protocol_id: str, run_id: str,
                                    attempt_id: str, stage: str, request_keys) -> None:
        """Require a resumed L5 checkpoint to match all durable reservations exactly."""
        if (stage not in {'E1','E2'} or not isinstance(run_id,str) or not run_id
                or not isinstance(attempt_id,str) or not attempt_id):
            raise ValueError('ES lane reservation verification requires E1 or E2')
        keys=tuple(request_keys)
        if any(not isinstance(key,tuple) or len(key)!=6
               or key[0]!=run_id or key[1]!=attempt_id
               or type(key[2]) is not int or key[2]<0
               or not isinstance(key[3],str) or not key[3]
               or not isinstance(key[4],str) or not key[4]
               or type(key[5]) is not int or not 0<=key[5]<2**32 for key in keys):
            raise ResearchLockError('malformed logical request key in ES resume checkpoint')
        expected=tuple(_json(key) for key in keys)
        if len(expected)!=len(set(expected)):
            raise ResearchLockError('duplicate logical request in ES resume checkpoint')
        with self._tx() as c:
            rows=c.execute('SELECT request_key FROM es_lane_request WHERE protocol_id=? AND run_id=? AND attempt_id=? AND stage=?',
                           (protocol_id,run_id,attempt_id,stage)).fetchall()
            reserved={row[0] for row in rows}
            if reserved!=set(expected):
                raise ResearchLockError('ES resume checkpoint differs from durable lane reservations')


    def evidence_summary(self, protocol_id: str) -> dict[str, object]:
        """Read-only audit inventory for statistical trial-count review.

        Counts are RAW queries, NOT an estimate of effective independent trials.
        Deriving effective_trials (DSR) needs reviewer approval and ICR-L6-008.
        """
        with self._tx() as c:
            fields=('run_id','state','validation_limit','hpo_limit','es_limit',
                    'recipe_sha','code_sha','selection_spec_sha','retrain_sha',
                    'pre_holdout_end_ns','holdout_start_ns','holdout_sha')
            row=c.execute(f'SELECT {",".join(fields)} FROM protocol WHERE id=?',(protocol_id,)).fetchone()
            if row is None:
                raise ResearchLockError('unknown protocol')
            queries=c.execute('SELECT query_id,kind,candidate,fold_id FROM query WHERE protocol_id=? ORDER BY query_id',
                              (protocol_id,)).fetchall()
            lane_requests=c.execute('SELECT run_id,attempt_id,stage,request_key,request_hash FROM es_lane_request WHERE protocol_id=? ORDER BY run_id,attempt_id,stage,request_key',
                                    (protocol_id,)).fetchall()
            stages=c.execute('SELECT run_id,attempt_id,stage,artifact_id,payload_hash,run_fingerprint,usage_json,receipt_hash FROM stage_receipt WHERE protocol_id=? ORDER BY run_id,attempt_id,stage',
                             (protocol_id,)).fetchall()
            counters={kind:sum(q[1]==kind for q in queries) for kind in self.VALID_QUERIES}
            audit=c.execute('SELECT seq,row_hash FROM audit WHERE protocol_id=? ORDER BY seq DESC LIMIT 1',
                            (protocol_id,)).fetchone()
            payload={'protocol_id':protocol_id,'protocol':dict(zip(fields,row)),
                     'queries':[dict(zip(('query_id','kind','candidate','fold_id'),q)) for q in queries],
                     'stage_receipts':[dict(zip(('run_id','attempt_id','stage','artifact_id','payload_hash','run_fingerprint','usage_json','receipt_hash'),r)) for r in stages],
                     'es_lane_requests':[dict(zip(('run_id','attempt_id','stage','request_key','request_hash'),r)) for r in lane_requests],
                     'raw_query_counts':counters,
                     'es_lane_request_count':len(lane_requests),
                     'trial_count_inventory':{
                         'scope':'current_protocol_only',
                         'observed_candidate_ids':sorted({q[2] for q in queries if q[2]}),
                         'observed_fold_ids':sorted({q[3] for q in queries if q[3]}),
                         'raw_query_counts':counters,
                         'es_lane_request_count':len(lane_requests),
                         'stage_receipt_count':len(stages),
                         'project_historical_effective_trials':'not_estimated',
                         'limitations':[
                             'protocol queries do not record every unqueried or failed candidate attempt',
                             'candidate IDs are not canonical parameter-level deduplication keys',
                             'seeds and feature versions are not globally reconciled across protocols',
                             'manual selections and unledgered historical experiments are not inventoried',
                         ],
                     },
                     'protocol_last_audit_seq':audit[0] if audit else None,
                     'protocol_last_audit_hash':audit[1] if audit else None,
                     'effective_trials':'REQUIRES_EXPLICIT_APPROVED_STATISTICAL_POLICY'}
            payload['summary_sha256']=_sha(_json(payload))
            return payload


def _stage_receipt_payload(protocol_id: str, spec: object, artifact: object) -> tuple[dict[str, object], dict[str, int]]:
    if (not isinstance(protocol_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',protocol_id)
            or protocol_id in {'.','..'}):
        raise ValueError('invalid research protocol ID')
    run_id, attempt_id = getattr(spec,'run_id',None), getattr(spec,'attempt_id',None)
    stage, artifact_id = getattr(artifact,'stage',None), getattr(artifact,'artifact_id',None)
    run_fingerprint = getattr(spec,'fingerprint',None)
    if (not isinstance(run_id,str) or not run_id or not isinstance(attempt_id,str) or not attempt_id
            or not isinstance(artifact_id,str) or not artifact_id or stage not in {'E1','R1','E2'}
            or not isinstance(run_fingerprint,str) or not re.fullmatch(r'[0-9a-f]{64}',run_fingerprint)
            or getattr(artifact,'run_fingerprint',None) != run_fingerprint):
        raise ResearchLockError('invalid or mismatched stage run identity')
    payload_hash = getattr(artifact,'payload_hash',None)
    if not isinstance(payload_hash,str) or not re.fullmatch(r'[0-9a-f]{64}',payload_hash):
        raise ResearchLockError('invalid stage artifact payload fingerprint')
    budgets = getattr(spec,'stage_budget',None)
    if not isinstance(budgets,Mapping):
        raise ResearchLockError('run has no frozen stage budgets')
    try:
        budget = budgets[stage]
        usage_obj = getattr(artifact,'usage')
    except (KeyError, AttributeError, TypeError) as exc:
        raise ResearchLockError('stage or usage missing from frozen run') from exc
    names=('max_population','max_generations','max_evaluations','max_rl_steps',
           'max_validation_queries','max_lane_requests')
    usage_names=('population','generations','evaluations','rl_steps',
                 'validation_queries','lane_requests')
    caps_raw={name:getattr(budget,name,None) for name in names}
    usage_raw={name:getattr(usage_obj,name,None) for name in usage_names}
    if (any(type(value) is not int or value < 0 for value in (*caps_raw.values(),*usage_raw.values()))):
        raise ResearchLockError('stage usage exceeds or invalidates its frozen budget')
    caps=cast(dict[str,int],caps_raw)
    usage=cast(dict[str,int],usage_raw)
    if (caps['max_validation_queries'] != 0
            or usage['validation_queries'] != 0
            or any(usage[key] > caps[cap] for key,cap in zip(usage_names,names))):
        raise ResearchLockError('stage usage exceeds or invalidates its frozen budget')
    if stage in {'E1','E2'}:
        if (caps['max_population'] < 4 or caps['max_generations'] < 1
                or caps['max_evaluations'] < caps['max_population']*caps['max_generations']
                or caps['max_rl_steps'] != 0
                or usage['population'] < 4 or usage['generations'] < 1
                or usage['evaluations'] < usage['population']*usage['generations']
                or usage['rl_steps'] != 0):
            raise ResearchLockError('ES stage usage is incomplete or inconsistent')
        if ((caps['max_lane_requests'] > 0) != (usage['lane_requests'] > 0)
                or (usage['lane_requests'] > 0
                    and usage['lane_requests'] < usage['evaluations'])):
            raise ResearchLockError('ES lane request usage is incomplete or inconsistent')
    elif (caps['max_rl_steps'] < 1
            or any(caps[key] for key in ('max_population','max_generations','max_evaluations',
                                         'max_lane_requests'))
            or usage['rl_steps'] < 1
            or any(usage[key] for key in ('population','generations','evaluations','lane_requests'))):
        raise ResearchLockError('RL stage usage is incomplete or contains ES counters')
    payload={
        'protocol_id':protocol_id,'run_id':run_id,'attempt_id':attempt_id,'stage':stage,
        'artifact_id':artifact_id,'payload_hash':payload_hash,'run_fingerprint':run_fingerprint,
        'budget':caps,'usage':usage,
    }
    return payload, usage


class SelectionLedger:
    """L6 durable StageUsage attestor compatible with L5 ``UsageAttestor``.

    Create a receipt hash before constructing the immutable StageArtifact with
    ``receipt_hash_for``; pass this callable directly to L5 Orchestrator.
    The ledger does not import L5 types and independently checks frozen caps.
    """

    def __init__(self, ledger: ResearchLedger, protocol_id: str):
        if not isinstance(ledger,ResearchLedger):
            raise TypeError('ResearchLedger required')
        self.ledger=ledger
        self.protocol_id=protocol_id

    def receipt_hash_for(self, spec: object, *, stage: str, artifact_id: str,
                         payload_hash: str, usage: object) -> str:
        if stage in {'E1','E2'}:
            run_id=getattr(spec,'run_id',None);attempt_id=getattr(spec,'attempt_id',None)
            if not isinstance(run_id,str) or not isinstance(attempt_id,str):
                raise ResearchLockError('stage lane receipt requires run/attempt identity')
            reserved=self.ledger.es_lane_request_count(
                self.protocol_id,run_id,attempt_id,stage,
            )
            if reserved!=getattr(usage,'lane_requests',None):
                raise ResearchLockError('stage lane usage differs from durable pre-dispatch reservations')
        artifact=SimpleNamespace(stage=stage,artifact_id=artifact_id,payload_hash=payload_hash,
                                 run_fingerprint=getattr(spec,'fingerprint',None),usage=usage)
        payload,_=_stage_receipt_payload(self.protocol_id,spec,artifact)
        return _sha(_json(payload))

    def reserve_lane_requests(self, spec: object, *, stage: str, requests) -> int:
        """Callback for L5 EvaluationBudget; persists charges before dispatch."""
        return self.ledger.reserve_es_lane_requests(
            self.protocol_id,spec,stage=stage,requests=requests,
        )

    def verify_lane_requests(self, spec: object, *, stage: str, request_keys) -> None:
        """Callback for L5 checkpoint restore; no evaluation resumes on ledger drift."""
        run_id=getattr(spec,'run_id',None);attempt_id=getattr(spec,'attempt_id',None)
        if not isinstance(run_id,str) or not isinstance(attempt_id,str):
            raise ResearchLockError('ES resume verification requires run/attempt identity')
        self.ledger.verify_es_lane_reservations(
            self.protocol_id,run_id,attempt_id,stage,request_keys,
        )

    def reservation_scope(self, spec: object, *, stage: str) -> str:
        """Stable checkpoint identity for this protocol/run/attempt/stage budget."""
        if stage not in {'E1','E2'}:
            raise ValueError('ES reservation scope requires E1 or E2')
        run_id=getattr(spec,'run_id',None);attempt_id=getattr(spec,'attempt_id',None)
        if not isinstance(run_id,str) or not isinstance(attempt_id,str):
            raise ResearchLockError('ES reservation scope requires run/attempt identity')
        return _sha(_json((self.protocol_id,run_id,attempt_id,stage)))

    def __call__(self, spec: object, artifact: object) -> str:
        return self.ledger.attest_stage_usage(self.protocol_id,spec,artifact)
