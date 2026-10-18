"""Project-wide interval accounting, independent of output paths and data versions.

Reservations are charged before evaluation; a crash never refunds a read. This
is local research governance, not a security boundary against database deletion.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from .ledger import ResearchLedger, ResearchLockError


_TRIAL_EVENT_TRANSITIONS = {
    "registered": {"started", "resumed"},
    "started": {"completed", "selected", "not_selected", "failed", "interrupted"},
    "resumed": {"completed", "selected", "not_selected", "failed", "interrupted"},
    "failed": {"resumed"},
    "interrupted": {"resumed"},
    "completed": set(),
    "selected": set(),
    "not_selected": set(),
}
_STANDALONE_TRIAL_PROTOCOL = "standalone-application-training-v1"
_WFO_TRIAL_EVENT_TRANSITIONS = {
    **_TRIAL_EVENT_TRANSITIONS,
    "started": _TRIAL_EVENT_TRANSITIONS["started"] | {"started"},
    "resumed": _TRIAL_EVENT_TRANSITIONS["resumed"] | {"resumed"},
    "interrupted": {"resumed", "started"},
    "selected": {"resumed"},
    "not_selected": {"resumed"},
}


def _transition_sql(transitions: dict[str, set[str]]) -> str:
    clauses = []
    for previous, next_statuses in transitions.items():
        if not next_statuses:
            continue
        values = ",".join(f"'{status}'" for status in sorted(next_statuses))
        clauses.append(f"(previous.status='{previous}' AND NEW.status IN ({values}))")
    return " OR ".join(clauses)


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class ProjectResearchRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tx() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS access (
                access_key TEXT PRIMARY KEY, owner TEXT NOT NULL, market TEXT NOT NULL,
                family TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL,
                purpose TEXT NOT NULL, identity TEXT NOT NULL, token TEXT NOT NULL,
                state TEXT NOT NULL, report_sha TEXT)""")
            c.execute("CREATE INDEX IF NOT EXISTS interval_lookup ON access(market,family,start_ns,end_ns)")
            c.execute("""CREATE TABLE IF NOT EXISTS query_intent (
                protocol_id TEXT NOT NULL, intent_id TEXT NOT NULL, owner TEXT NOT NULL,
                market TEXT NOT NULL, family TEXT NOT NULL, start_ns INTEGER NOT NULL,
                end_ns INTEGER NOT NULL, purpose TEXT NOT NULL, identity TEXT NOT NULL,
                state TEXT NOT NULL, PRIMARY KEY(protocol_id,intent_id))""")
            c.execute("CREATE INDEX IF NOT EXISTS query_intent_interval_lookup "
                      "ON query_intent(market,family,start_ns,end_ns)")
            c.execute("""CREATE TRIGGER IF NOT EXISTS query_intent_identity_immutable
                BEFORE UPDATE OF protocol_id,intent_id,owner,market,family,start_ns,end_ns,purpose,identity
                ON query_intent BEGIN
                SELECT RAISE(ABORT,'project query intent identity is immutable'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS query_intent_state_transition_guard
                BEFORE UPDATE ON query_intent
                WHEN NOT (NEW.state=OLD.state OR (OLD.state='reserved' AND NEW.state='started')) BEGIN
                SELECT RAISE(ABORT,'invalid project query intent state transition'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS query_intent_no_delete
                BEFORE DELETE ON query_intent BEGIN
                SELECT RAISE(ABORT,'project query intents are append-only'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS access_identity_immutable
                BEFORE UPDATE OF access_key,owner,market,family,start_ns,end_ns,purpose,identity,token
                ON access BEGIN
                SELECT RAISE(ABORT,'project access identity is immutable'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS access_state_transition_guard
                BEFORE UPDATE ON access
                WHEN NOT (
                    (NEW.state=OLD.state AND NEW.report_sha IS OLD.report_sha)
                    OR (OLD.state='reserved' AND NEW.state='started'
                     AND NEW.report_sha IS OLD.report_sha)
                    OR (OLD.state='started' AND NEW.state='completed'
                        AND NEW.report_sha IS NOT NULL AND length(NEW.report_sha)=64
                        AND NEW.report_sha NOT GLOB '*[^0-9a-f]*')
                ) BEGIN
                SELECT RAISE(ABORT,'invalid project access state transition'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS access_no_delete
                BEFORE DELETE ON access BEGIN
                SELECT RAISE(ABORT,'project access records are append-only'); END""")
            c.execute("CREATE TABLE IF NOT EXISTS inventory_import (sha TEXT PRIMARY KEY)")
            c.execute("""CREATE TABLE IF NOT EXISTS trial_attempt (
                attempt_key TEXT PRIMARY KEY, protocol_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL, candidate_sha TEXT NOT NULL,
                feature_sha TEXT NOT NULL, seed INTEGER NOT NULL,
                recipe_json TEXT NOT NULL, recorded REAL NOT NULL)""")
            c.execute("CREATE INDEX IF NOT EXISTS trial_protocol_lookup ON trial_attempt(protocol_id)")
            c.execute("""CREATE TABLE IF NOT EXISTS trial_attempt_event (
                attempt_key TEXT NOT NULL, event_key TEXT NOT NULL,
                status TEXT NOT NULL, detail_json TEXT NOT NULL, recorded REAL NOT NULL,
                PRIMARY KEY(attempt_key,event_key),
                FOREIGN KEY(attempt_key) REFERENCES trial_attempt(attempt_key))""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS trial_attempt_no_update
                BEFORE UPDATE ON trial_attempt BEGIN
                SELECT RAISE(ABORT,'trial attempts are append-only'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS trial_attempt_no_delete
                BEFORE DELETE ON trial_attempt BEGIN
                SELECT RAISE(ABORT,'trial attempts are append-only'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS trial_event_no_update
                BEFORE UPDATE ON trial_attempt_event BEGIN
                SELECT RAISE(ABORT,'trial events are append-only'); END""")
            c.execute("""CREATE TRIGGER IF NOT EXISTS trial_event_no_delete
                BEFORE DELETE ON trial_attempt_event BEGIN
                SELECT RAISE(ABORT,'trial events are append-only'); END""")
            c.execute(f"""CREATE TRIGGER IF NOT EXISTS trial_event_transition_guard
                BEFORE INSERT ON trial_attempt_event
                WHEN
                    (NOT EXISTS (SELECT 1 FROM trial_attempt
                                 WHERE attempt_key=NEW.attempt_key)
                     OR (NOT EXISTS (SELECT 1 FROM trial_attempt_event
                                 WHERE attempt_key=NEW.attempt_key)
                         AND NEW.status<>'registered'))
                    OR
                    (EXISTS (SELECT 1 FROM trial_attempt_event
                             WHERE attempt_key=NEW.attempt_key)
                     AND NOT (
                        ((SELECT protocol_id FROM trial_attempt
                          WHERE attempt_key=NEW.attempt_key)='{_STANDALONE_TRIAL_PROTOCOL}'
                         AND EXISTS (
                            SELECT 1 FROM trial_attempt_event AS previous
                            WHERE previous.attempt_key=NEW.attempt_key
                              AND previous.rowid=(SELECT MAX(rowid) FROM trial_attempt_event
                                                  WHERE attempt_key=NEW.attempt_key)
                              AND ({_transition_sql(_TRIAL_EVENT_TRANSITIONS)})
                         ))
                        OR
                        ((SELECT protocol_id FROM trial_attempt
                          WHERE attempt_key=NEW.attempt_key)<>'{_STANDALONE_TRIAL_PROTOCOL}'
                         AND EXISTS (
                            SELECT 1 FROM trial_attempt_event AS previous
                            WHERE previous.attempt_key=NEW.attempt_key
                              AND previous.rowid=(SELECT MAX(rowid) FROM trial_attempt_event
                                                  WHERE attempt_key=NEW.attempt_key)
                              AND ({_transition_sql(_WFO_TRIAL_EVENT_TRANSITIONS)})
                         ))
                     ))
                BEGIN
                    SELECT RAISE(ABORT,'invalid project trial lifecycle transition');
                END""")

    def import_inventory(self, entries: list[dict], inventory_sha: str) -> None:
        with self._tx() as c:
            if c.execute("SELECT 1 FROM inventory_import WHERE sha=?", (inventory_sha,)).fetchone():
                return
            # Known historical accesses can already overlap. Import records
            # that contamination, it does not grant permission for new reads.
            # Keep the entries and completion marker in one transaction.
            for entry in entries:
                expected = self._scope(owner="legacy-inventory", **entry)
                self._reserve_record(c, entry["key"], expected, historical=True)
            c.execute("INSERT OR IGNORE INTO inventory_import VALUES(?)", (inventory_sha,))

    @contextmanager
    def _tx(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
            c.execute("COMMIT")
        except BaseException:
            if c.in_transaction:
                c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    @staticmethod
    def _scope(*, key: str, owner: str, market: str, family: str,
               start_ns: int, end_ns: int, purpose: str, identity: dict) -> tuple:
        if (any(not isinstance(v, str) or not v for v in (key, owner, market, family))
                or type(start_ns) is not int or type(end_ns) is not int
                or not 0 < start_ns < end_ns or not isinstance(purpose, str)
                or purpose not in {"development", "test", "holdout"}
                or not isinstance(identity, dict)
                or any(not isinstance(name, str) for name in identity)):
            raise ValueError("valid research scope and half-open UTC ns interval required")
        try:
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("research scope identity must be finite JSON data") from exc
        return (owner, market, family, start_ns, end_ns, purpose, encoded)

    def reserve(self, *, key: str, owner: str, market: str, family: str,
                start_ns: int, end_ns: int, purpose: str, identity: dict) -> str:
        expected = self._scope(key=key, owner=owner, market=market, family=family,
                               start_ns=start_ns, end_ns=end_ns, purpose=purpose, identity=identity)
        with self._tx() as c:
            return self._reserve_record(c, key, expected)

    def reserve_query_intent(self, *, protocol_id: str, intent_id: str, owner: str,
                             market: str, family: str, start_ns: int, end_ns: int,
                             purpose: str, identity: dict) -> None:
        """Predeclare a formal WFO data scope before the source payload is decoded.

        An intent reserves the protocol query identity and its immutable source
        scope. It does not open the evaluation; the normal ResearchLedger and
        EvaluationAccess gates still control when a Test/Holdout is consumed.
        """
        expected = self._scope(
            key=intent_id, owner=owner, market=market, family=family,
            start_ns=start_ns, end_ns=end_ns, purpose=purpose, identity=identity,
        )
        if (purpose not in {"test", "holdout"}
                or not isinstance(protocol_id, str) or not protocol_id
                or not isinstance(identity, dict)
                or identity.get("protocol_id") != protocol_id
                or identity.get("query_id") != intent_id
                or not isinstance(identity.get("source_sha256"), str)
                or len(identity["source_sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in identity["source_sha256"])
                or not isinstance(identity.get("candidate_ids"), list)
                or not identity["candidate_ids"]
                or any(not isinstance(value, str) or not value
                       for value in identity["candidate_ids"])):
            raise ResearchLockError("invalid pre-read formal query intent")
        with self._tx() as c:
            row = c.execute(
                "SELECT owner,market,family,start_ns,end_ns,purpose,identity "
                "FROM query_intent WHERE protocol_id=? AND intent_id=?",
                (protocol_id, intent_id),
            ).fetchone()
            expected_row = (*expected,)
            if row is not None:
                if row != expected_row:
                    raise ResearchLockError("project query intent identity changed")
                return
            formal_rows = c.execute(
                "SELECT owner,market,family,start_ns,end_ns,purpose,identity "
                "FROM access WHERE purpose IN ('test','holdout') "
                "AND market=? AND family=? AND start_ns<? AND end_ns>?",
                (market, family, end_ns, start_ns),
            ).fetchall()
            for (previous_owner, _m, _f, _s, _e, _purpose, previous_identity) in formal_rows:
                try:
                    prior = json.loads(previous_identity)
                except (TypeError, json.JSONDecodeError):
                    prior = {}
                if (previous_owner != owner or prior.get("protocol_id") != protocol_id):
                    raise ResearchLockError("project formal interval already reserved/used")
            intent_rows = c.execute(
                "SELECT protocol_id,intent_id,owner,start_ns,end_ns FROM query_intent "
                "WHERE market=? AND family=? AND start_ns<? AND end_ns>?",
                (market, family, end_ns, start_ns),
            ).fetchall()
            for previous_protocol, previous_id, previous_owner, previous_start, previous_end in intent_rows:
                same_reservation = (previous_protocol == protocol_id
                                    and previous_id == intent_id and previous_owner == owner
                                    and previous_start == start_ns and previous_end == end_ns)
                if not same_reservation:
                    raise ResearchLockError("project formal query intent interval already reserved")
            development_rows = c.execute(
                "SELECT owner FROM access WHERE purpose='development' "
                "AND market=? AND family=? AND start_ns<? AND end_ns>?",
                (market, family, end_ns, start_ns),
            ).fetchall()
            if any(previous_owner != owner for (previous_owner,) in development_rows):
                raise ResearchLockError("project data interval already reserved for development")
            c.execute(
                "INSERT INTO query_intent VALUES(?,?,?,?,?,?,?,?,?,'reserved')",
                (protocol_id, intent_id, *expected),
            )

    def start_query_intents(self, *, protocol_id: str, owner: str,
                            intent_ids: Sequence[str]) -> None:
        """Mark all predeclared scopes started atomically immediately before reading."""
        if (not isinstance(protocol_id, str) or not protocol_id
                or not isinstance(owner, str) or not owner
                or not intent_ids or any(not isinstance(value, str) or not value
                                          for value in intent_ids)
                or len(set(intent_ids)) != len(intent_ids)):
            raise ResearchLockError("invalid project query intent start batch")
        with self._tx() as c:
            for intent_id in intent_ids:
                changed = c.execute(
                    "UPDATE query_intent SET state='started' "
                    "WHERE protocol_id=? AND intent_id=? AND owner=? "
                    "AND state IN ('reserved','started')",
                    (protocol_id, intent_id, owner),
                ).rowcount
                if changed != 1:
                    raise ResearchLockError("formal source read lacks its predeclared query intent")

    def verify_query_intent(self, *, protocol_id: str, intent_id: str, owner: str,
                            market: str, family: str, start_ns: int, end_ns: int,
                            purpose: str, source_sha256: str, candidate_id: str) -> None:
        """Bind a later evaluation to a started, exact pre-read query intent."""
        with self._tx() as c:
            row = c.execute(
                "SELECT owner,market,family,start_ns,end_ns,purpose,identity,state "
                "FROM query_intent WHERE protocol_id=? AND intent_id=?",
                (protocol_id, intent_id),
            ).fetchone()
        if row is None:
            raise ResearchLockError("formal evaluation lacks a pre-read query intent")
        expected = (owner, market, family, start_ns, end_ns, purpose)
        if row[:6] != expected or row[7] != "started":
            raise ResearchLockError("formal evaluation differs from its pre-read query intent")
        identity = json.loads(row[6])
        if (identity.get("source_sha256") != source_sha256
                or candidate_id not in identity.get("candidate_ids", [])):
            raise ResearchLockError("formal evaluation source/candidate differs from its pre-read intent")

    @staticmethod
    def _reserve_record(c, key: str, expected: tuple, *, historical: bool = False) -> str:
        owner, market, family, start_ns, end_ns, purpose, encoded = expected
        previous = c.execute("SELECT owner,market,family,start_ns,end_ns,purpose,identity,token "
                             "FROM access WHERE access_key=?", (key,)).fetchone()
        if previous is not None:
            if previous[:-1] != expected:
                raise ResearchLockError("project reservation identity changed")
            return previous[-1]
        if not historical:
            # Query IDs are frozen ledger identities, not per-output labels.
            # A copied WFO ledger or a new output directory must not reuse one
            # query ID for a disjoint formal window. The only exception is the
            # paired policy/baseline component for the exact same Test scope.
            if purpose in {"test", "holdout"}:
                current_identity = json.loads(encoded)
                protocol_id = current_identity.get("protocol_id")
                query_id = current_identity.get("query_id")
                if (isinstance(protocol_id, str) and protocol_id
                        and isinstance(query_id, str) and query_id):
                    formal_rows = c.execute(
                        "SELECT owner,market,family,start_ns,end_ns,purpose,identity "
                        "FROM access WHERE purpose IN ('test','holdout')",
                    ).fetchall()
                    for (previous_owner, previous_market, previous_family,
                         previous_start, previous_end, previous_purpose,
                         previous_identity) in formal_rows:
                        try:
                            prior_identity = json.loads(previous_identity)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if (prior_identity.get("protocol_id") != protocol_id
                                or prior_identity.get("query_id") != query_id):
                            continue
                        if (market == previous_market and family == previous_family
                                and purpose == previous_purpose == "test"
                                and _same_test_query_component(
                                    owner, purpose, start_ns, end_ns, encoded,
                                    previous_owner, previous_purpose, previous_start,
                                    previous_end, previous_identity,
                                )):
                            continue
                        raise ResearchLockError(
                            f"project query identity already reserved: {query_id}"
                        )
            # Once any formal interval has been read/reserved, it cannot be
            # relabelled as development even by the same owner. The sole
            # exception is policy/baseline access for the same frozen Test
            # query; component identity differs, but the data read is one
            # protocol query. Holdout never has a second-key exception.
            conflicts = c.execute(
                "SELECT access_key,owner,purpose,start_ns,end_ns,identity FROM access "
                "WHERE market=? AND family=? AND start_ns<? AND end_ns>?",
                (market, family, end_ns, start_ns),
            ).fetchall()
            for previous_key, previous_owner, previous_purpose, previous_start, previous_end, previous_identity in conflicts:
                if purpose not in {"test", "holdout"} and previous_purpose not in {"test", "holdout"}:
                    continue
                # WFO declares its pre-Holdout development window before it
                # separately authorizes frozen fold Test queries. This only
                # permits development -> Test for the same run owner; it does
                # not permit a formal Test interval to be relabelled later.
                if (previous_purpose == "development" and purpose == "test"
                        and owner == previous_owner):
                    continue
                if _same_test_query_component(
                    owner, purpose, start_ns, end_ns, encoded,
                    previous_owner, previous_purpose, previous_start, previous_end,
                    previous_identity,
                ):
                    continue
                raise ResearchLockError(f"project data interval already reserved/used: {previous_key}")
            intents = c.execute(
                "SELECT owner,purpose,identity FROM query_intent "
                "WHERE market=? AND family=? AND start_ns<? AND end_ns>?",
                (market, family, end_ns, start_ns),
            ).fetchall()
            for previous_owner, previous_purpose, previous_identity in intents:
                try:
                    prior = json.loads(previous_identity)
                except (TypeError, json.JSONDecodeError):
                    prior = {}
                if purpose not in {"test", "holdout"}:
                    if previous_owner != owner:
                        raise ResearchLockError("project formal query intent blocks development access")
                    continue
                current_identity = json.loads(encoded)
                if (previous_owner != owner or previous_purpose != purpose
                        or prior.get("protocol_id") != current_identity.get("protocol_id")):
                    raise ResearchLockError("project formal query intent blocks evaluation access")
        token = uuid.uuid4().hex
        c.execute("INSERT INTO access VALUES(?,?,?,?,?,?,?,?,?,?,NULL)",
                  (key, *expected, token,
                   "recorded" if historical or purpose == "development" else "reserved"))
        return token

    def consume(self, key: str, token: str, identity: dict) -> None:
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._tx() as c:
            changed = c.execute("UPDATE access SET state='started' WHERE access_key=? "
                                "AND token=? AND identity=? AND state='reserved'",
                                (key, token, encoded)).rowcount
            if changed != 1:
                raise ResearchLockError("evaluation already started; only a completed report may be reused")

    def complete(self, key: str, token: str, report_sha: str) -> None:
        if len(report_sha) != 64 or any(ch not in "0123456789abcdef" for ch in report_sha):
            raise ValueError("report SHA256 required")
        with self._tx() as c:
            changed = c.execute("UPDATE access SET state='completed',report_sha=? "
                                "WHERE access_key=? AND token=? AND state='started'",
                                (report_sha, key, token)).rowcount
            if changed != 1:
                raise ResearchLockError("evaluation completion lacks a started reservation")

    def verify_report(self, key: str, report_sha: str) -> None:
        with self._tx() as c:
            row = c.execute("SELECT state,report_sha FROM access WHERE access_key=?", (key,)).fetchone()
            if row != ("completed", report_sha):
                raise ResearchLockError("report differs from its project completion receipt")

    def register_trial_attempt(self, *, attempt_key: str, protocol_id: str,
                               candidate_id: str, recipe: dict,
                               feature_sha: str, seed: int) -> None:
        """Append a frozen candidate/seed/feature identity to the project ledger."""
        if (any(not isinstance(value, str) or not value
                for value in (attempt_key, protocol_id, candidate_id))
                or not isinstance(recipe, dict) or type(seed) is not int or not 0 <= seed < 2**32
                or not isinstance(feature_sha, str) or len(feature_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in feature_sha)):
            raise ValueError("invalid project trial attempt identity")
        try:
            recipe_json = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
            candidate_sha = _hash(recipe)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("project trial attempt recipe must be finite JSON data") from exc
        expected = (protocol_id, candidate_id, candidate_sha, feature_sha, seed, recipe_json)
        with self._tx() as c:
            previous = c.execute(
                "SELECT protocol_id,candidate_id,candidate_sha,feature_sha,seed,recipe_json "
                "FROM trial_attempt WHERE attempt_key=?", (attempt_key,),
            ).fetchone()
            if previous is not None:
                if previous != expected:
                    raise ResearchLockError("project trial attempt identity changed")
            else:
                c.execute(
                    "INSERT INTO trial_attempt VALUES(?,?,?,?,?,?,?,strftime('%s','now'))",
                    (attempt_key, *expected),
                )
            self._append_trial_event(c, attempt_key, "registered", "registered", {})

    def record_trial_attempt_event(self, attempt_key: str, *, event_key: str,
                                   status: str, detail: dict | None = None) -> None:
        """Append an idempotent lifecycle event; prior attempt evidence is immutable."""
        allowed = {"started", "resumed", "completed", "selected", "not_selected",
                   "failed", "interrupted"}
        if (not isinstance(attempt_key, str) or not attempt_key
                or not isinstance(event_key, str) or not event_key
                or not isinstance(status, str) or status not in allowed
                or (detail is not None and not isinstance(detail, dict))):
            raise ValueError("invalid project trial attempt event")
        with self._tx() as c:
            if c.execute("SELECT 1 FROM trial_attempt WHERE attempt_key=?", (attempt_key,)).fetchone() is None:
                raise ResearchLockError("project trial attempt is not registered")
            self._append_trial_event(c, attempt_key, event_key, status, detail or {})

    def trial_attempt_events(self, attempt_key: str) -> list[dict]:
        """Read ordered lifecycle evidence for one immutable registered attempt."""
        if not isinstance(attempt_key, str) or not attempt_key:
            raise ValueError("trial attempt key is required")
        with self._tx() as c:
            if c.execute("SELECT 1 FROM trial_attempt WHERE attempt_key=?", (attempt_key,)).fetchone() is None:
                raise ResearchLockError("project trial attempt is not registered")
            rows = c.execute(
                "SELECT event_key,status,detail_json FROM trial_attempt_event "
                "WHERE attempt_key=? ORDER BY rowid", (attempt_key,),
            ).fetchall()
        return [{"event_key": row[0], "status": row[1], "detail": json.loads(row[2])}
                for row in rows]

    @staticmethod
    def _append_trial_event(c, attempt_key: str, event_key: str,
                            status: str, detail: dict) -> None:
        try:
            detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("project trial event detail must be finite JSON data") from exc
        previous = c.execute(
            "SELECT status,detail_json FROM trial_attempt_event WHERE attempt_key=? AND event_key=?",
            (attempt_key, event_key),
        ).fetchone()
        if previous is not None:
            if previous != (status, detail_json):
                raise ResearchLockError("project trial event identity changed")
            return
        last = c.execute(
            "SELECT status FROM trial_attempt_event WHERE attempt_key=? ORDER BY rowid DESC LIMIT 1",
            (attempt_key,),
        ).fetchone()
        protocol = c.execute(
            "SELECT protocol_id FROM trial_attempt WHERE attempt_key=?", (attempt_key,),
        ).fetchone()
        # WFO selection is a phase result and remains resumable until the
        # complete protocol finishes; standalone selection is terminal.
        transitions = (_TRIAL_EVENT_TRANSITIONS if protocol is None
                       or protocol[0] == _STANDALONE_TRIAL_PROTOCOL
                       else _WFO_TRIAL_EVENT_TRANSITIONS)
        if ((last is None and status != "registered")
                or (last is not None
                    and status not in transitions.get(last[0], set()))):
            prior_status = "missing" if last is None else last[0]
            raise ResearchLockError(
                f"invalid project trial lifecycle transition: {prior_status} -> {status}"
            )
        c.execute(
            "INSERT INTO trial_attempt_event VALUES(?,?,?,?,strftime('%s','now'))",
            (attempt_key, event_key, status, detail_json),
        )

    def trial_attempt_summary(self, protocol_id: str) -> dict:
        """Return project and protocol coverage without inferring missing history."""
        with self._tx() as c:
            rows = c.execute(
                "SELECT attempt_key,protocol_id,candidate_id,candidate_sha,feature_sha,seed,recipe_json "
                "FROM trial_attempt WHERE protocol_id=? ORDER BY candidate_id", (protocol_id,),
            ).fetchall()
            total = int(c.execute("SELECT COUNT(*) FROM trial_attempt").fetchone()[0])
            project_rows = c.execute(
                "SELECT protocol_id, COUNT(*), COUNT(DISTINCT candidate_sha), "
                "COUNT(DISTINCT feature_sha), COUNT(DISTINCT seed) "
                "FROM trial_attempt GROUP BY protocol_id ORDER BY protocol_id",
            ).fetchall()
            project_events = c.execute(
                "SELECT a.protocol_id, e.status, COUNT(*) "
                "FROM trial_attempt AS a JOIN trial_attempt_event AS e "
                "ON e.attempt_key=a.attempt_key "
                "GROUP BY a.protocol_id, e.status ORDER BY a.protocol_id, e.status",
            ).fetchall()
            attempts = []
            for row in rows:
                events = c.execute(
                    "SELECT event_key,status,detail_json FROM trial_attempt_event "
                    "WHERE attempt_key=? ORDER BY rowid", (row[0],),
                ).fetchall()
                attempts.append({
                    "attempt_key": row[0], "protocol_id": row[1], "candidate_id": row[2],
                    "candidate_sha256": row[3], "feature_sha256": row[4], "seed": row[5],
                    "recipe": json.loads(row[6]),
                    "events": [{"event_key": event[0], "status": event[1],
                                "detail": json.loads(event[2])} for event in events],
                })
            return {
                "schema": "rles.project-trial-attempt-inventory.v1",
                "protocol_id": protocol_id,
                "project_registered_attempt_count": total,
                "project_protocol_inventory": [
                    {
                        "protocol_id": row[0],
                        "registered_attempt_count": row[1],
                        "distinct_recipe_count": row[2],
                        "distinct_feature_count": row[3],
                        "distinct_seed_count": row[4],
                        "lifecycle_event_counts": {
                            event[1]: event[2]
                            for event in project_events if event[0] == row[0]
                        },
                    }
                    for row in project_rows
                ],
                "protocol_registered_attempt_count": len(attempts),
                "attempts": attempts,
                "historical_coverage": "incomplete; registration covers protocols run after this ledger capability is enabled",
                "unrecorded_dimensions": [
                    "manual selections outside registered WFO protocols",
                    "historical and unledgered attempts before registration was enabled",
                    "failed or abandoned experiments that never reached candidate registration",
                ],
            }


def _same_test_query_component(owner: str, purpose: str, start_ns: int, end_ns: int,
                               encoded: str, previous_owner: str, previous_purpose: str,
                               previous_start: int, previous_end: int,
                               previous_identity: str) -> bool:
    """Permit separate Test components only for one identical frozen query."""
    if (purpose != "test" or previous_purpose != "test" or owner != previous_owner
            or start_ns != previous_start or end_ns != previous_end):
        return False
    try:
        current = json.loads(encoded)
        previous = json.loads(previous_identity)
    except (TypeError, json.JSONDecodeError):
        return False
    shared_keys = ("protocol_id", "query_id", "source_sha", "bundle_sha", "recipe_sha",
                   "code_sha", "split", "mode")
    components = {current.get("component"), previous.get("component")}
    return (components == {"policy", "baselines"}
            and all(key in current and current[key] == previous.get(key) for key in shared_keys))


@dataclass(frozen=True)
class EvaluationAccess:
    registry: ProjectResearchRegistry
    key: str
    token: str
    identity_json: str

    @property
    def identity(self) -> dict:
        return json.loads(self.identity_json)

    def consume(self, *, bundle_sha: str, split: str, start_ns: int, end_ns: int,
                market: str, mode: str) -> None:
        identity = self.identity
        if any(identity[name] != value for name, value in {
            "bundle_sha": bundle_sha, "split": split, "start_ns": start_ns,
            "end_ns": end_ns, "market": market, "mode": mode,
        }.items()):
            raise ResearchLockError("evaluation request differs from its frozen access scope")
        self.registry.consume(self.key, self.token, identity)

    def complete(self, report_sha: str) -> None:
        self.registry.complete(self.key, self.token, report_sha)


def authorize_evaluation(registry: ProjectResearchRegistry, ledger: ResearchLedger, *,
                         protocol_id: str, query_id: str, owner: str, market: str,
                         family: str, start_ns: int, end_ns: int, split: str,
                         source_sha: str, bundle_sha: str, mode: str = "base",
                         component: str = "policy") -> EvaluationAccess:
    state = ledger.state(protocol_id)
    if component not in ({"policy", "baselines"} if split == "test" else {"policy"}):
        raise ResearchLockError("evaluation component is not part of the fixed protocol")
    if split == "test":
        query = ledger.query(protocol_id, query_id)
        if (state["state"] != "RECIPE_FROZEN" or query is None or query["kind"] != "test"):
            raise ResearchLockError("Test access requires frozen recipe and durable query")
    elif split == "holdout":
        if (state["state"] != "HOLDOUT_OPENED" or state["retrain_sha"] != bundle_sha
                or state["holdout_start_ns"] != start_ns
                or state["holdout_sha"] != _hash([source_sha, start_ns, end_ns])):
            raise ResearchLockError("Holdout access requires registered retrain and matching one-shot scope")
    else:
        raise ResearchLockError("formal access supports only Test/Holdout")
    identity = {"protocol_id": protocol_id, "query_id": query_id, "component": component,
                "bundle_sha": bundle_sha, "source_sha": source_sha, "market": market,
                "start_ns": start_ns, "end_ns": end_ns, "split": split, "mode": mode,
                "recipe_sha": state["recipe_sha"], "code_sha": state["code_sha"]}
    key = _hash([owner, query_id, component])
    token = registry.reserve(key=key, owner=owner, market=market, family=family,
                             start_ns=start_ns, end_ns=end_ns, purpose=split, identity=identity)
    return EvaluationAccess(registry, key, token, json.dumps(identity, sort_keys=True))
