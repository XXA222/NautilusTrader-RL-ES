"""Explicit operator CLI for inspecting and recovering supervised resource leases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .errors import ComputeError
from .resources import Lease, ResourceQuota


def inspect_quota(path: str | Path) -> dict:
    database = Path(path)
    if database.is_symlink() or not database.is_file():
        raise ComputeError('quota inspection requires an existing regular database')
    database = database.resolve()
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('BEGIN')
        limits = connection.execute('SELECT cpu,ram,gpu FROM quota_meta WHERE id=1').fetchone()
        if limits is None:
            raise ComputeError('quota database has no frozen limits')
        leases = [dict(zip(('token', 'owner', 'cpu_cores', 'ram_bytes', 'gpu_count', 'acquired'), row))
                  for row in connection.execute('SELECT token,owner,cpu,ram,gpu,acquired FROM lease ORDER BY acquired')]
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        enrollments = ({row[0]: row[1] for row in connection.execute('SELECT token,state FROM lease_recovery')}
                       if 'lease_recovery' in tables else {})
        for lease in leases:
            enrolled = enrollments.get(lease['token'])
            lease['recovery_state'] = (enrolled if enrolled in {'application_preparing', 'application_ready'}
                                       else 'unsupported_gpu' if lease['gpu_count']
                                       else enrolled or 'unmanaged_or_legacy')
        recovered = ([json.loads(row[0]) for row in
                      connection.execute('SELECT receipt_json FROM lease_recovery_log ORDER BY recovered')]
                     if 'lease_recovery_log' in tables else [])
        return {'schema': 'rles.quota-inspection.v1', 'database': str(database),
                'limits': dict(zip(('cpu_cores', 'ram_bytes', 'gpus'), limits)),
                'leases': leases, 'recoveries': recovered,
                'scope': 'read-only quota state; ready does not prove owner/tree exit'}
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='read lease state without changing the database')
    inspect.add_argument('--database', required=True)
    recover = commands.add_parser('recover', help='reclaim one exact dead-owner CPU worker lease')
    recover.add_argument('--database', required=True)
    recover.add_argument('--token', required=True)
    recover.add_argument('--owner', required=True)
    recover.add_argument('--cpu-cores', type=int, required=True)
    recover.add_argument('--ram-bytes', type=int, required=True)
    recover.add_argument('--reason', required=True)
    recover_gpu = commands.add_parser(
        'recover-supervised-gpu', help='reclaim one exact supervised application GPU lease',
    )
    recover_gpu.add_argument('--database', required=True)
    recover_gpu.add_argument('--token', required=True)
    recover_gpu.add_argument('--owner', required=True)
    recover_gpu.add_argument('--cpu-cores', type=int, required=True)
    recover_gpu.add_argument('--ram-bytes', type=int, required=True)
    recover_gpu.add_argument('--reason', required=True)
    args = parser.parse_args(argv)
    try:
        state = inspect_quota(args.database)
        if args.command == 'inspect':
            result = state
        else:
            # Exact immutable descriptor must match the reviewed inspection.
            gpu_count = 1 if args.command == 'recover-supervised-gpu' else 0
            lease = Lease(args.token, args.cpu_cores, args.ram_bytes, gpu_count, args.owner)
            expected = {'token': lease.token, 'owner': lease.owner, 'cpu_cores': lease.cpu_cores,
                        'ram_bytes': lease.ram_bytes, 'gpu_count': lease.gpu_count}
            if not any(all(row[key] == value for key, value in expected.items()) for row in state['leases']):
                raise ComputeError('exact lease descriptor does not match inspection')
            quota = ResourceQuota(args.database, **state['limits'])
            result = (quota.recover_supervised_gpu_lease(lease, reason=args.reason)
                      if args.command == 'recover-supervised-gpu'
                      else quota.recover_cpu_lease(lease, reason=args.reason))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': type(exc).__name__, 'detail': str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
