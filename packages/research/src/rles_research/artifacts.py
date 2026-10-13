"""Atomic, content-addressed research artifacts; untrusted bytes are never loaded as objects."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
from collections.abc import Mapping

_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
_EXT = {'.json','.safetensors','.npz','.arrow','.parquet','.txt','.bin','.zip'}
_HASH_FIELDS = {'data_hash','code_hash','config_hash','seed_hash','deps_hash',
                'adapter_hash','compute_hash','execution_hash','risk_hash'}


class ArtifactError(RuntimeError):
    pass


def _canonical(data: object) -> bytes:
    return json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode('utf-8')


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    if not isinstance(value,str) or not _ID.fullmatch(value) or value in {'.','..'}:
        raise ArtifactError('invalid run/attempt ID')
    return value


def _relative(name: str) -> PurePosixPath:
    if not isinstance(name,str) or '\\' in name or '\x00' in name or name.startswith('/'):
        raise ArtifactError('invalid artifact name')
    path = PurePosixPath(name)
    if any(piece in {'.','..'} for piece in name.split('/')) or len(path.parts) > 8:
        raise ArtifactError('unsafe artifact path')
    if path.suffix.lower() not in _EXT or name in {'COMPLETED','FAILED','integrity.json'}:
        raise ArtifactError('forbidden artifact extension/name')
    return path


class ArtifactWriter:
    def __init__(self, store: 'ArtifactStore', run_id: str, attempt_id: str):
        self.run_id, self.attempt_id = _safe_id(run_id), _safe_id(attempt_id)
        self._store = store
        self._root = store.root / self.run_id
        self._stage = self._root / '.staging' / self.attempt_id
        self._target = self._root / 'attempts' / self.attempt_id
        self._stage.mkdir(mode=0o700,parents=True,exist_ok=False)
        self._files: dict[str,str] = {}
        self._finished = False

    def add(self, name: str, content: bytes) -> None:
        if self._finished:
            raise ArtifactError('writer already sealed')
        path = _relative(name)
        if not isinstance(content,bytes) or str(path) in self._files:
            raise ArtifactError('bytes expected / duplicate artifact')
        dest = self._stage.joinpath(*path.parts)
        dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if dest.exists() or dest.is_symlink():
            raise ArtifactError('artifact already exists')
        with dest.open('xb') as stream:
            stream.write(content)
            stream.flush(); os.fsync(stream.fileno())
        self._files[str(path)] = _digest(content)

    def complete(self, *, l1_manifest: Mapping[str, object], required_files: list[str],
                 trusted_source_id: str) -> Path:
        if self._finished:
            raise ArtifactError('already sealed')
        if not trusted_source_id or trusted_source_id not in self._store.trusted_sources:
            raise ArtifactError('unknown/untrusted artifact source')
        if not isinstance(l1_manifest,Mapping) or not _HASH_FIELDS.issubset(l1_manifest):
            raise ArtifactError('L1 experiment manifest is missing mandatory fingerprints')
        for key in _HASH_FIELDS:
            v = l1_manifest[key]
            if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
                raise ArtifactError(f'invalid L1 {key}')
        if not isinstance(required_files,list) or len(required_files)!=len(set(required_files)):
            raise ArtifactError('required artifact inventory must be unique')
        required = {str(_relative(x)) for x in required_files}
        if not required or not required.issubset(self._files):
            raise ArtifactError('required output missing; refusing completion')
        manifest_bytes = _canonical(dict(l1_manifest))
        self.add('manifest.json',manifest_bytes)
        record = {'schema_version':1,'run_id':self.run_id,'attempt_id':self.attempt_id,
                  'source_id':trusted_source_id,'files':self._files,'required':sorted(required),
                  'manifest_sha256':_digest(manifest_bytes)}
        index = _canonical(record)
        with (self._stage/'integrity.json').open('xb') as stream:
            stream.write(index); stream.flush(); os.fsync(stream.fileno())
        # Re-read every artifact before commit, not merely trust previous writes.
        for name,hash_value in self._files.items():
            p=self._stage/name
            if p.is_symlink() or _file_digest(p) != hash_value:
                raise ArtifactError('staging content changed before commit')
        self._target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self._target.exists():
            raise ArtifactError('attempt already exists')
        os.rename(self._stage,self._target)
        # Marker is LAST; crashes before the marker are invisible to cache/registry.
        marker = _canonical({'integrity_sha256':_digest(index),'state':'COMPLETED'})
        fd = os.open(self._target/'COMPLETED', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(marker);stream.flush();os.fsync(stream.fileno())
        self._finished = True
        return self._target

    def fail(self, reason_code: str) -> None:
        if self._finished:
            raise ArtifactError('already sealed')
        if not isinstance(reason_code,str) or not re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}',reason_code):
            raise ArtifactError('failure reason must be a nonsecret code')
        (self._stage/'FAILED').write_bytes(_canonical({'state':'FAILED','reason_code':reason_code}))
        self._finished = True


class ArtifactStore:
    """Trust pins are operator-supplied source IDs, NOT a substitute for signatures."""

    def __init__(self, root: str|Path, *, trusted_sources: set[str]):
        if not trusted_sources or any(not _ID.fullmatch(x) for x in trusted_sources):
            raise ArtifactError('trusted_sources allowlist is required')
        self.root=Path(root)
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.trusted_sources=frozenset(trusted_sources)

    def begin(self, run_id: str, attempt_id: str) -> ArtifactWriter:
        return ArtifactWriter(self,run_id,attempt_id)

    def verify(self, run_id: str, attempt_id: str, *, expected_manifest_hash: str|None=None) -> dict:
        directory=self.root/_safe_id(run_id)/'attempts'/_safe_id(attempt_id)
        if not directory.is_dir() or directory.is_symlink() or not (directory/'COMPLETED').is_file():
            raise ArtifactError('attempt is not committed')
        try:
            index_data=(directory/'integrity.json').read_bytes()
            record=json.loads(index_data)
            marker=json.loads((directory/'COMPLETED').read_bytes())
        except (OSError,ValueError) as exc:
            raise ArtifactError('invalid marker/index') from exc
        if marker!={'integrity_sha256':_digest(index_data),'state':'COMPLETED'}:
            raise ArtifactError('integrity marker does not match index')
        if record.get('source_id') not in self.trusted_sources or record.get('run_id')!=run_id or record.get('attempt_id')!=attempt_id:
            raise ArtifactError('untrusted source or attempt mismatch')
        files=record.get('files')
        if not isinstance(files,dict) or not files or 'manifest.json' not in files:
            raise ArtifactError('missing hash inventory')
        if set(record) != {'schema_version','run_id','attempt_id','source_id','files','required','manifest_sha256'} or record['schema_version']!=1:
            raise ArtifactError('artifact index schema drift')
        if not isinstance(record['required'],list) or not record['required'] or len(set(record['required']))!=len(record['required']):
            raise ArtifactError('missing/ambiguous required artifact inventory')
        if any(not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in files.values()):
            raise ArtifactError('invalid artifact hash inventory')
        # A committed attempt is an exact inventory, not merely a set of files
        # whose hashes happen to match. Unindexed files can otherwise be picked
        # up by directory-scanning consumers without being covered by the L1
        # manifest or the integrity record.
        expected_names = set(files) | {'COMPLETED', 'integrity.json'}
        actual_names: set[str] = set()
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ArtifactError('symlink found in committed artifact tree')
            if path.is_file():
                actual_names.add(path.relative_to(directory).as_posix())
            elif not path.is_dir():
                raise ArtifactError('non-regular entry found in committed artifact tree')
        if actual_names != expected_names:
            raise ArtifactError('unexpected or unindexed artifact files')
        for name,checksum in files.items():
            relative=_relative(name)
            if str(relative) != name:
                raise ArtifactError('noncanonical artifact path in integrity inventory')
            p=directory.joinpath(*relative.parts)
            if not p.is_file() or p.is_symlink() or any(part.is_symlink() for part in p.parents if part!=directory.parent.parent.parent) or _file_digest(p)!=checksum:
                raise ArtifactError(f'artifact hash mismatch: {name}')
        if not set(record.get('required',[])).issubset(files):
            raise ArtifactError('mandatory artifact inventory incomplete')
        if record.get('manifest_sha256')!=files['manifest.json']:
            raise ArtifactError('L1 manifest digest mismatch')
        try:
            manifest=json.loads((directory/'manifest.json').read_bytes())
        except (OSError,ValueError) as exc:
            raise ArtifactError('unreadable L1 experiment manifest') from exc
        if not isinstance(manifest,dict) or not _HASH_FIELDS.issubset(manifest) or any(
            not isinstance(manifest[k],str) or len(manifest[k])!=64 or
            any(c not in '0123456789abcdef' for c in manifest[k]) for k in _HASH_FIELDS):
            raise ArtifactError('committed L1 manifest schema/hash fields invalid')
        if expected_manifest_hash is not None and record['manifest_sha256']!=expected_manifest_hash:
            raise ArtifactError('cache identity mismatch')
        return record

    def read_bytes(self, run_id: str, attempt_id: str, name: str, *,
                   expected_manifest_hash: str, allow_serialized_model: bool=False) -> bytes:
        record=self.verify(run_id,attempt_id,expected_manifest_hash=expected_manifest_hash)
        path=_relative(name)
        if str(path) not in record['files']:
            raise ArtifactError('unlisted artifact')
        if path.suffix in {'.zip','.npz','.bin'} and not allow_serialized_model:
            raise ArtifactError('opaque serialized model needs explicit trusted consumer authorization')
        data=(self.root/run_id/'attempts'/attempt_id/str(path)).read_bytes()
        if _digest(data)!=record['files'][str(path)]:
            raise ArtifactError('concurrent artifact mutation detected')
        return data
