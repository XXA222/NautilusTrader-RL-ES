"""Source/log policy scanning; scanning is a gate input, not credential protection by itself."""
from __future__ import annotations

from pathlib import Path
import re

SECRET_PATTERNS={
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'aws_access_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'github_token':re.compile(r'\bgh[pousr]_[A-Za-z0-9_]{20,}\b'),
    'authorization_bearer':re.compile(r'(?i)\bauthorization\s*[:=]\s*["\']?Bearer\s+\S+'),
    'named_secret_value':re.compile(r'(?im)^\s*(?:api_key|secret_key|private_key|password)\s*[:=]\s*["\']?[^\s"\'${}][^\s"\']{7,}'),
}


def scan_text(text:str)->list[dict[str,object]]:
    # Only type+line; NEVER echo sensitive matched content into evidence/logs.
    return [{'kind':name,'line':text.count('\n',0,match.start())+1}
            for name,regex in SECRET_PATTERNS.items() for match in regex.finditer(text)]


def audit_files(files:list[str|Path], *, max_size:int=2_000_000)->dict:
    findings=[]
    for item in files:
        path=Path(item)
        if path.is_symlink() or not path.is_file() or path.stat().st_size>max_size:
            raise ValueError('security scanner refuses symlinks/missing/oversize files')
        for problem in scan_text(path.read_text(encoding='utf-8')):
            findings.append({'path':str(path),'kind':problem['kind'],'line':problem['line']})
    return {'status':'PASS' if not findings else 'BLOCKED','findings':findings,'files_checked':len(files)}
