from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4


def save_uploaded_file(upload, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.name).suffix.lower()
    safe_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}{suffix}"
    target = upload_dir / safe_name
    target.write_bytes(upload.getbuffer())
    return target
