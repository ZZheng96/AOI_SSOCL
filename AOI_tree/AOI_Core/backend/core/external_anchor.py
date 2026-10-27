"""可选的外部可信锚定：运维密钥签名后导出，外部存证由运维负责。"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path


def _payload(chain: dict) -> bytes:
    return json.dumps(chain, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def export_anchor(chain: dict, output_path: str | os.PathLike[str] | None = None,
                  secret: str | None = None) -> dict:
    """签名并导出链摘要；未同时提供密钥和外部路径时明确返回未启用。"""
    path = str(output_path or os.environ.get("AOI_AUDIT_ANCHOR_PATH") or "").strip()
    key = secret or os.environ.get("AOI_AUDIT_ANCHOR_KEY", "")
    chain_hash = hashlib.sha256(_payload(chain)).hexdigest()
    if not key or not path:
        return {"enabled": False, "reason": "未配置运维签名密钥或外部存证路径",
                "chain_hash": chain_hash}
    signature = hmac.new(key.encode(), chain_hash.encode(), hashlib.sha256).hexdigest()
    record = {"chain_hash": chain_hash, "signature": signature, "algorithm": "HMAC-SHA256"}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(target)
    return {"enabled": True, "path": str(target), "chain_hash": chain_hash,
            "algorithm": record["algorithm"]}
