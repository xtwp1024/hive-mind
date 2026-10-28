# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import time
import json
import traceback
from pathlib import Path

@dataclass
class DeathCertificate:
    """
    死亡证明 (Death Certificate)
    
    当工蜂崩溃时生成的诊断文件，包含完整的上下文信息，
    用于后续的自动修复 (Self-Healing) 和尸检 (Post-Mortem).
    """
    bee_id: str
    bee_type: str
    error_type: str
    error_message: str
    timestamp: float = field(default_factory=time.time)
    traceback_str: str = ""
    target_name: str = "Unknown"
    dna_snapshot: Dict[str, Any] = field(default_factory=dict)
    environment_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, morgue_dir: Path) -> Path:
        """
        将死亡证明归档到停尸房（原子写入，防止崩溃导致数据丢失）

        Args:
            morgue_dir: 停尸房目录路径

        Returns:
            归档文件的路径
        """
        import tempfile
        import os
        if not morgue_dir.exists():
            morgue_dir.mkdir(parents=True, exist_ok=True)

        filename = f"death_cert_{self.bee_type}_{self.bee_id}_{int(self.timestamp)}.json"
        file_path = morgue_dir / filename

        data = asdict(self)

        try:
            # 先写入临时文件，然后原子重命名，防止进程崩溃导致证书丢失
            dir_fd, tmp_path = tempfile.mkstemp(dir=morgue_dir, suffix='.json.tmp')
            try:
                with os.fdopen(dir_fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                # 原子重命名（在同一目录，无需权限问题）
                os.replace(tmp_path, file_path)
            except Exception:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise
            return file_path
        except Exception as e:
            # 万一写文件也挂了，至少打印出来
            print(f"CRITICAL: Failed to write death certificate to {file_path}: {e}")
            return None

_SENSITIVE_KEYWORDS = {
    "key", "token", "secret", "password", "passwd", "api_key",
    "openai", "anthropic", "gemini", "github", "auth", "credential",
}

def _redact_dict(d: Dict[str, Any], max_depth: int = 10, _path: str = "") -> Dict[str, Any]:
    """过滤字典中的敏感字段，防止密钥写入诊断文件"""
    if max_depth <= 0:
        path_info = f" at '{_path}'" if _path else ""
        return {"_truncated": f"Max depth exceeded{path_info}"}
    result = {}
    for k, v in d.items():
        if any(kw in k.lower() for kw in _SENSITIVE_KEYWORDS):
            result[k] = "[REDACTED]"
        elif isinstance(v, dict):
            result[k] = _redact_dict(v, max_depth - 1, f"{_path}.{k}" if _path else k)
        else:
            result[k] = v
    return result


def create_death_certificate(
    bee: Any,
    exception: Exception
) -> DeathCertificate:
    """
    为濒死的工蜂开具死亡证明
    """

    # 尝试捕获一些环境信息
    env_snapshot = {
        "cwd": str(Path.cwd()),
    }
    
    # 尝试获取 DNA，做安全处理
    dna_data = {}
    if hasattr(bee, 'dna') and bee.dna:
        if hasattr(bee.dna, '__dict__'):
            dna_data = _redact_dict(dict(bee.dna.__dict__))
        else:
            dna_data = str(bee.dna)

    return DeathCertificate(
        bee_id=str(getattr(bee, 'id', 'unknown')),
        bee_type=bee.__class__.__name__,
        error_type=type(exception).__name__,
        error_message=str(exception),
        traceback_str=traceback.format_exc(),
        target_name=str(getattr(bee.dna, 'target_name', 'unknown')) if hasattr(bee, 'dna') else "unknown",
        dna_snapshot=dna_data,
        environment_snapshot=_redact_dict(env_snapshot)
    )
