#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miner Bee - Unified Asset Miner
Mines and processes non-code assets (Data, Weights, Configs).
"""

import ast
import csv
import hashlib
import json
import logging
import multiprocessing
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hive.bees.base_bee import BaseBee
from hive.config.settings import get_settings, settings
from hive.core.dna import BeeDNA
from hive.core.tool_registry import BeeToolRegistry, bee_tool


@bee_tool(
    name="分析代码结构", description="解析 Python/JS/Go 源码文件，返回函数列表、类列表、import 依赖"
)
def _analyze_code_file(file_path: str) -> dict:
    """提取代码结构信息（全局工具，可被任意工蜂调用）"""
    path = Path(file_path)
    if not path.exists() or path.stat().st_size > 5 * 1024 * 1024:
        return {"error": "file_too_large_or_missing"}
    suffix = path.suffix.lower()
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        result = {"lines": len(lines), "suffix": suffix}
        if suffix == ".py":
            result.update(_extract_python_structure(content))
        else:
            result.update(_extract_generic_structure(content, suffix))
        return result
    except Exception as e:
        return {"error": str(e)[:80]}


def _extract_python_structure(content: str) -> dict:
    """用 AST 解析 Python 代码结构"""
    try:
        tree = ast.parse(content)
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        imports = [
            n.names[0].name if isinstance(n, ast.Import) else n.module
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom)) and getattr(n, "names", None)
        ]
        return {
            "language": "python",
            "functions": functions[:20],
            "classes": classes[:10],
            "imports": [i for i in imports if i][:15],
        }
    except SyntaxError:
        return {"language": "python", "error": "syntax_error"}


def _extract_generic_structure(content: str, suffix: str) -> dict:
    """用正则提取其他语言结构"""
    lang_map = {
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
    }
    functions = re.findall(r"(?:func|function|def)\s+(\w+)\s*\(", content)
    classes = re.findall(r"(?:class|struct|interface)\s+(\w+)", content)
    return {
        "language": lang_map.get(suffix, suffix.lstrip(".")),
        "functions": functions[:20],
        "classes": classes[:10],
    }


logger = logging.getLogger("Hive.Miner")


class MinerBee(BaseBee):
    """
    Miner Bee - Unified non-code asset extraction and analysis.
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue: Any):
        super().__init__(dna, id, result_queue)
        safety_dna = BeeDNA(species="SafetyBee", target_name="Internal_Guard")
        self.safety_guard = None

        self.settings = get_settings()
        self.base_dir = Path(self.settings.BASE_DIR)
        self.ledger_path = self.base_dir / "mining_ledger.json"

        self.asset_types = {
            "json": [".json"],
            "csv": [".csv"],
            "yaml": [".yaml", ".yml"],
            "toml": [".toml"],
            "weights": [
                ".h5",
                ".hdf5",
                ".pkl",
                ".pickle",
                ".pt",
                ".pth",
                ".onnx",
                ".safetensors",
                ".weights",
                ".model",
                ".ckpt",
                ".bin",
            ],
            "config": [".toml", ".ini", ".cfg", ".conf", ".env", "config", "settings"],
            "data": [".npy", ".npz", ".mat", ".data", ".db", ".sqlite", ".dat"],
            "docs": [".md", ".log", ".xml", ".html", ".txt"],
            "code": [
                ".py",
                ".js",
                ".ts",
                ".java",
                ".go",
                ".rs",
                ".cpp",
                ".c",
                ".h",
                ".cs",
                ".php",
                ".rb",
                ".sh",
                ".bat",
                ".ps1",
            ],
        }
        self.minable_extensions = [ext for types in self.asset_types.values() for ext in types]
        self.performance_metrics = {}

    def specialized_task(self, target_path: Path) -> Dict[str, Any]:
        """Execute mining task with optional deep analysis."""
        logger.info(f"⛏️ MinerBee Starting: {self.dna.target_name}")
        start_time = time.time()

        try:
            if not self._perform_safety_check(target_path):
                return {"status": "blocked", "reason": "Safety check failed"}

            minerals = self._scan_minerals(target_path)
            if not minerals:
                logger.info(f"⛏️ No minerals found in: {self.dna.target_name}")
                return {"status": "empty"}

            categorized = self._categorize_minerals(minerals)
            if self.dna.metadata.get("deep_scan", True):
                self._enrich_minerals_with_analysis(target_path, categorized)

            recommendations = self._generate_recommendations(categorized, minerals)
            self._update_ledger(target_path, categorized, recommendations)
            self._index_to_vault(target_path, categorized)

            return self._finalize_metrics(start_time, minerals, categorized, recommendations)

        except Exception as e:
            logger.error(f"❌ Mining Failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _perform_safety_check(self, target_path: Path) -> bool:
        """PHASE 0: PRE-FLIGHT SAFETY SCAN"""
        # Placeholder logic as SafetyBee instance creation was mocked in __init__
        if not self.safety_guard:
            return True

        try:
            logger.info(f"🛡️ [Safety Protocol] Scanning {target_path}...")
            res = self.safety_guard.specialized_task(target_path)
            if res.get("status") == "error" or res.get("threats_found", 0) > 0:
                logger.warning(f"Safety scan issues found in {target_path}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Safety scan warning: {e}")
            return True  # Fail open if safety scan errors out? Or fail closed?

    def _finalize_metrics(
        self, start_time: float, minerals: List, categorized: Dict, recommendations: List
    ) -> Dict:
        scan_time = time.time() - start_time
        self.performance_metrics["scan_time"] = scan_time
        self.performance_metrics["total_assets"] = len(minerals)
        logger.info(
            f"✅ Mining Complete: {self.dna.target_name} ({len(minerals)} assets in {scan_time:.2f}s)"
        )

        return {
            "status": "success",
            "minerals_found": len(minerals),
            "categorized": {k: len(v) for k, v in categorized.items()},
            "recommendations": len(recommendations),
            "ledger_updated": True,
        }

    def _scan_minerals(self, target_path: Path) -> List[Dict[str, Any]]:
        """Fast scan for assets."""
        minerals = []
        for file_path in target_path.rglob("*"):
            if self._should_skip_file(file_path):
                continue

            info = self._process_file_info(file_path, target_path)
            if info:
                minerals.append(info)

        return minerals

    def _should_skip_file(self, file_path: Path) -> bool:
        if not file_path.is_file():
            return True
        exclude_dirs = {".git", "__pycache__", "node_modules", "venv", ".env", "dist", "build"}
        if any(part in exclude_dirs for part in file_path.parts):
            return True

        suffix = file_path.suffix.lower()
        is_known_type = any(suffix in exts for exts in self.asset_types.values())
        if not is_known_type and suffix not in self.minable_extensions:
            return True
        return False

    def _process_file_info(self, file_path: Path, target_path: Path) -> Optional[Dict]:
        suffix = file_path.suffix.lower()
        asset_type = self._determine_asset_type(suffix)

        info = {
            "name": file_path.name,
            "path": str(file_path.relative_to(target_path)),
            "extension": suffix,
            "type": asset_type,
            "size": 0,
            "modified_time": "",
            "is_large": False,
            "hash": "not_calculated",
            "safety_status": "clean",
        }

        try:
            if not self._scan_content_safety(file_path, asset_type, info):
                return info  # Return blocked info

            stat = file_path.stat()
            info.update(
                {
                    "size": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_large": stat.st_size > 10 * 1024 * 1024,
                }
            )

            if stat.st_size < 100 * 1024 * 1024:
                info["hash"] = self._calculate_fast_hash(file_path)
            return info
        except Exception as e:
            logger.debug(f"Failed to get file info: {e}")
            return None

    def _determine_asset_type(self, suffix: str) -> str:
        for type_name, exts in self.asset_types.items():
            if suffix in exts:
                return type_name
        return "other"

    def _scan_content_safety(self, file_path: Path, asset_type: str, info: Dict) -> bool:
        if not self.safety_guard:
            return True
        if asset_type in ["weights", "data"] or file_path.stat().st_size >= 5 * 1024 * 1024:
            return True

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        vulns = self.safety_guard.scan_vulnerabilities(content)
        if vulns:
            logger.warning(f"⚠️ THREAT DETECTED in {file_path.name}: {vulns[0]['type']}")
            info["safety_status"] = "blocked"
            info["threats"] = vulns
            return False
        return True

    def _calculate_fast_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                sha256.update(f.read(1024 * 1024))
            return sha256.hexdigest()[:16]
        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            return "hash_error"

    def _categorize_minerals(self, minerals: List[Dict]) -> Dict[str, List[Dict]]:
        categorized = {k: [] for k in self.asset_types.keys()}
        categorized["other"] = []
        for m in minerals:
            t = m["type"]
            if t in categorized:
                categorized[t].append(m)
            else:
                categorized["other"].append(m)
        return categorized

    def _enrich_minerals_with_analysis(self, target_path: Path, categorized: Dict[str, List[Dict]]):
        for item in categorized.get("json", [])[:50]:
            item["analysis"] = self._analyze_json(target_path / item["path"])
        for item in categorized.get("csv", [])[:20]:
            item["analysis"] = self._analyze_csv(target_path / item["path"])
        for item in categorized.get("yaml", [])[:20]:
            item["analysis"] = self._analyze_yaml(target_path / item["path"])
        # 🆕 代码结构分析（借鉴 LangFlow Component 模式）
        for item in categorized.get("code", [])[:30]:
            item["code_analysis"] = BeeToolRegistry.invoke(
                "分析代码结构", file_path=str(target_path / item["path"])
            )

    def _analyze_json(self, file_path: Path) -> Dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            structure = {"type": type(data).__name__}
            if isinstance(data, dict):
                structure["keys"] = list(data.keys())[:10]
                structure["keys_count"] = len(data)
            elif isinstance(data, list):
                structure["count"] = len(data)
            return structure
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            logger.debug(f"JSON parse failed: {e}")
            return {"error": "parse_failed"}

    def _analyze_csv(self, file_path: Path) -> Dict:
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if not rows:
                    return {"empty": True}
                return {"rows": len(rows), "cols": len(rows[0]), "headers": rows[0]}
        except (csv.Error, IOError, UnicodeDecodeError) as e:
            logger.debug(f"CSV parse failed: {e}")
            return {"error": "parse_failed"}

    def _analyze_yaml(self, file_path: Path) -> Dict:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return {"lines": len(lines), "keys_estimate": len([l for l in lines if ":" in l])}
        except (IOError, UnicodeDecodeError) as e:
            logger.debug(f"YAML read failed: {e}")
            return {"error": "read_failed"}

    def _generate_recommendations(self, categorized: Dict, minerals: List[Dict]) -> List[str]:
        recs = []
        if any(m.get("is_large") for m in minerals):
            recs.append("Found large files (>10MB).")
        if categorized.get("weights"):
            recs.append("Found model weights.")
        if len(categorized.get("csv", [])) > 10:
            recs.append("Found many CSV files.")
        return recs

    def _update_ledger(self, target_path: Path, categorized: Dict, recommendations: List[str]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "project": self.dna.target_name,
            "project_path": str(target_path),
            "stats": {k: len(v) for k, v in categorized.items()},
            "recommendations": recommendations,
            "details": categorized,
        }
        try:
            if self.ledger_path.exists():
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
            else:
                ledger = []
            ledger.append(entry)
            ledger = ledger[-50:]
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to update ledger: {e}")

    def _index_to_vault(self, target_path: Path, categorized: Dict):
        try:
            from hive.core.vector_vault import get_vector_vault

            vault = get_vector_vault()
            for cat, items in categorized.items():
                if cat in ["binary", "weights"]:
                    continue
                for item in items:
                    doc_id = f"{self.dna.target_name}/{item['path']}"
                    desc = f"Asset: {item['name']}\nType: {item['type']}"
                    if "analysis" in item:
                        desc += f"\nAnalysis: {json.dumps(item['analysis'])}"
                    vault.add_document(doc_id, desc, item)
            logger.info(f"🧠 VectorVault updated for {self.dna.target_name}")
        except Exception as e:
            logger.warning(f"VectorVault indexing failed: {e}")
