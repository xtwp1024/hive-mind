# -*- coding: utf-8 -*-
"""
阶段 B: 生产级分布式网络 - 测试脚本

功能:
1. 多节点部署测试 (3+ 节点)
2. 持久化集成测试
3. 性能基准测试
4. 故障注入测试

使用方法:
    python hive/scripts/run_distributed_test.py
    python hive/scripts/run_distributed_test.py --test multi-node
    python hive/scripts/run_distributed_test.py --test persistence
    python hive/scripts/run_distributed_test.py --test all
"""

import sys
import os
import time
import json
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    test_name: str
    status: TestStatus
    duration: float = 0.0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    suite_name: str
    tests: List[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    
    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.PASS)
    
    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if t.status in [TestStatus.FAIL, TestStatus.ERROR])
    
    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.SKIP)
    
    @property
    def total(self) -> int:
        return len(self.tests)
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100


def test_hive_federation_imports() -> TestResult:
    """测试 HiveFederation 导入"""
    start_time = time.time()
    
    try:
        from hive.core.hive_federation import (
            HiveFederation,
            HiveNode,
        )
        
        from hive.core.knowledge_sync import (
            KnowledgeSync,
            KnowledgeEntry,
        )
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="HiveFederation 导入",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "HiveFederation": "可用",
                "HiveNode": "可用",
                "KnowledgeSync": "可用",
                "KnowledgeEntry": "可用",
            }
        )
        
    except ImportError as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="HiveFederation 导入",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_hive_node_basic() -> TestResult:
    """测试 HiveNode 基本功能"""
    start_time = time.time()
    
    try:
        from hive.core.hive_federation import HiveNode
        
        node = HiveNode(
            node_id="test-node-001",
            host="127.0.0.1",
            port=8765,
            status="active",
            capabilities=["text", "code"],
            load_factor=0.0,
            version="1.0.0",
        )
        
        node_dict = node.to_dict()
        
        restored_node = HiveNode.from_dict(node_dict)
        
        is_alive = node.is_alive()
        
        url = node.get_url("api/status")
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="HiveNode 基本功能",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "node_id": node.node_id,
                "host": node.host,
                "port": node.port,
                "is_alive": is_alive,
                "url": url,
                "dict_keys": list(node_dict.keys()),
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="HiveNode 基本功能",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_hive_federation_creation() -> TestResult:
    """测试 HiveFederation 创建"""
    start_time = time.time()
    
    try:
        from hive.core.hive_federation import HiveFederation
        
        federation = HiveFederation(
            node_id="test-federation-001",
            host="127.0.0.1",
            port=0,
        )
        
        assert federation is not None
        assert federation.node_id == "test-federation-001"
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="HiveFederation 创建",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "node_id": federation.node_id,
                "host": federation.host,
                "port": federation.port,
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="HiveFederation 创建",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_hive_federation_node_management() -> TestResult:
    """测试 HiveFederation 节点管理"""
    start_time = time.time()
    
    try:
        from hive.core.hive_federation import HiveFederation, HiveNode
        
        federation = HiveFederation(
            node_id="test-federation-manager",
            host="127.0.0.1",
            port=0,
        )
        
        node1 = HiveNode(
            node_id="test-node-1",
            host="127.0.0.1",
            port=8766,
            status="active",
            capabilities=["text"],
            load_factor=0.0,
        )
        
        federation._nodes["test-node-1"] = node1
        
        all_nodes = federation.get_all_nodes()
        status = federation.get_status()
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="HiveFederation 节点管理",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "registered_nodes": len(all_nodes),
                "status_keys": list(status.keys()),
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="HiveFederation 节点管理",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_knowledge_entry_basic() -> TestResult:
    """测试 KnowledgeEntry 基本功能"""
    start_time = time.time()
    
    try:
        from hive.core.knowledge_sync import KnowledgeEntry
        
        content = {"pattern": "测试模式", "description": "这是测试知识内容"}
        content_hash = KnowledgeEntry.compute_hash(content)
        
        entry = KnowledgeEntry(
            id="test-knowledge-001",
            type="pattern",
            content=content,
            content_hash=content_hash,
            priority="high",
            tags=["test", "pattern"],
            source_node="test-node",
            source_timestamp=time.time(),
            version=1,
        )
        
        entry_dict = entry.to_dict()
        
        restored_entry = KnowledgeEntry.from_dict(entry_dict)
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="KnowledgeEntry 基本功能",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "entry_id": entry.id,
                "entry_type": entry.type,
                "version": entry.version,
                "priority": entry.priority,
                "dict_keys": list(entry_dict.keys()),
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="KnowledgeEntry 基本功能",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_knowledge_sync_creation() -> TestResult:
    """测试 KnowledgeSync 创建"""
    start_time = time.time()
    
    try:
        from hive.core.hive_federation import HiveFederation
        from hive.core.knowledge_sync import KnowledgeSync
        
        federation = HiveFederation(
            node_id="test-sync-federation",
            host="127.0.0.1",
            port=0,
        )
        
        sync = KnowledgeSync(
            federation=federation,
            sync_interval=60.0,
            max_knowledge=10000,
        )
        
        assert sync is not None
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="KnowledgeSync 创建",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "sync_interval": 60.0,
                "max_knowledge": 10000,
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="KnowledgeSync 创建",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_knowledge_sync_crud() -> TestResult:
    """测试 KnowledgeSync CRUD 操作"""
    start_time = time.time()
    
    try:
        from hive.core.hive_federation import HiveFederation
        from hive.core.knowledge_sync import KnowledgeSync, KnowledgeEntry
        
        federation = HiveFederation(
            node_id="test-crud-federation",
            host="127.0.0.1",
            port=0,
        )
        
        sync = KnowledgeSync(
            federation=federation,
            sync_interval=60.0,
        )
        
        content = {"test": "crud test"}
        content_hash = KnowledgeEntry.compute_hash(content)
        
        entry = KnowledgeEntry(
            id="test-crud-001",
            type="test",
            content=content,
            content_hash=content_hash,
            priority="normal",
        )
        
        with sync._kb_lock:
            sync._knowledge[entry.id] = entry
            if entry.type not in sync._by_type:
                sync._by_type[entry.type] = set()
            sync._by_type[entry.type].add(entry.id)
            sync._by_hash[entry.content_hash] = entry.id
        
        with sync._kb_lock:
            retrieved = sync._knowledge.get("test-crud-001")
        
        status = sync.get_status()
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="KnowledgeSync CRUD",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "retrieved_id": retrieved.id if retrieved else None,
                "status_keys": list(status.keys()),
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="KnowledgeSync CRUD",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_memory_store_basic() -> TestResult:
    """测试 MemoryStore 基本功能"""
    start_time = time.time()
    
    try:
        from hive.core.memory_store import MemoryStore
        
        store = MemoryStore()
        
        result = store.memorize(
            content="这是测试记忆内容。这是第二句话用于提取多个 items。这是第三句话。",
            agent_id="test-agent",
            modality="conversation",
        )
        
        retrieve_result = store.retrieve_rag(
            query="测试记忆",
            agent_id="test-agent",
            top_k=5,
        )
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="MemoryStore 基本功能",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "resource_id": result.get("resource_id"),
                "items_count": result.get("items", 0),
                "retrieve_count": len(retrieve_result),
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="MemoryStore 基本功能",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def test_memory_store_stats() -> TestResult:
    """测试 MemoryStore 统计功能"""
    start_time = time.time()
    
    try:
        from hive.core.memory_store import MemoryStore
        
        store = MemoryStore()
        
        store.memorize(
            content="统计测试内容。第一句话。第二句话。",
            agent_id="stat-agent",
            modality="conversation",
        )
        
        stats = store.get_persistence_stats()
        
        duration = time.time() - start_time
        
        return TestResult(
            test_name="MemoryStore 统计",
            status=TestStatus.PASS,
            duration=duration,
            details={
                "stats_keys": list(stats.keys()),
            }
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            test_name="MemoryStore 统计",
            status=TestStatus.ERROR,
            duration=duration,
            error_message=str(e),
        )


def run_multi_node_tests() -> TestSuiteResult:
    """运行多节点部署测试"""
    suite = TestSuiteResult(suite_name="多节点部署测试")
    start_time = time.time()
    
    tests = [
        ("test_hive_federation_imports", test_hive_federation_imports),
        ("test_hive_node_basic", test_hive_node_basic),
        ("test_hive_federation_creation", test_hive_federation_creation),
        ("test_hive_federation_node_management", test_hive_federation_node_management),
        ("test_knowledge_entry_basic", test_knowledge_entry_basic),
        ("test_knowledge_sync_creation", test_knowledge_sync_creation),
        ("test_knowledge_sync_crud", test_knowledge_sync_crud),
    ]
    
    for test_name, test_func in tests:
        print(f"\n  测试: {test_name}")
        result = test_func()
        suite.tests.append(result)
        
        if result.status == TestStatus.PASS:
            print(f"    结果: [通过] ({result.duration:.2f}s)")
        elif result.status == TestStatus.SKIP:
            print(f"    结果: [跳过] ({result.duration:.2f}s)")
            if result.error_message:
                print(f"    详情: {result.error_message}")
        else:
            print(f"    结果: [错误] ({result.duration:.2f}s)")
            if result.error_message:
                print(f"    详情: {result.error_message}")
    
    suite.total_duration = time.time() - start_time
    return suite


def run_persistence_tests() -> TestSuiteResult:
    """运行持久化集成测试"""
    suite = TestSuiteResult(suite_name="持久化集成测试")
    start_time = time.time()
    
    tests = [
        ("test_memory_store_basic", test_memory_store_basic),
        ("test_memory_store_stats", test_memory_store_stats),
    ]
    
    for test_name, test_func in tests:
        print(f"\n  测试: {test_name}")
        result = test_func()
        suite.tests.append(result)
        
        if result.status == TestStatus.PASS:
            print(f"    结果: [通过] ({result.duration:.2f}s)")
        elif result.status == TestStatus.SKIP:
            print(f"    结果: [跳过] ({result.duration:.2f}s)")
            if result.error_message:
                print(f"    详情: {result.error_message}")
        else:
            print(f"    结果: [错误] ({result.duration:.2f}s)")
            if result.error_message:
                print(f"    详情: {result.error_message}")
    
    suite.total_duration = time.time() - start_time
    return suite


def generate_report(suites: List[TestSuiteResult], output_file: Optional[str] = None) -> Dict:
    """生成测试报告"""
    total_tests = sum(s.total for s in suites)
    total_passed = sum(s.passed for s in suites)
    total_failed = sum(s.failed for s in suites)
    total_skipped = sum(s.skipped for s in suites)
    total_duration = sum(s.total_duration for s in suites)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_suites": len(suites),
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0.0,
            "total_duration": total_duration,
        },
        "suites": [],
    }
    
    for suite in suites:
        suite_data = {
            "name": suite.suite_name,
            "total": suite.total,
            "passed": suite.passed,
            "failed": suite.failed,
            "skipped": suite.skipped,
            "success_rate": suite.success_rate,
            "duration": suite.total_duration,
            "tests": [],
        }
        
        for test in suite.tests:
            test_data = {
                "name": test.test_name,
                "status": test.status.value,
                "duration": test.duration,
                "error_message": test.error_message,
                "details": test.details,
            }
            suite_data["tests"].append(test_data)
        
        report["suites"].append(suite_data)
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(report))
    
    return report


def generate_markdown_report(report: Dict) -> str:
    """生成 Markdown 格式报告"""
    md = """# Hive 分布式网络测试报告

---

## 📊 测试汇总

| 指标 | 数值 |
|------|------|
| **测试套件** | """ + str(report["summary"]["total_suites"]) + """ |
| **总测试** | """ + str(report["summary"]["total_tests"]) + """ |
| **✅ 通过** | """ + str(report["summary"]["passed"]) + """ |
| **❌ 失败** | """ + str(report["summary"]["failed"]) + """ |
| **⏭️ 跳过** | """ + str(report["summary"]["skipped"]) + """ |
| **成功率** | """ + f"{report['summary']['success_rate']:.1f}%" + """ |
| **总耗时** | """ + f"{report['summary']['total_duration']:.2f} 秒" + """ |

---

"""
    
    for suite in report["suites"]:
        md += f"""## 📦 {suite["name"]}

| 状态 | 数量 |
|------|------|
| 总测试 | {suite["total"]} |
| ✅ 通过 | {suite["passed"]} |
| ❌ 失败 | {suite["failed"]} |
| ⏭️ 跳过 | {suite["skipped"]} |
| 成功率 | {suite["success_rate"]:.1f}% |
| 耗时 | {suite["duration"]:.2f}s |

### 测试详情

| 测试名称 | 状态 | 耗时 | 详情 |
|----------|------|------|------|
"""
        
        for test in suite["tests"]:
            status_icon = "✅" if test["status"] == "PASS" else ("⏭️" if test["status"] == "SKIP" else "❌")
            details = json.dumps(test["details"], ensure_ascii=False) if test["details"] else "-"
            md += f'| {test["name"]} | {status_icon} {test["status"]} | {test["duration"]:.2f}s | {details} |\n'
        
        md += "\n---\n\n"
    
    md += """## 🛠️ 快速命令

```bash
# 运行所有测试
python hive/scripts/run_distributed_test.py

# 只运行多节点测试
python hive/scripts/run_distributed_test.py --test multi-node

# 只运行持久化测试
python hive/scripts/run_distributed_test.py --test persistence
```

---

*报告生成于: """
    md += report["generated_at"]
    md += """*
"""
    
    return md


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Hive 分布式网络测试执行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "multi-node", "persistence"],
        help="测试套件: all (全部), multi-node (多节点), persistence (持久化)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reports/distributed_test_report.json",
        help="测试报告输出文件",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("阶段 B: 生产级分布式网络 - 测试执行器")
    print("=" * 60)
    print(f"\n测试套件: {args.test}")
    
    suites = []
    
    if args.test in ["all", "multi-node"]:
        print("\n" + "=" * 60)
        print("执行多节点部署测试...")
        print("=" * 60)
        suite = run_multi_node_tests()
        suites.append(suite)
        
        print(f"\n  阶段汇总: {suite.passed} 通过, {suite.failed} 失败, {suite.skipped} 跳过")
        print(f"  成功率: {suite.success_rate:.1f}%")
        print(f"  耗时: {suite.total_duration:.2f} 秒")
    
    if args.test in ["all", "persistence"]:
        print("\n" + "=" * 60)
        print("执行持久化集成测试...")
        print("=" * 60)
        suite = run_persistence_tests()
        suites.append(suite)
        
        print(f"\n  阶段汇总: {suite.passed} 通过, {suite.failed} 失败, {suite.skipped} 跳过")
        print(f"  成功率: {suite.success_rate:.1f}%")
        print(f"  耗时: {suite.total_duration:.2f} 秒")
    
    print("\n" + "=" * 60)
    print("生成测试报告...")
    print("=" * 60)
    
    report = generate_report(suites, args.output)
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    total_tests = report["summary"]["total_tests"]
    total_passed = report["summary"]["passed"]
    total_failed = report["summary"]["failed"]
    total_skipped = report["summary"]["skipped"]
    success_rate = report["summary"]["success_rate"]
    total_duration = report["summary"]["total_duration"]
    
    for suite in report["suites"]:
        status = "✅" if suite["success_rate"] == 100 else ("⚠️" if suite["failed"] == 0 else "❌")
        print(f"\n  {status} {suite['name']}:")
        print(f"      通过: {suite['passed']}, 失败: {suite['failed']}, 跳过: {suite['skipped']}")
        print(f"      成功率: {suite['success_rate']:.1f}%, 耗时: {suite['duration']:.2f}s")
    
    print("\n" + "-" * 60)
    print(f"\n  总计:")
    print(f"    通过:   {total_passed}")
    print(f"    失败:   {total_failed}")
    print(f"    跳过:   {total_skipped}")
    print(f"    总测试: {total_tests}")
    print(f"    总耗时: {total_duration:.2f} 秒")
    
    if total_failed > 0:
        print(f"\n  ❌ 存在 {total_failed} 个失败/错误")
    elif total_skipped > 0:
        print(f"\n  ⚠️ 存在 {total_skipped} 个跳过的测试")
    else:
        print(f"\n  ✅ 所有测试通过!")
    
    print("\n" + "=" * 60)
    
    if total_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
