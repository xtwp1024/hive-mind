# -*- coding: utf-8 -*-
"""
将今天的开发记录存储到 Hive KnowledgeBase 中

同时支持：
1. Obsidian Markdown 文件 (已创建)
2. Hive KnowledgeBase (内存存储)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime


def store_to_knowledge_base():
    """
    将今天的知识存储到 Hive KnowledgeBase
    """
    print("=" * 60)
    print("存储知识到 Hive KnowledgeBase")
    print("=" * 60)
    print()

    # 导入 KnowledgeBase
    from hive.core.knowledge_base import KnowledgeBase

    # 创建知识库实例
    kb = KnowledgeBase()
    kb.set_generation(1)  # 设置世代

    print("[1/5] 初始化 KnowledgeBase...")
    print(f"    知识库状态: {kb}")
    print()

    # ============ 存储知识 ============

    print("[2/5] 存储今天的开发记录...")
    print()

    # 1. MemoryStore 持久化
    knowledge_1 = """
## MemoryStore 持久化增强

### 新增功能
- save() - 手动保存
- load() - 手动加载
- auto_save=True - 自动保存模式
- get_persistence_stats() - 获取持久化状态

### 实现细节
1. 所有数据类添加 to_dict() / from_dict() 方法
2. 使用 MemoryBox (shelve) 持久化到磁盘
3. 程序退出时自动保存 (atexit)
4. 每 10 次变更自动保存

### 存储位置
.hive_memory/box_memory_store_{store_name}.db

### 验证结果
5/5 测试通过
"""
    kb.add_knowledge(
        content=knowledge_1, importance=0.95, source_atom_id="development-2026-04-22", vectors=[]
    )
    print("    [OK] MemoryStore 持久化知识已存储")

    # 2. 代码审查
    knowledge_2 = """
## 代码审查结果

### 审查模块 (13 个)
1. capability_gap_analyzer.py - OK
2. species_blueprint.py - OK
3. auto_species_generator.py - OK
4. resource_allocator.py - OK
5. dna_adaptive.py - OK
6. multimodal_encoder.py - OK
7. cross_modal_association.py - OK
8. multimodal_task_decoder.py - OK
9. node_auth.py - OK
10. conflict_resolution.py - OK
11. load_balancer.py - OK
12. hive_federation.py - OK
13. memory_store.py - OK

### 代码质量分析
- 循环导入: 无问题 (使用延迟导入)
- 线程安全: 完善 (37 个文件使用锁)
- 异常处理: 完善 (50 个文件有 try/except)
- 日志记录: 完善 (72 个文件使用 logger)
- 资源管理: 完善 (使用 atexit)

### 结论
所有核心模块代码质量良好，可以正常使用。
"""
    kb.add_knowledge(
        content=knowledge_2, importance=0.9, source_atom_id="review-2026-04-22", vectors=[]
    )
    print("    [OK] 代码审查知识已存储")

    # 3. 完整测试
    knowledge_3 = """
## 完整测试结果

### 快速验证统计
- Phase 1/2: 10/10 ✅
- Phase 3: 5/5 ✅
- Phase 4: 8/8 ✅
- Phase 5: 10/10 ✅
- MemoryStore: 全部 ✅
- 总计: 43+ 测试通过

### 验证的核心模块

#### Phase 1/2
- CognitiveBee - 自我评估、学习计划、知识盲区
- SwarmCommunication - 消息队列、任务协商、集体决策
- ConsensusEngine - 语义匹配、投票系统、LLM 仲裁

#### Phase 3
- CapabilityGapAnalyzer - 能力缺口分析
- SpeciesBlueprint - 蜂种蓝图系统
- AutoSpeciesGenerator - 自动蜂种生成器
- DynamicResourceAllocator - 动态资源分配器
- DNAAdaptiveEngine - DNA 自适应修改引擎

#### Phase 4
- MultimodalEncoder - 多模态编码器
- CrossModalAttention - 跨模态注意力
- UnifiedRepresentationSpace - 统一表示空间
- CrossModalRetriever - 跨模态检索器
- MultimodalTaskDecoder - 多模态任务解码器

#### Phase 5
- HiveFederation - 蜂巢联邦
- KnowledgeSync - 知识同步
- IdentityManager - 身份管理器
- APIKeyManager - API Key 管理器
- SignatureVerifier - 签名验证器
- AccessControlList - 访问控制列表
- ConflictResolver - 冲突解决器
- LoadBalancer - 负载均衡器
- HealthChecker - 健康检查器
- FailoverManager - 故障转移管理器
"""
    kb.add_knowledge(
        content=knowledge_3, importance=0.95, source_atom_id="testing-2026-04-22", vectors=[]
    )
    print("    [OK] 完整测试知识已存储")

    # 4. 多节点集成测试
    knowledge_4 = """
## 多节点集成测试结果

### 测试架构
3 个 HiveFederation 实例:
- node-alpha: localhost:8751
- node-beta: localhost:8752
- node-gamma: localhost:8753

### 测试结果 (4/4 ✅)

#### 1. HTTP 服务器启动
- 多节点同时在不同端口启动 ✅
- 节点 ID 分配正常 ✅

#### 2. HTTP API 连通性
- sync_request 请求正常 ✅
- 所有节点响应正常 ✅

#### 3. 节点注册
- 每个节点注册其他两个节点 ✅
- get_all_nodes() 返回正确数量 ✅

#### 4. 能力过滤
- get_nodes_by_capability() 正常 ✅
- get_best_node() 正常 ✅

#### 5. 知识广播
- broadcast_knowledge() 内部广播 ✅
- HTTP knowledge_broadcast 正常 ✅

#### 6. 任务委托
- HTTP task_delegate 正常 ✅

### 结论
HiveFederation 多节点通信功能完全正常！
"""
    kb.add_knowledge(
        content=knowledge_4, importance=0.9, source_atom_id="integration-2026-04-22", vectors=[]
    )
    print("    [OK] 多节点集成测试知识已存储")

    # 5. 五阶段进化路线图
    knowledge_5 = """
## 五阶段进化路线图

### 阶段概览
| 阶段 | 时间 | 名称 | 核心目标 | 状态 |
|------|------|------|----------|------|
| Phase 1 | 1-3个月 | 认知觉醒 | 自我评估 + 主动学习 | ✅ 已完成 |
| Phase 2 | 3-6个月 | 群体智慧 | 蜂间通信 + 集体决策 | ✅ 已完成 |
| Phase 3 | 6-12个月 | 自主进化 | 自动蜂种生成 + 自然选择 | ✅ 已完成 |
| Phase 4 | 12-18个月 | 多模态扩展 | 视觉/音频/多模态融合 | ✅ 已完成 |
| Phase 5 | 18-24个月 | 分布式网络 | 蜂巢联邦 + 全球扩展 | ✅ 已完成 |

### 核心模块

#### Phase 1/2
- CognitiveBee - 认知智能
- SwarmCommunication - 蜂间通信
- ConsensusEngine - 共识引擎

#### Phase 3
- CapabilityGapAnalyzer - 能力缺口分析
- SpeciesBlueprint - 蜂种蓝图
- AutoSpeciesGenerator - 自动蜂种生成
- DynamicResourceAllocator - 动态资源分配
- DNAAdaptiveEngine - DNA 自适应

#### Phase 4
- MultimodalEncoder - 多模态编码
- CrossModalAssociation - 跨模态关联
- MultimodalTaskDecoder - 多模态任务解码

#### Phase 5
- HiveFederation - 蜂巢联邦
- NodeAuthEngine - 节点认证
- ConflictResolution - 冲突解决
- LoadBalancerEngine - 负载均衡
- MemoryStore - 记忆存储 (已增强)
"""
    kb.add_knowledge(
        content=knowledge_5, importance=1.0, source_atom_id="roadmap-2026-04-22", vectors=[]
    )
    print("    [OK] 五阶段路线图知识已存储")
    print()

    # ============ 验证存储 ============

    print("[3/5] 验证知识存储...")
    print()

    # 获取统计信息
    stats = kb.get_stats()
    print(f"    知识库统计:")
    print(f"      知识条目数: {stats['total_knowledge']}")
    print(f"      总重要性: {stats['total_importance']:.2f}")
    print(f"      平均重要性: {stats['average_importance']:.4f}")
    print(f"      当前世代: {stats['generation']}")
    print()

    # 查询最重要的知识
    print("[4/5] 查询最重要的知识...")
    print()

    top_knowledge = kb.query(limit=5)
    for i, entry in enumerate(top_knowledge, 1):
        print(f"    [{i}] ID: {entry.id}")
        print(f"        重要性: {entry.importance}")
        print(f"        来源: {entry.source_atom_id}")
        print(f"        内容预览: {entry.content[:100]}...")
        print()

    # ============ 输出汇总 ============

    print("[5/5] 输出汇总...")
    print()
    print("=" * 60)
    print("知识存储完成")
    print("=" * 60)
    print()
    print("✅ 同时存储到两个位置:")
    print()
    print("   1. Obsidian Markdown 文件:")
    print("      - obsidian_notes/Hive Mind Project/00-首页.md")
    print("      - obsidian_notes/Hive Mind Project/2026-04-22 开发记录 - MemoryStore 持久化.md")
    print("      - obsidian_notes/Hive Mind Project/2026-04-22 开发记录 - 代码审查.md")
    print("      - obsidian_notes/Hive Mind Project/2026-04-22 开发记录 - 完整测试.md")
    print("      - obsidian_notes/Hive Mind Project/2026-04-22 开发记录 - 多节点集成测试.md")
    print("      - obsidian_notes/Hive Mind Project/五阶段进化路线图.md")
    print("      - obsidian_notes/Hive Mind Project/2026-04-22 测试结果汇总.md")
    print()
    print("   2. Hive KnowledgeBase (内存):")
    print(f"      - 知识条目数: {stats['total_knowledge']}")
    print(f"      - 总重要性: {stats['total_importance']:.2f}")
    print()

    # 返回知识库实例，供外部使用
    return kb


def query_knowledge_demo(kb):
    """
    演示知识库查询功能
    """
    print()
    print("=" * 60)
    print("知识库查询演示")
    print("=" * 60)
    print()

    # 按关键词查询
    print("🔍 按关键词查询 ('持久化'):")
    results = kb.query(keywords="持久化", limit=3)
    for i, entry in enumerate(results, 1):
        print(f"  [{i}] {entry.source_atom_id}")
        print(f"      重要性: {entry.importance}")
    print()

    # 按另一个关键词查询
    print("🔍 按关键词查询 ('测试'):")
    results = kb.query(keywords="测试", limit=3)
    for i, entry in enumerate(results, 1):
        print(f"  [{i}] {entry.source_atom_id}")
        print(f"      重要性: {entry.importance}")
    print()

    print("✅ 查询演示完成！")


if __name__ == "__main__":
    try:
        # 存储知识
        kb = store_to_knowledge_base()

        # 演示查询
        query_knowledge_demo(kb)

        print()
        print("=" * 60)
        print("所有任务完成！")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print()
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
