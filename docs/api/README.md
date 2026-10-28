# Hive API 文档索引

> 生成时间: 2026-04-23T01:21:36.826722

## 模块列表

### Phase 1-2: 认知觉醒与群体智慧

- **hive.bees.cognitive_bee** - CognitiveBee - 认知蜂 蜂巢的"大脑皮层" - 负责自我评估和元认知 增强功能 (Phase 1 增强): 1. 历史性能分析和对比 2. 改进建议生成 3. 主动学习触发器 4. 元学...
  - 包含: 7 个类
- **hive.core.swarm_communication** - SwarmCommunication - 蜂群通信系统 实现蜂种之间的协作和集体智慧 增强功能 (Phase 1 增强): 1. 优先级队列 - 高优先级消息优先处理 2. 消息过期机制 - 防止过时...
  - 包含: 8 个类, 1 个函数
- **hive.core.consensus** - Consensus Engine (多端共识状态匹配器) 灵感来自于 ethereum/hive。 用来防范大型语言模型（LLM）"单点故障"或"幻觉臆想"。 对于高危指令，中枢系统会分叉出多个独立的...
  - 包含: 6 个类, 1 个函数

### Phase 3: 自主进化能力

- **hive.core.capability_gap_analyzer** - Capability Gap Analyzer - 能力缺口分析器 Phase 3: 自主进化能力 分析历史任务数据，发现能力缺口，为自动蜂种生成提供输入。 核心功能: 1. 任务历史追踪 - 收集所...
  - 包含: 6 个类, 1 个函数
- **hive.core.species_blueprint** - Species Blueprint - 蜂种蓝图系统 Phase 3: 自主进化能力 定义新蜂种的完整规格，作为自动蜂种生成的模板。 核心概念: 1. SpeciesBlueprint - 蜂种蓝图，...
  - 包含: 8 个类, 1 个函数
- **hive.core.auto_species_generator** - Auto Species Generator - 自动蜂种生成器 Phase 3: 自主进化能力 基于能力缺口分析自动生成新的蜂种蓝图。 核心功能: 1. 缺口分析驱动 - 从 CapabilityG...
  - 包含: 6 个类, 1 个函数
- **hive.core.resource_allocator** - Dynamic Resource Allocation - 动态资源分配系统 Phase 3: 自主进化能力 基于适应度和优先级动态分配系统资源。 核心功能: 1. 资源配额管理 - 为每个蜂种分配资...
  - 包含: 7 个类, 1 个函数
- **hive.core.dna_adaptive** - DNA Adaptive Modification - DNA 自适应修改系统 Phase 3: 自主进化能力 为 DNA 系统添加参数自适应调整能力。 核心功能: 1. 性能反馈驱动 - 基于任务结...
  - 包含: 6 个类, 1 个函数

### Phase 4: 多模态扩展

- **hive.core.multimodal_encoder** - Multimodal Encoder - 多模态编码器 Phase 4: 多模态扩展 统一编码文本、图像、音频等多种模态的输入。 核心功能: 1. 文本编码器 - 使用 BERT/GPT 等模型编码文...
  - 包含: 9 个类, 1 个函数
- **hive.core.cross_modal_association** - Cross-Modal Association - 跨模态关联模块 Phase 4: 多模态扩展 实现不同模态之间的关联和理解。 核心功能: 1. 跨模态注意力对齐 - 使用注意力机制对齐不同模态 2...
  - 包含: 9 个类, 1 个函数
- **hive.core.multimodal_task_decoder** - Multimodal Task Decoder - 多模态任务解码器 Phase 4: 多模态扩展 根据融合后的多模态表示执行具体任务。 核心功能: 1. 任务类型识别 - 自动识别任务类型 2. 融...
  - 包含: 9 个类, 1 个函数

### Phase 5: 分布式网络

- **hive.core.hive_federation** - Hive Federation (蜂巢联邦) Phase 5 P1: 多实例协作 多个 Hive 实例互联互通，形成分布式蜂群网络。 核心职责: 1. 节点注册与发现 — 多 Hive 实例自动发现彼...
  - 包含: 2 个类, 2 个函数
- **hive.core.knowledge_sync** - Knowledge Sync (知识同步) Phase 5 P2: 全球知识共享 任意蜂群学到的知识可被其他蜂群使用。 核心职责: 1. 增量同步 — 只同步新增知识，避免重复传输 2. 知识去重 —...
  - 包含: 2 个类, 2 个函数
- **hive.core.node_auth** - Node Authentication (节点身份认证) Phase 5: 分布式网络 - 安全增强 核心职责: 1. 节点身份生成与验证 - 唯一身份、密钥对 2. API Key 管理 - 访问令...
  - 包含: 11 个类, 2 个函数
- **hive.core.conflict_resolution** - Conflict Resolution (冲突解决策略) Phase 5: 分布式网络 - 知识同步增强 核心职责: 1. 知识版本冲突检测 - 检测同时更新导致的冲突 2. 自动冲突解决 - 基于规...
  - 包含: 10 个类, 2 个函数
- **hive.core.load_balancer** - Load Balancer & Failover (负载均衡与故障转移) Phase 5: 分布式网络 - 高可用增强 核心职责: 1. 负载均衡策略 - 多种算法选择最优节点 2. 健康检查 - 持...
  - 包含: 11 个类, 2 个函数

### PRD UI 控制台

- **hive.core.dispatcher** - Hive V2.0 Neural Orchestration Engine (Dispatcher) -------------------------------------------------...
  - 包含: 8 个类, 2 个函数
- **hive.evolution.migrations** - Hive 数据库版本化迁移系统 基于 GEPStore 的 Schema Versioning 实现。 确保数据库平滑升级，数据不丢失。 设计原则: 1. 每个迁移是一个独立的版本 2. 支持升级和降...
  - 包含: 5 个类, 2 个函数

### 存储系统

- **hive.core.memory_store** - MemoryStore - 三层层次记忆仓库 设计灵感来源于 NevaMind-AI/memU（"记忆即文件系统"哲学）。 三层架构（Hierarchical Memory Architecture）...
  - 包含: 4 个类

## 统计

- **分析模块数:** 19
- **总类数:** 134
- **总函数数:** 28

---

*本文档由 `generate_api_docs.py` 自动生成*
