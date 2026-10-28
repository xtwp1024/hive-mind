# hive.core.consensus

Consensus Engine (多端共识状态匹配器) 灵感来自于 ethereum/hive。 用来防范大型语言模型（LLM）"单点故障"或"幻觉臆想"。 对于高危指令，中枢系统会分叉出多个独立的 Worker（模拟多客户端节点），分别得出自己的答案。 该引擎作为"仲裁官"，应用多数投票（Nakamoto-style 变种）裁决最终可用状态。 Phase 2 P2: LLM Reasoning 加持 - 纯字符串Counter无法判断语义相近的答案 - 当多数投票失败时，调用 BrainInterface 进行语义分析 - 支持多轮投票收敛

## 目录

- [类](#类)
  - [ConsensusEngine](#consensusengine)
  - [DecisionProposal](#decisionproposal)
  - [DecisionResult](#decisionresult)
  - [DecisionType](#decisiontype)
  - [Vote](#vote)
  - [VoteOption](#voteoption)
- [函数](#函数)
  - [get_consensus_engine](#get_consensus_engine)

## 类

### ConsensusEngine

状态裁决与幻觉过滤器

#### 方法

```python
def cancel_proposal(proposal_id: str) -> bool
```

取消提案

**参数:**

- `proposal_id`: 提案ID

**返回:** 是否成功取消

```python
def cast_vote(proposal_id: str, voter_id: str, option: VoteOption, justification: Optional = None) -> bool
```

提交投票

**参数:**

- `proposal_id`: 提案ID
- `voter_id`: 投票者ID
- `option`: 投票选项
- `justification`: 投票理由（可选）

**返回:** 是否成功提交

```python
def get_active_proposals() -> List
```

获取活跃提案列表

```python
def get_decision_history() -> List
```

获取决策历史

```python
def get_vote_status(proposal_id: str) -> Dict
```

获取提案的投票状态

**参数:**

- `proposal_id`: 提案ID

**返回:** 投票状态字典

```python
def match_semantics(results: List, enable_llm: bool = True) -> Tuple
```

基于绝对字符串或基础语义的硬分叉判定。 Phase 2 P2 升级： - 第一轮：严格 Counter 多数投票（极速） - 第二轮（可选）：LLM 语义分析（处理语义相近的答案） - 第三轮：多轮投票收敛

**参数:**

- `results`: 各节点返回的结果列表
- `enable_llm`: 是否启用 LLM 语义分析（当 Counter 失败时）

**返回:** (is_consensus, canonical_result)

```python
def process_node_payloads(payloads: List, enable_llm: bool = True) -> Dict
```

从一堆工蜂传回的 payload 里提取核心 result 并发起裁决

**参数:**

- `payloads`: 各节点返回的 payload 列表
- `enable_llm`: 是否启用 LLM 增强（默认开启）

**返回:** 包含 status, canonical_result, consensus_id, nodes_participated 的字典

```python
async def propose_decision(proposal: DecisionProposal, voter_ids: List) -> DecisionResult
```

提出决策提案并等待结果

**参数:**

- `proposal`: 决策提案
- `voter_ids`: 投票者ID列表

**返回:** 决策结果

#### 属性

- `FORK_DIVERSITY_THRESHOLD`

### DecisionProposal

**@dataclass**

决策提案 (Phase 2 新增)

#### 方法

```python
def is_expired() -> bool
```

#### 属性

- `decision_type`
- `expires_at`
- `quorum`
- `time_limit`

### DecisionResult

**@dataclass**

决策结果 (Phase 2 新增)

### DecisionType

**Enum**

**继承:** `Enum`

决策类型 (Phase 2 新增)

#### 属性

- `MAJORITY`
- `PLURALITY`
- `SUPER_MAJORITY`
- `UNANIMOUS`
- `WEIGHTED`
- `name`
- `value`

### Vote

**@dataclass**

投票记录 (Phase 2 新增)

#### 属性

- `justification`
- `weight`

### VoteOption

**Enum**

**继承:** `Enum`

投票选项 (Phase 2 新增)

#### 属性

- `ABSTAIN`
- `APPROVE`
- `DEFER`
- `REJECT`
- `name`
- `value`

## 函数

### get_consensus_engine

```python
def get_consensus_engine(communication_bus: Any = None) -> ConsensusEngine
```

获取共识引擎单例

**参数:**

- `communication_bus`: 通信总线实例（可选）

**返回:** ConsensusEngine 实例

---

*文档生成于: 2026-04-23T01:21:36.316638*
