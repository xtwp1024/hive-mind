# hive.bees.cognitive_bee

CognitiveBee - 认知蜂 蜂巢的"大脑皮层" - 负责自我评估和元认知 增强功能 (Phase 1 增强): 1. 历史性能分析和对比 2. 改进建议生成 3. 主动学习触发器 4. 元学习能力增强 5. 与进化系统集成

## 目录

- [类](#类)
  - [ActiveLearningTrigger](#activelearningtrigger)
  - [BlindSpot](#blindspot)
  - [CapabilityAssessment](#capabilityassessment)
  - [CognitiveBee](#cognitivebee)
  - [HistoricalComparison](#historicalcomparison)
  - [ImprovementRecommendation](#improvementrecommendation)
  - [LearningPlan](#learningplan)

## 类

### ActiveLearningTrigger

**@dataclass**

主动学习触发器

#### 属性

- `auto_execute`

### BlindSpot

**@dataclass**

知识盲区

### CapabilityAssessment

**@dataclass**

能力评估结果

#### 属性

- `confidence`
- `evidence`
- `improvement_rate`
- `is_uncertain`
- `last_updated`

### CognitiveBee

**继承:** `BaseBee`

认知蜂 - 蜂巢的自我意识

#### 方法

```python
def activate() -> None
```

激活蜜蜂

```python
def analyze_improvement_recommendations() -> List
```

生成改进建议 基于当前能力评估和历史趋势，生成优先级排序的改进建议

**返回:** ImprovementRecommendation 列表

```python
def assess_capabilities() -> Dict
```

```python
def check_active_learning_triggers() -> List
```

检查主动学习触发器 识别需要主动学习的情况： 1. 低分能力 2. 高不确定性 3. 知识缺口 4. 下降趋势

**返回:** ActiveLearningTrigger 列表

```python
def close()
```

Close the Process object. This method releases resources held by the Process object.  It is an error to call this method if the child process is still running.

```python
def compare_with_history(capability_name: str) -> Optional
```

对比特定能力的历史性能

**参数:**

- `capability_name`: 能力名称

**返回:** HistoricalComparison 或 None（如果没有历史数据）

```python
def create_learning_plan(focus: Optional = None) -> LearningPlan
```

```python
def deactivate() -> None
```

停用蜜蜂

```python
def devour() -> None
```

吞噬循环 - 执行任务的核心流程 (Unified Logic)

```python
def execute_active_learning(trigger: ActiveLearningTrigger) -> Dict
```

执行主动学习任务

**参数:**

- `trigger`: ActiveLearningTrigger

**返回:** 学习结果

```python
async def execute_task(task: Task) -> TaskResult
```

执行任务

**参数:**

- `task`: 要执行的任务

**返回:** 任务结果

```python
def generate_enhanced_report(target_path: Path) -> Dict
```

生成增强版认知报告（包含历史对比、改进建议、主动学习触发

**参数:**

- `target_path`: 目标路径

**返回:** 增强版报告

```python
def get_all_historical_trends() -> Dict
```

获取所有能力的历史趋势

**返回:** 能力名称到 HistoricalComparison 的字典

```python
def get_cognitive_status() -> Dict
```

```python
def get_status() -> dict
```

获取蜜蜂状态

**返回:** 状态字典

```python
def identify_blind_spots() -> List
```

```python
def is_alive()
```

Return whether process is alive

```python
def join(timeout = None)
```

Wait until child process terminates

```python
def kill()
```

Terminate process; sends SIGKILL signal or uses TerminateProcess()

```python
def optimize_learning_strategy() -> Dict
```

```python
def recommend_learning_method(capability_name: str) -> Tuple
```

基于元学习推荐最佳学习方法

**参数:**

- `capability_name`: 能力名称

**返回:** (推荐方法, 预期效率)

```python
def record_learning_outcome(capability_name: str, success: bool, effort_hours: float, improvement: float)
```

记录学习结果，用于元学习

**参数:**

- `capability_name`: 能力名称
- `success`: 是否成功
- `effort_hours`: 投入时间（小时）
- `improvement`: 改进幅度

```python
def report_to_evolution_coordinator()
```

将认知状态报告给进化协调器 这使得进化系统可以基于认知状态调整进化策略

```python
def request_human_intervention(reason: str, risk_level: str = HIGH, context: dict = None) -> dict
```

[HITL] 人类介入挂起接口。 任何子类工蜂在执行到敏感、危险或无法自行判断的逻辑时（如资金交易配置、核心文件删除）， 调用此方法。当前进程将会把请求封装并返回给 Queen，由 Queen 派发给 WardenBee 处理， 或者当前进程直接挂起阻塞等待通信通道（简单实现上返回挂起载荷）。

```python
def return_payload(nutrient)
```

```python
def run()
```

Method to be run in sub-process; can be overridden in sub-class

```python
def spawn_sub_tasks(strategies: list, target_bee_type: str = Standard)
```

[HiVA] 递归派发/工蜂繁育请求。 当当前工蜂遇到超出单次上下文理解或解决尺寸的复杂体时， 使用该方法将大任务"切割"成子战略，要求统筹端动态繁育出新的工蜂同时去处理。

```python
def specialized_task(target_path: Path) -> Dict
```

```python
def start()
```

Start child process

```python
def terminate()
```

Terminate process; sends SIGTERM signal or uses TerminateProcess()

```python
def update_capability(domain: str, score_delta: float = 0.1)
```

```python
def update_dna(performance: TaskResult) -> None
```

根据任务执行结果更新DNA

**参数:**

- `performance`: 任务执行结果

#### 属性

- `authkey`
- `daemon`
- `exitcode`
- `ident`
- `name`
- `pid`
- `sentinel`

### HistoricalComparison

**@dataclass**

历史对比结果

### ImprovementRecommendation

**@dataclass**

改进建议

### LearningPlan

**@dataclass**

学习计划

## 导出

```python
__all__ = ['CognitiveBee', 'CapabilityAssessment', 'BlindSpot', 'LearningPlan', 'HistoricalComparison', 'ImprovementRecommendation', 'ActiveLearningTrigger']
```

---

*文档生成于: 2026-04-23T01:21:36.219198*
