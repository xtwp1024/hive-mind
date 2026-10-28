# hive.core.dna_adaptive

DNA Adaptive Modification - DNA 自适应修改系统 Phase 3: 自主进化能力 为 DNA 系统添加参数自适应调整能力。 核心功能: 1. 性能反馈驱动 - 基于任务结果自动调整 DNA 参数 2. 学习率自适应 - 根据性能变化动态调整学习率 3. 变异率自适应 - 根据适应度动态调整变异率 4. 能力权重优化 - 基于任务表现优化能力权重 5. 进化方向引导 - 根据缺口分析引导进化方向 工作原理: - 收集性能反馈 → 分析趋势 → 调整参数 → 验证效果 → 持续迭代

## 目录

- [类](#类)
  - [AdaptationAction](#adaptationaction)
  - [AdaptationType](#adaptationtype)
  - [DNAAdaptiveEngine](#dnaadaptiveengine)
  - [ParameterState](#parameterstate)
  - [PerformanceSample](#performancesample)
  - [PerformanceTrend](#performancetrend)
- [函数](#函数)
  - [get_dna_adaptive_engine](#get_dna_adaptive_engine)

## 类

### AdaptationAction

**@dataclass**

适应动作

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `applied`
- `applied_at`
- `result`

### AdaptationType

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `ALL_PARAMS`
- `CAPABILITY_WEIGHT`
- `IMMUNE_RESPONSE`
- `LEARNING_RATE`
- `MUTATION_RATE`
- `name`
- `value`

### DNAAdaptiveEngine

DNA 自适应引擎 Phase 3 核心组件。为 DNA 系统添加参数自适应调整能力。 主要功能: 1. 性能监控 - 持续收集 DNA 性能数据 2. 趋势分析 - 分析性能变化趋势 3. 参数调整 - 根据趋势自动调整 DNA 参数 4. 效果验证 - 验证调整是否有效 5. 反馈闭环 - 根据验证结果优化调整策略 自适应策略: - 当性能下降时：增加学习率，增加变异率 - 当性能提升时：降低学习率，稳定变异率 - 当性能波动时：降低学习率，增加探索 - 当性能稳定时：微调参数，优化细节 使用方式: engine = DNAAdaptiveEngine() engine.monitor_dna(bee_dna, "bee_001") engine.record_performance("bee_001", sample) actions = engine.suggest_adaptations("bee_001")

#### 方法

```python
def add_adaptation_callback(callback: Callable)
```

添加适应动作回调

```python
def apply_adaptation(dna_id: str, action: AdaptationAction) -> bool
```

应用适应动作

**参数:**

- `dna_id`: DNA 标识
- `action`: 适应动作

**返回:** bool - 是否成功应用

```python
def apply_all_suggested(dna_id: str, min_confidence: float = 0.6, max_actions: int = 5) -> int
```

应用所有建议的适应动作

**参数:**

- `dna_id`: DNA 标识
- `min_confidence`: 最小置信度阈值
- `max_actions`: 最大动作数量

**返回:** int - 成功应用的动作数量

```python
def get_status(dna_id: Optional = None) -> Dict
```

获取状态

```python
def monitor_dna(dna, dna_id: str, dna_type: str = bee)
```

开始监控 DNA

**参数:**

- `dna`: DNA 实例（BeeDNA 或 DNA）
- `dna_id`: DNA 唯一标识
- `dna_type`: DNA 类型

```python
def record_performance(dna_id: str, success: bool, duration: float, error: bool = False, task_type: Optional = None, capabilities_used: Optional = None) -> bool
```

记录性能数据

**参数:**

- `dna_id`: DNA 标识
- `success`: 是否成功
- `duration`: 耗时
- `error`: 是否错误
- `task_type`: 任务类型
- `capabilities_used`: 使用的能力

**返回:** bool - 是否成功记录

```python
def start_auto_adapt(interval_seconds: float = 300.0)
```

启动自动适应

```python
def stop_auto_adapt()
```

停止自动适应

```python
def stop_monitoring(dna_id: str) -> bool
```

停止监控 DNA

```python
def suggest_adaptations(dna_id: str, adaptation_types: Optional = None) -> List
```

建议适应动作

**参数:**

- `dna_id`: DNA 标识
- `adaptation_types`: 要考虑的适应类型

**返回:** List[AdaptationAction] - 建议的适应动作列表

#### 属性

- `ADAPTATION_COOLDOWN`
- `LEARNING_RATE_INITIAL`
- `LEARNING_RATE_MAX`
- `LEARNING_RATE_MIN`
- `MUTATION_RATE_INITIAL`
- `MUTATION_RATE_MAX`
- `MUTATION_RATE_MIN`

### ParameterState

**@dataclass**

参数状态

#### 方法

```python
def get_average(window: int = 10) -> float
```

获取平均值

```python
def get_trend() -> Tuple
```

获取趋势

```python
def record_value(value: float, timestamp: Optional = None)
```

记录新值

```python
def to_dict() -> Dict
```

#### 属性

- `adaptation_count`
- `last_adaptation_at`
- `max_value`
- `min_value`
- `successful_adaptations`

### PerformanceSample

**@dataclass**

性能样本

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `task_complexity`
- `task_type`

### PerformanceTrend

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `DECLINING`
- `IMPROVING`
- `STABLE`
- `VOLATILE`
- `name`
- `value`

## 函数

### get_dna_adaptive_engine

```python
def get_dna_adaptive_engine() -> DNAAdaptiveEngine
```

获取 DNA 自适应引擎单例

---

*文档生成于: 2026-04-23T01:21:36.456436*
