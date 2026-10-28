# hive.core.capability_gap_analyzer

Capability Gap Analyzer - 能力缺口分析器 Phase 3: 自主进化能力 分析历史任务数据，发现能力缺口，为自动蜂种生成提供输入。 核心功能: 1. 任务历史追踪 - 收集所有任务执行数据 2. 能力缺口检测 - 分析成功率、耗时、错误模式发现缺口 3. 缺口严重度评估 - 基于业务影响和频率评估优先级 4. 缺口报告生成 - 为进化系统提供结构化输入 5. 趋势分析 - 识别能力随时间的变化趋势

## 目录

- [类](#类)
  - [CapabilityGap](#capabilitygap)
  - [CapabilityGapAnalyzer](#capabilitygapanalyzer)
  - [GapAnalysisReport](#gapanalysisreport)
  - [GapSeverity](#gapseverity)
  - [GapType](#gaptype)
  - [TaskExecutionRecord](#taskexecutionrecord)
- [函数](#函数)
  - [get_capability_gap_analyzer](#get_capability_gap_analyzer)

## 类

### CapabilityGap

**@dataclass**

CapabilityGap(gap_id: str, gap_type: hive.core.capability_gap_analyzer.GapType, severity: hive.core.capability_gap_analyzer.GapSeverity, capability_name: str, task_category: str, affected_tasks_count: int = 0, success_rate: float = 0.0, avg_duration: float = 0.0, expected_duration: Optional[float] = None, efficiency_ratio: Optional[float] = None, error_patterns: List[str] = <factory>, root_cause_hypothesis: List[str] = <factory>, trend_direction: str = 'stable', trend_confidence: float = 0.0, first_detected: float = <factory>, last_observed: float = <factory>, business_impact_score: float = 0.0, remediation_priority: float = 0.0, suggested_remediation: List[str] = <factory>, estimated_improvement: float = 0.0)

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `affected_tasks_count`
- `avg_duration`
- `business_impact_score`
- `efficiency_ratio`
- `estimated_improvement`
- `expected_duration`
- `remediation_priority`
- `success_rate`
- `trend_confidence`
- `trend_direction`

### CapabilityGapAnalyzer

能力缺口分析器 Phase 3 核心组件。分析历史任务数据，发现能力缺口， 为自动蜂种生成和进化系统提供输入。 分析维度: 1. 成功率缺口 - 低于阈值的任务类型 2. 效率缺口 - 耗时超过预期的任务 3. 错误模式 - 特定错误类型的聚集 4. 工作负载瓶颈 - 特定蜂种过载 5. 能力不匹配 - 分配的任务与能力不符 使用方式: analyzer = CapabilityGapAnalyzer() analyzer.record_task(TaskExecutionRecord(...)) report = analyzer.analyze(analysis_window_hours=24)

#### 方法

```python
def analyze(analysis_window_hours: float = 24.0) -> GapAnalysisReport
```

执行能力缺口分析

**参数:**

- `analysis_window_hours`: 分析时间窗口（小时）

**返回:** GapAnalysisReport - 完整的缺口分析报告

```python
def get_latest_report() -> Optional
```

获取最新报告

```python
def get_records(hours: Optional = None) -> List
```

获取记录（可选时间窗口）

```python
def get_stats() -> Dict
```

获取统计信息

```python
def integrate_with_swarm_evolution(coordinator) -> Dict
```

与蜂群进化系统集成 将能力缺口分析结果提供给进化系统， 指导下一代蜂的进化方向。

```python
def record_task(record: TaskExecutionRecord) -> None
```

记录任务执行

```python
def record_task_simple(task_id: str, bee_id: str, bee_type: str, task_category: str, success: bool, duration: float, error_message: Optional = None, capability_used: Optional = None, metadata: Optional = None) -> None
```

简化的任务记录接口

#### 属性

- `EFFICIENCY_RATIO_THRESHOLDS`
- `MIN_SAMPLE_SIZE`
- `SUCCESS_RATE_THRESHOLDS`

### GapAnalysisReport

**@dataclass**

GapAnalysisReport(report_id: str, generated_at: float, analysis_window_hours: float, total_tasks_analyzed: int, total_gaps_detected: int, gaps_by_severity: Dict[str, int], gaps_by_type: Dict[str, int], gaps_by_capability: Dict[str, int], critical_gaps: List[hive.core.capability_gap_analyzer.CapabilityGap], high_gaps: List[hive.core.capability_gap_analyzer.CapabilityGap], all_gaps: List[hive.core.capability_gap_analyzer.CapabilityGap], remediation_plan: List[Dict[str, Any]], evolutionary_insights: Dict[str, Any])

#### 方法

```python
def to_dict() -> Dict
```

### GapSeverity

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `CRITICAL`
- `HIGH`
- `LOW`
- `MEDIUM`
- `name`
- `value`

### GapType

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `CAPABILITY_MISMATCH`
- `EFFICIENCY`
- `ERROR_PATTERN`
- `KNOWLEDGE_GAP`
- `SUCCESS_RATE`
- `WORKLOAD_BOTTLENECK`
- `name`
- `value`

### TaskExecutionRecord

**@dataclass**

TaskExecutionRecord(task_id: str, bee_id: str, bee_type: str, task_category: str, success: bool, duration: float, error_message: Optional[str] = None, capability_used: Optional[str] = None, expected_duration: Optional[float] = None, timestamp: float = <factory>, metadata: Dict[str, Any] = <factory>)

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `capability_used`
- `error_message`
- `expected_duration`

## 函数

### get_capability_gap_analyzer

```python
def get_capability_gap_analyzer() -> CapabilityGapAnalyzer
```

获取能力缺口分析器单例

---

*文档生成于: 2026-04-23T01:21:36.336241*
