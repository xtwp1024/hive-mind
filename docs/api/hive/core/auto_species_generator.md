# hive.core.auto_species_generator

Auto Species Generator - 自动蜂种生成器 Phase 3: 自主进化能力 基于能力缺口分析自动生成新的蜂种蓝图。 核心功能: 1. 缺口分析驱动 - 从 CapabilityGapAnalyzer 获取进化需求 2. 蓝图模板系统 - 基于现有成功蓝图创建新蓝图 3. 参数优化 - 遗传算法优化蓝图参数 4. 蓝图验证 - 自动验证生成的蓝图 5. 孵化调度 - 将验证通过的蓝图提交给 Incubator 工作流程: 1. 收集缺口分析 → 2. 确定进化目标 → 3. 生成候选蓝图 → 4. 优化筛选 → 5. 验证 → 6. 注册/孵化

## 目录

- [类](#类)
  - [AutoSpeciesGenerator](#autospeciesgenerator)
  - [BlueprintOptimizer](#blueprintoptimizer)
  - [EvolutionTarget](#evolutiontarget)
  - [GenerationResult](#generationresult)
  - [GenerationStrategy](#generationstrategy)
  - [OptimizationObjective](#optimizationobjective)
- [函数](#函数)
  - [get_auto_species_generator](#get_auto_species_generator)

## 类

### AutoSpeciesGenerator

自动蜂种生成器 Phase 3 核心组件。基于能力缺口分析自动生成新的蜂种蓝图。 主要功能: 1. 从 CapabilityGapAnalyzer 获取进化需求 2. 基于现有蓝图生成新蓝图（变异/交叉/混合） 3. 使用遗传算法优化蓝图参数 4. 自动验证生成的蓝图 5. 将验证通过的蓝图注册到 BlueprintRegistry 使用方式: generator = AutoSpeciesGenerator() result = generator.generate_from_gap_analysis( gap_analyzer, max_blueprints=3 )

#### 方法

```python
def generate_from_gap_analysis(gap_analyzer: CapabilityGapAnalyzer, max_blueprints: int = 3, strategy: GenerationStrategy = GenerationStrategy.HYBRID, analysis_window_hours: float = 24.0) -> GenerationResult
```

基于能力缺口分析生成新蜂种

**参数:**

- `gap_analyzer`: 能力缺口分析器
- `max_blueprints`: 最大生成蓝图数量
- `strategy`: 生成策略
- `analysis_window_hours`: 分析时间窗口

**返回:** GenerationResult - 生成结果

```python
def get_generation_history(limit: int = 10) -> List
```

获取生成历史

```python
def get_stats() -> Dict
```

获取统计信息

```python
def optimize_blueprint(blueprint: SpeciesBlueprint, objectives: Optional = None) -> SpeciesBlueprint
```

优化现有蓝图

### BlueprintOptimizer

蓝图优化器 使用遗传算法优化蓝图参数。

#### 方法

```python
def optimize(base_blueprint: SpeciesBlueprint, objectives: List, evaluator) -> SpeciesBlueprint
```

优化蓝图

**参数:**

- `base_blueprint`: 基础蓝图
- `objectives`: 优化目标列表
- `evaluator`: 评估函数，接收蓝图返回适应度分数

**返回:** 优化后的蓝图

### EvolutionTarget

**@dataclass**

进化目标

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `current_efficiency`
- `priority`
- `source_gap_id`
- `target_efficiency`
- `urgency`

### GenerationResult

**@dataclass**

生成结果

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `end_time`

### GenerationStrategy

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `CROSSOVER`
- `HYBRID`
- `MUTATION`
- `NOVEL`
- `name`
- `value`

### OptimizationObjective

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `BALANCED`
- `EFFICIENCY`
- `ROBUSTNESS`
- `SUCCESS_RATE`
- `name`
- `value`

## 函数

### get_auto_species_generator

```python
def get_auto_species_generator() -> AutoSpeciesGenerator
```

获取自动蜂种生成器单例

---

*文档生成于: 2026-04-23T01:21:36.417274*
