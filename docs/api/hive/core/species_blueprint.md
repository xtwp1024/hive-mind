# hive.core.species_blueprint

Species Blueprint - 蜂种蓝图系统 Phase 3: 自主进化能力 定义新蜂种的完整规格，作为自动蜂种生成的模板。 核心概念: 1. SpeciesBlueprint - 蜂种蓝图，定义新蜂种的完整规格 2. CapabilitySpecification - 能力规格（目标值、权重、约束） 3. TaskAffinity - 任务亲和度（偏好的任务类型） 4. BlueprintOrigin - 蓝图来源（手动创建/自动生成/变异而来） 5. BlueprintRegistry - 蓝图注册表，管理所有蓝图 蓝图生命周期: 1. 设计 → 2. 验证 → 3. 孵化 → 4. 评估 → 5. 晋升/淘汰

## 目录

- [类](#类)
  - [BlueprintOrigin](#blueprintorigin)
  - [BlueprintRegistry](#blueprintregistry)
  - [BlueprintStatus](#blueprintstatus)
  - [BlueprintValidationResult](#blueprintvalidationresult)
  - [CapabilityPriority](#capabilitypriority)
  - [CapabilitySpecification](#capabilityspecification)
  - [SpeciesBlueprint](#speciesblueprint)
  - [TaskAffinity](#taskaffinity)
- [函数](#函数)
  - [get_blueprint_registry](#get_blueprint_registry)

## 类

### BlueprintOrigin

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `AUTO_GENERATED`
- `CROSSED`
- `EVOLVED`
- `MANUAL`
- `MUTATED`
- `name`
- `value`

### BlueprintRegistry

蓝图注册表 管理所有蜂种蓝图，支持: - 注册/注销蓝图 - 按状态/标签/能力查询 - 版本管理 - 持久化存储

#### 方法

```python
def get(blueprint_id: str) -> Optional
```

获取蓝图

```python
def get_all() -> List
```

获取所有蓝图

```python
def get_by_capability(capability_name: str) -> List
```

按能力获取蓝图

```python
def get_by_status(status: BlueprintStatus) -> List
```

按状态获取蓝图

```python
def get_by_tag(tag: str) -> List
```

按标签获取蓝图

```python
def get_for_task(task_category: str) -> List
```

获取适合特定任务的蓝图

```python
def get_stats() -> Dict
```

获取统计信息

```python
def register(blueprint: SpeciesBlueprint) -> bool
```

注册蓝图

```python
def unregister(blueprint_id: str) -> bool
```

注销蓝图

```python
def update_status(blueprint_id: str, new_status: BlueprintStatus) -> bool
```

更新蓝图状态

### BlueprintStatus

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `ACTIVE`
- `DEPRECATED`
- `DRAFT`
- `INCUBATING`
- `REJECTED`
- `VALIDATED`
- `name`
- `value`

### BlueprintValidationResult

**@dataclass**

蓝图验证结果

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `overall_score`

### CapabilityPriority

**Enum**

**继承:** `Enum`

Create a collection of name/value pairs. Example enumeration: >>> class Color(Enum): ...     RED = 1 ...     BLUE = 2 ...     GREEN = 3 Access them by: - attribute access: >>> Color.RED <Color.RED: 1> - value lookup: >>> Color(1) <Color.RED: 1> - name lookup: >>> Color['RED'] <Color.RED: 1> Enumerations can be iterated over, and know how many members they have: >>> len(Color) 3 >>> list(Color) [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>] Methods can be added to enumerations, and members can have their own attributes -- see the documentation for details.

#### 属性

- `OPTIONAL`
- `PRIMARY`
- `SECONDARY`
- `TERTIARY`
- `name`
- `value`

### CapabilitySpecification

**@dataclass**

能力规格 定义单个能力的目标值、权重和约束条件。 用于 AutoSpeciesGenerator 生成优化的 DNA。

#### 方法

```python
def from_dict(data: Dict) -> CapabilitySpecification
```

```python
def to_dict() -> Dict
```

```python
def validate() -> Tuple
```

验证能力规格

#### 属性

- `is_mutable`
- `max_value`
- `min_value`
- `mutation_rate_override`
- `priority`
- `target_value`
- `weight`

### SpeciesBlueprint

**@dataclass**

蜂种蓝图 定义新蜂种的完整规格，包括： - 基础信息（名称、描述、版本） - 能力规格（目标值、权重、约束） - 任务亲和度（偏好的任务类型） - 遗传参数（变异率、学习率等） - 评估指标（用于判断孵化是否成功） 蓝图是 AutoSpeciesGenerator 的输入， 也是 Incubator 的孵化模板。

#### 方法

```python
def add_capability(name: str, target_value: float = 0.8, priority: CapabilityPriority = CapabilityPriority.SECONDARY, weight: float = 1.0, min_value: float = 0.3, max_value: float = 1.0, is_mutable: bool = True) -> SpeciesBlueprint
```

添加能力规格

```python
def add_task_affinity(task_category: str, affinity_score: float = 0.5, is_exclusive: bool = False) -> SpeciesBlueprint
```

添加任务亲和度

```python
def crossover(other: SpeciesBlueprint) -> Tuple
```

与另一个蓝图交叉 使用遗传算法的交叉操作， 创建两个子代蓝图。

```python
def from_dict(data: Dict) -> SpeciesBlueprint
```

从字典创建蓝图

```python
def get_preferred_task_categories() -> List
```

获取偏好的任务类别

```python
def get_primary_capabilities() -> List
```

获取主要能力

```python
def mutate(mutation_rate: float = 0.2) -> SpeciesBlueprint
```

创建变异蓝图 基于当前蓝图创建一个变异版本， 用于进化探索。

```python
def to_dict() -> Dict
```

转换为字典

```python
def to_dna_params() -> Dict
```

转换为 DNA 参数 从蓝图生成 BeeDNA 的初始化参数。

```python
def validate() -> BlueprintValidationResult
```

验证蓝图是否有效 检查: 1. 必填字段是否存在 2. 能力规格是否有效 3. 任务亲和度是否有效 4. 遗传参数是否在合理范围 5. 成功标准是否合理

**返回:** BlueprintValidationResult - 验证结果

#### 属性

- `base_species`
- `generation`
- `immune_response`
- `learning_rate`
- `mutation_rate`
- `origin`
- `role`
- `status`
- `validated_at`
- `version`

### TaskAffinity

**@dataclass**

任务亲和度 定义蜂种偏好的任务类型，用于任务分配优化。

#### 方法

```python
def from_dict(data: Dict) -> TaskAffinity
```

```python
def to_dict() -> Dict
```

```python
def validate() -> Tuple
```

验证任务亲和度

#### 属性

- `affinity_score`
- `is_exclusive`
- `max_task_complexity`
- `min_task_complexity`
- `preferred_task_size`

## 函数

### get_blueprint_registry

```python
def get_blueprint_registry() -> BlueprintRegistry
```

获取蓝图注册表单例

---

*文档生成于: 2026-04-23T01:21:36.390958*
