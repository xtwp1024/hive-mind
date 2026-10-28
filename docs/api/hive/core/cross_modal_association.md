# hive.core.cross_modal_association

Cross-Modal Association - 跨模态关联模块 Phase 4: 多模态扩展 实现不同模态之间的关联和理解。 核心功能: 1. 跨模态注意力对齐 - 使用注意力机制对齐不同模态 2. 模态关联学习 - 学习模态之间的映射关系 3. 统一表示空间 - 将所有模态映射到同一空间 4. 跨模态检索 - 用一种模态检索另一种模态 5. 模态补全 - 用一种模态补全另一种模态的缺失信息 架构设计: ``` ┌─────────────────────────────────────────────────────────┐ │              跨模态关联引擎                              │ ├─────────────────────────────────────────────────────────┤ │                                                         │ │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐│ │  │  文本嵌入   │    │  图像嵌入   │    │  音频嵌入   ││ │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘│ │         │                  │                  │         │ │         ▼                  ▼                  ▼         │ │  ┌─────────────────────────────────────────────────┐   │ │  │           跨模态注意力对齐层                     │   │ │  │  • 自注意力: 模态内关联                          │   │ │  │  • 互注意力: 模态间关联                          │   │ │  │  • 交叉注意力: 引导对齐                          │   │ │  └─────────────────────────────────────────────────┘   │ │                           │                             │ │                           ▼                             │ │  ┌─────────────────────────────────────────────────┐   │ │  │           模态关联学习器                         │   │ │  │  • 相似度度量学习                                │   │ │  │  • 对比学习                                      │   │ │  │  • 映射函数学习                                  │   │ │  └─────────────────────────────────────────────────┘   │ │                           │                             │ │                           ▼                             │ │  ┌─────────────────────────────────────────────────┐   │ │  │           统一表示空间                           │   │ │  │  • 共享嵌入空间                                  │   │ │  │  • 跨模态相似度计算                              │   │ │  │  • 语义对齐                                      │   │ │  └─────────────────────────────────────────────────┘   │ │                                                         │ └─────────────────────────────────────────────────────────┘ ```

## 目录

- [类](#类)
  - [AssociationType](#associationtype)
  - [CrossModalAssociationEngine](#crossmodalassociationengine)
  - [CrossModalAttention](#crossmodalattention)
  - [CrossModalMapping](#crossmodalmapping)
  - [CrossModalRetriever](#crossmodalretriever)
  - [ModalityAlignment](#modalityalignment)
  - [ModalityAssociationLearner](#modalityassociationlearner)
  - [RetrievalResult](#retrievalresult)
  - [UnifiedRepresentationSpace](#unifiedrepresentationspace)
- [函数](#函数)
  - [get_cross_modal_engine](#get_cross_modal_engine)

## 类

### AssociationType

**Enum**

**继承:** `Enum`

关联类型

#### 属性

- `CAUSAL`
- `CORRELATIONAL`
- `SEMANTIC`
- `SPATIAL`
- `TEMPORAL`
- `name`
- `value`

### CrossModalAssociationEngine

跨模态关联引擎 整合所有跨模态功能: 1. 注意力对齐 2. 关联学习 3. 统一表示空间 4. 跨模态检索

#### 方法

```python
def align_modalities(source_embedding: List, source_modality: str, target_embeddings: List, target_modality: str) -> Tuple
```

对齐两种模态

**参数:**

- `source_embedding`: 源嵌入
- `source_modality`: 源模态
- `target_embeddings`: 目标嵌入列表
- `target_modality`: 目标模态

**返回:** (aligned_embedding, attention_weights, best_match_index, alignment)

```python
def fuse_multiple_modalities(embeddings: List, strategy: str = weighted_average, weights: Optional = None) -> List
```

融合多种模态的嵌入

**参数:**

- `embeddings`: 嵌入列表 [(embedding, modality)]
- `strategy`: 融合策略
- `weights`: 可选权重

**返回:** 融合后的嵌入

```python
def get_stats() -> Dict
```

获取统计信息

### CrossModalAttention

跨模态注意力机制 实现多种注意力对齐方式: 1. 自注意力 (Self-Attention): 模态内关联 2. 互注意力 (Cross-Attention): 模态间关联 3. 引导注意力 (Guided Attention): 用一种模态引导另一种

#### 方法

```python
def cross_attention(source_embeddings: List, target_embeddings: List, num_heads: int = 8) -> Tuple
```

互注意力 - 计算模态间的关联

**参数:**

- `source_embeddings`: 源模态嵌入
- `target_embeddings`: 目标模态嵌入
- `num_heads`: 注意力头数

**返回:** (output_embeddings, attention_weights)

```python
def guided_attention(source_embedding: List, target_embeddings: List, guidance_weights: Optional = None) -> Tuple
```

引导注意力 - 用源模态引导目标模态

**参数:**

- `source_embedding`: 源嵌入
- `target_embeddings`: 目标嵌入列表
- `guidance_weights`: 可选指导权重

**返回:** (aligned_embedding, attention_weights, best_match_index)

```python
def self_attention(embeddings: List, num_heads: int = 8) -> Tuple
```

自注意力 - 计算模态内的关联

**参数:**

- `embeddings`: 嵌入列表
- `num_heads`: 注意力头数

**返回:** (output_embeddings, attention_weights)

### CrossModalMapping

**@dataclass**

跨模态映射

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `inverse_mapping`
- `is_learned`
- `is_reversible`
- `mapping_function`
- `training_samples`
- `validation_score`

### CrossModalRetriever

跨模态检索器 支持: 1. 文本检索图像 2. 图像检索文本 3. 音频检索文本/图像 4. 任意模态组合检索

#### 方法

```python
def cross_modal_search(query_embedding: List, query_modality: str, target_modalities: List, top_k: Optional = None) -> RetrievalResult
```

通用跨模态搜索

**参数:**

- `query_embedding`: 查询嵌入
- `query_modality`: 查询模态
- `target_modalities`: 目标模态列表
- `top_k`: 返回数量

**返回:** RetrievalResult - 检索结果

```python
def get_recent_retrievals(count: int = 10) -> List
```

获取最近的检索历史

```python
def image_to_text(image_embedding: List, top_k: Optional = None) -> RetrievalResult
```

图像检索文本

**参数:**

- `image_embedding`: 图像嵌入
- `top_k`: 返回数量

**返回:** RetrievalResult - 检索结果

```python
def text_to_image(text_query: str, top_k: Optional = None, text_encoder: Optional = None) -> RetrievalResult
```

文本检索图像

**参数:**

- `text_query`: 文本查询
- `top_k`: 返回数量
- `text_encoder`: 可选文本编码器

**返回:** RetrievalResult - 检索结果

### ModalityAlignment

**@dataclass**

模态对齐结果

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `aligned_regions`
- `alignment_type`
- `attention_weights`

### ModalityAssociationLearner

模态关联学习器 学习不同模态之间的映射关系: 1. 相似度度量学习 2. 对比学习 3. 映射函数学习

#### 方法

```python
def get_mapping(source_modality: str, target_modality: str) -> Optional
```

获取映射

```python
def learn_mapping(source_modality: str, target_modality: str, source_embeddings: List, target_embeddings: List, pairs: Optional = None) -> CrossModalMapping
```

学习模态间的映射

**参数:**

- `source_modality`: 源模态
- `target_modality`: 目标模态
- `source_embeddings`: 源嵌入列表
- `target_embeddings`: 目标嵌入列表
- `pairs`: 可选的配对索引列表

**返回:** CrossModalMapping - 学习到的映射

```python
def list_mappings() -> List
```

列出所有映射

### RetrievalResult

检索结果

#### 方法

```python
def to_dict() -> Dict
```

### UnifiedRepresentationSpace

统一表示空间 将所有模态映射到共享的嵌入空间: 1. 共享嵌入空间 2. 跨模态相似度计算 3. 语义对齐

#### 方法

```python
def add_embedding(embedding_id: str, embedding: List, modality: str, metadata: Optional = None) -> bool
```

添加嵌入到统一空间

**参数:**

- `embedding_id`: 嵌入唯一标识
- `embedding`: 嵌入向量
- `modality`: 模态类型
- `metadata`: 附加元数据

**返回:** bool - 是否成功

```python
def cross_modal_similarity(embedding_id1: str, embedding_id2: str) -> float
```

计算跨模态相似度

**参数:**

- `embedding_id1`: 第一个嵌入ID
- `embedding_id2`: 第二个嵌入ID

**返回:** float - 余弦相似度 (-1 到 1)

```python
def get_stats() -> Dict
```

获取统计信息

```python
def project_to_unified(embedding: List, source_modality: str) -> List
```

将嵌入投影到统一空间

**参数:**

- `embedding`: 原始嵌入
- `source_modality`: 源模态

**返回:** 统一空间中的嵌入

```python
def remove_embedding(embedding_id: str) -> bool
```

移除嵌入

```python
def similarity_search(query_embedding: List, query_modality: str, target_modalities: Optional = None, top_k: int = 10, min_score: float = 0.0) -> List
```

跨模态相似度搜索

**参数:**

- `query_embedding`: 查询嵌入
- `query_modality`: 查询模态
- `target_modalities`: 目标模态列表 (None 表示所有)
- `top_k`: 返回前 K 个结果
- `min_score`: 最小分数阈值

**返回:** List[(embedding_id, similarity_score, modality)]

## 函数

### get_cross_modal_engine

```python
def get_cross_modal_engine() -> CrossModalAssociationEngine
```

获取跨模态关联引擎单例

---

*文档生成于: 2026-04-23T01:21:36.593840*
