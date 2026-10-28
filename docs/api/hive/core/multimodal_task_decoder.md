# hive.core.multimodal_task_decoder

Multimodal Task Decoder - 多模态任务解码器 Phase 4: 多模态扩展 根据融合后的多模态表示执行具体任务。 核心功能: 1. 任务类型识别 - 自动识别任务类型 2. 融合表示解码 - 将统一嵌入解码为具体输出 3. 多模态输出生成 - 支持文本、代码、结构化数据等输出 4. 跨模态推理 - 基于多模态信息进行推理 架构设计: ``` ┌─────────────────────────────────────────────────────────┐ │              多模态任务解码器                            │ ├─────────────────────────────────────────────────────────┤ │                                                         │ │  ┌─────────────────────────────────────────────────┐   │ │  │           任务类型识别器                         │   │ │  │  • 从融合表示推断任务类型                        │   │ │  │  • 支持问答、分类、摘要、生成等                  │   │ │  └─────────────────────────────────────────────────┘   │ │                           │                             │ │                           ▼                             │ │  ┌─────────────────────────────────────────────────┐   │ │  │           融合表示解码器                         │   │ │  │  • 将统一嵌入解码为结构化输出                    │   │ │  │  • 支持文本、代码、JSON 等格式                  │   │ │  └─────────────────────────────────────────────────┘   │ │                           │                             │ │                           ▼                             │ │  ┌─────────────────────────────────────────────────┐   │ │  │           跨模态推理引擎                         │   │ │  │  • 基于多模态信息进行推理                        │   │ │  │  • 支持因果推理、类比推理等                      │   │ │  └─────────────────────────────────────────────────┘   │ │                                                         │ └─────────────────────────────────────────────────────────┘ ```

## 目录

- [类](#类)
  - [CrossModalReasoningEngine](#crossmodalreasoningengine)
  - [DecodedOutput](#decodedoutput)
  - [FusionDecoder](#fusiondecoder)
  - [MultimodalTaskDecoder](#multimodaltaskdecoder)
  - [OutputFormat](#outputformat)
  - [ReasoningStep](#reasoningstep)
  - [TaskContext](#taskcontext)
  - [TaskType](#tasktype)
  - [TaskTypeClassifier](#tasktypeclassifier)
- [函数](#函数)
  - [get_multimodal_decoder](#get_multimodal_decoder)

## 类

### CrossModalReasoningEngine

跨模态推理引擎 基于多模态信息进行复杂推理。

#### 方法

```python
def analogical_reasoning(source_inputs: List, target_inputs: List, question: str = 进行类比推理) -> Dict
```

类比推理

```python
def causal_reasoning(cause_inputs: List, effect_inputs: List, question: str = 分析因果关系) -> Dict
```

因果推理

```python
def reason(question: str, modality_inputs: Dict, reasoning_type: str = comprehensive) -> Dict
```

执行跨模态推理

**参数:**

- `question`: 推理问题
- `modality_inputs`: 各模态输入 {modality: [inputs]}
- `reasoning_type`: 推理类型

**返回:** 推理结果字典

### DecodedOutput

**@dataclass**

解码输出结果

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `confidence`
- `content`
- `processing_time_ms`
- `text_content`

### FusionDecoder

融合表示解码器 将统一嵌入解码为具体输出。

#### 方法

```python
def decode(unified_embedding: List, task_context: TaskContext, raw_inputs: Optional = None) -> DecodedOutput
```

解码统一嵌入为输出

**参数:**

- `unified_embedding`: 统一嵌入向量
- `task_context`: 任务上下文
- `raw_inputs`: 原始输入（文本、图像路径等）

**返回:** DecodedOutput - 解码结果

### MultimodalTaskDecoder

多模态任务解码器 整合任务分类、融合解码、跨模态推理功能。

#### 方法

```python
def decode_task(prompt: str, unified_embedding: List, input_modalities: List, raw_inputs: Optional = None, system_prompt: Optional = None, expected_output_format: Optional = None, expected_task_type: Optional = None, max_tokens: int = 4096, temperature: float = 0.7) -> DecodedOutput
```

解码任务

**参数:**

- `prompt`: 任务提示
- `unified_embedding`: 统一嵌入
- `input_modalities`: 输入模态列表
- `raw_inputs`: 原始输入字典 {"text": [...], "images": [...], "audio": [...]}
- `system_prompt`: 系统提示
- `expected_output_format`: 期望的输出格式
- `expected_task_type`: 期望的任务类型
- `max_tokens`: 最大令牌数
- `temperature`: 温度参数

**返回:** DecodedOutput - 解码输出

```python
def extract_knowledge_from_multimodal(text_inputs: Optional = None, image_inputs: Optional = None, audio_inputs: Optional = None, focus: Optional = None) -> Dict
```

从多模态输入提取知识

**参数:**

- `text_inputs`: 文本输入
- `image_inputs`: 图像输入
- `audio_inputs`: 音频输入
- `focus`: 关注重点

**返回:** 知识提取结果

```python
def generate_code_from_multimodal(requirement: str, text_inputs: Optional = None, image_inputs: Optional = None, language: str = python) -> Dict
```

从多模态输入生成代码

**参数:**

- `requirement`: 需求描述
- `text_inputs`: 文本输入（如文档、规范）
- `image_inputs`: 图像输入（如架构图、UI设计图）
- `language`: 目标编程语言

**返回:** 代码生成结果

```python
def get_stats() -> Dict
```

获取统计信息

```python
def reason_with_modalities(question: str, text_inputs: Optional = None, image_inputs: Optional = None, audio_inputs: Optional = None, reasoning_type: str = comprehensive) -> Dict
```

使用多模态输入进行推理

**参数:**

- `question`: 推理问题
- `text_inputs`: 文本输入列表
- `image_inputs`: 图像输入列表
- `audio_inputs`: 音频输入列表
- `reasoning_type`: 推理类型

**返回:** 推理结果字典

### OutputFormat

**Enum**

**继承:** `Enum`

输出格式

#### 属性

- `CODE`
- `HTML`
- `JSON`
- `MARKDOWN`
- `PLAIN_TEXT`
- `name`
- `value`

### ReasoningStep

**@dataclass**

推理步骤

#### 属性

- `confidence`
- `evidence`

### TaskContext

**@dataclass**

任务上下文

#### 属性

- `expected_output_format`
- `expected_task_type`
- `max_tokens`
- `prompt`
- `system_prompt`
- `temperature`

### TaskType

**Enum**

**继承:** `Enum`

任务类型

#### 属性

- `AUDIO_TRANSCRIPTION`
- `AUDIO_UNDERSTANDING`
- `CODE_ANALYSIS`
- `CODE_GENERATION`
- `IMAGE_CAPTIONING`
- `IMAGE_UNDERSTANDING`
- `KNOWLEDGE_EXTRACTION`
- `MULTIMODAL_REASONING`
- `QUESTION_ANSWERING`
- `TEXT_CLASSIFICATION`
- `TEXT_GENERATION`
- `TEXT_SUMMARIZATION`
- `UNKNOWN`
- `name`
- `value`

### TaskTypeClassifier

任务类型分类器 根据输入特征自动识别任务类型。

#### 方法

```python
def classify(prompt: str, input_modalities: Optional = None) -> Tuple
```

分类任务类型

**参数:**

- `prompt`: 任务提示
- `input_modalities`: 输入模态列表

**返回:** (TaskType, confidence)

```python
def detect_output_format(prompt: str, task_type: TaskType) -> OutputFormat
```

检测期望的输出格式

**参数:**

- `prompt`: 任务提示
- `task_type`: 检测到的任务类型

**返回:** OutputFormat

## 函数

### get_multimodal_decoder

```python
def get_multimodal_decoder() -> MultimodalTaskDecoder
```

获取多模态任务解码器单例

---

*文档生成于: 2026-04-23T01:21:36.612651*
