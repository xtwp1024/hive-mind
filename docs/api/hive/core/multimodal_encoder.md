# hive.core.multimodal_encoder

Multimodal Encoder - 多模态编码器 Phase 4: 多模态扩展 统一编码文本、图像、音频等多种模态的输入。 核心功能: 1. 文本编码器 - 使用 BERT/GPT 等模型编码文本 2. 图像编码器 - 使用 CLIP/Vision Transformer 编码图像 3. 音频编码器 - 使用 Whisper/Wav2Vec 编码音频 4. 统一表示空间 - 将不同模态映射到同一向量空间 5. 模态类型检测 - 自动识别输入模态类型 编码器架构: ``` ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ │  文本输入    │     │  图像输入    │     │  音频输入    │ └──────┬──────┘     └──────┬──────┘     └──────┬──────┘ │                   │                   │ ▼                   ▼                   ▼ ┌─────────────────────────────────────────────────┐ │              模态编码器 (Modal Encoder)          │ │  • 文本: BERT/GPT Embeddings                     │ │  • 图像: CLIP/Vision Transformer                 │ │  • 音频: Whisper/Wav2Vec                         │ └─────────────────────────────────────────────────┘ │ ▼ ┌─────────────────────────────────────────────────┐ │              投影层 (Projection Layer)           │ │  • 将不同模态映射到统一维度                        │ │  • 可选: 模态特定的投影网络                        │ └─────────────────────────────────────────────────┘ │ ▼ ┌─────────────────────────────────────────────────┐ │              统一表示空间 (Unified Space)         │ │  • 所有模态共享同一向量空间                        │ │  • 支持跨模态相似度计算                            │ └─────────────────────────────────────────────────┘ ```

## 目录

- [类](#类)
  - [AudioEncoder](#audioencoder)
  - [EncoderType](#encodertype)
  - [EncodingResult](#encodingresult)
  - [ImageEncoder](#imageencoder)
  - [ModalityEmbedding](#modalityembedding)
  - [ModalityType](#modalitytype)
  - [MultimodalEncoder](#multimodalencoder)
  - [TextEncoder](#textencoder)
  - [UnifiedEmbedding](#unifiedembedding)
- [函数](#函数)
  - [get_multimodal_encoder](#get_multimodal_encoder)

## 类

### AudioEncoder

音频编码器 编码音频为文本和向量表示: - BrainInterface (transcribe_audio) - Ollama (whisper 等模型)

#### 方法

```python
def encode(audio_path: str, language: Optional = None, prompt: Optional = None, model: Optional = None) -> EncodingResult
```

编码音频

**参数:**

- `audio_path`: 音频文件路径
- `language`: 语言代码
- `prompt`: 转录提示
- `model`: 模型覆盖

**返回:** EncodingResult - 编码结果

```python
def transcribe_and_summarize(audio_path: str, language: Optional = None) -> Dict
```

转录音频并生成总结

**参数:**

- `audio_path`: 音频路径
- `language`: 语言

**返回:** 包含转录和总结的字典

### EncoderType

**Enum**

**继承:** `Enum`

编码器类型

#### 属性

- `BERT`
- `BRAIN_INTERFACE`
- `CLIP`
- `GPT`
- `OLLAMA`
- `UNIFIED`
- `VIT`
- `WAV2VEC`
- `WHISPER`
- `name`
- `value`

### EncodingResult

**@dataclass**

编码结果

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `embedding`
- `error_details`
- `message`
- `modality_type`
- `raw_response`

### ImageEncoder

图像编码器 编码图像为向量表示: - BrainInterface (consult_vision) - Ollama (llava 等多模态模型) - 支持 base64 和文件路径输入

#### 方法

```python
def analyze_architecture(image_path: str, focus: str = components_and_relationships) -> Dict
```

分析架构图

**参数:**

- `image_path`: 架构图路径
- `focus`: 分析重点

**返回:** 包含组件和关系的字典

```python
def analyze_screenshot(image_path: str, focus: str = ui_layout) -> Dict
```

分析截图/UI界面

**参数:**

- `image_path`: 截图路径
- `focus`: 分析重点

**返回:** UI 分析结果

```python
def encode(image_input: Union, prompt: str = Describe this image in detail, model: Optional = None) -> EncodingResult
```

编码图像

**参数:**

- `image_input`: 图像路径或 base64 字符串
- `prompt`: 用于分析的提示
- `model`: 可选模型覆盖

**返回:** EncodingResult - 编码结果

### ModalityEmbedding

**@dataclass**

模态嵌入表示

#### 方法

```python
def similarity(other: ModalityEmbedding) -> float
```

计算与另一个嵌入的余弦相似度

```python
def to_dict() -> Dict
```

#### 属性

- `embedding_dim`
- `normalized_embedding`
- `processing_time_ms`
- `raw_input`
- `token_count`
- `tokens`

### ModalityType

**Enum**

**继承:** `Enum`

模态类型

#### 属性

- `AUDIO`
- `IMAGE`
- `MULTIMODAL`
- `TEXT`
- `UNKNOWN`
- `VIDEO`
- `name`
- `value`

### MultimodalEncoder

多模态编码器 统一编码多种模态的输入，并提供: 1. 模态类型自动检测 2. 多种编码器调度 3. 统一嵌入空间 4. 跨模态相似度计算

#### 方法

```python
def cross_modal_similarity(emb1: ModalityEmbedding, emb2: ModalityEmbedding) -> float
```

计算跨模态相似度

**参数:**

- `emb1`: 第一个嵌入
- `emb2`: 第二个嵌入

**返回:** float - 余弦相似度 (-1 到 1)

```python
def detect_modality(input_data: Any, metadata: Optional = None) -> ModalityType
```

自动检测模态类型

**参数:**

- `input_data`: 输入数据
- `metadata`: 元数据

**返回:** ModalityType - 检测到的模态类型

```python
def encode(input_data: Any, modality: Optional = None, metadata: Optional = None, kwargs) -> EncodingResult
```

编码输入数据

**参数:**

- `input_data`: 输入数据（文本、图像路径、音频路径等）
- `modality`: 可选模态类型覆盖
- `metadata`: 元数据
- `**kwargs`: 传递给具体编码器的参数

**返回:** EncodingResult - 编码结果

```python
def encode_batch(inputs: List) -> List
```

批量编码

```python
def fuse_embeddings(embeddings: List, strategy: str = weighted_average, weights: Optional = None) -> ModalityEmbedding
```

融合多个嵌入

**参数:**

- `embeddings`: 嵌入列表
- `strategy`: 融合策略 ("concatenation", "average", "weighted_average")
- `weights`: 可选权重字典

**返回:** ModalityEmbedding - 融合后的嵌入

```python
def get_encoder_for_modality(modality: ModalityType) -> Any
```

获取对应模态的编码器

### TextEncoder

文本编码器 使用各种模型编码文本: - BrainInterface (默认) - Ollama - 本地嵌入模型

#### 方法

```python
def encode(text: str, model: Optional = None, max_tokens: int = 8192) -> EncodingResult
```

编码文本

**参数:**

- `text`: 输入文本
- `model`: 可选模型覆盖
- `max_tokens`: 最大令牌数

**返回:** EncodingResult - 编码结果

```python
def encode_batch(texts: List, model: Optional = None) -> List
```

批量编码文本

### UnifiedEmbedding

**@dataclass**

统一嵌入表示

#### 方法

```python
def to_dict() -> Dict
```

#### 属性

- `embedding_dim`
- `fusion_strategy`
- `fusion_weights`

## 函数

### get_multimodal_encoder

```python
def get_multimodal_encoder() -> MultimodalEncoder
```

获取多模态编码器单例

---

*文档生成于: 2026-04-23T01:21:36.484089*
