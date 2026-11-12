# -*- coding: utf-8 -*-
"""
Phase 4: 多模态扩展 - 单元测试

测试以下模块:
1. MultimodalEncoder - 多模态编码器
   - TextEncoder - 文本编码器
   - ImageEncoder - 图像编码器
   - AudioEncoder - 音频编码器
   - MultimodalEncoder - 统一编码器

2. CrossModalAssociation - 跨模态关联
   - CrossModalAttention - 跨模态注意力
   - UnifiedRepresentationSpace - 统一表示空间
   - CrossModalRetriever - 跨模态检索器

3. MultimodalTaskDecoder - 多模态任务解码器
   - TaskTypeClassifier - 任务类型分类器
   - FusionDecoder - 融合表示解码器
   - CrossModalReasoningEngine - 跨模态推理引擎
"""

import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTextEncoder(unittest.TestCase):
    """测试文本编码器"""

    def setUp(self):
        from hive.core.multimodal_encoder import EncoderType, ModalityType, TextEncoder

        self.TextEncoder = TextEncoder
        self.ModalityType = ModalityType
        self.EncoderType = EncoderType

        self.encoder = self.TextEncoder()

    def test_encode_basic(self):
        """测试基本文本编码"""
        result = self.encoder.encode("Hello, this is a test text.")

        self.assertTrue(result.success)
        self.assertEqual(result.modality_type, self.ModalityType.TEXT)
        self.assertIsNotNone(result.embedding)
        self.assertGreater(result.embedding.embedding_dim, 0)

    def test_encode_empty_text(self):
        """测试空文本编码"""
        result = self.encoder.encode("")

        self.assertFalse(result.success)

    def test_encode_batch(self):
        """测试批量编码"""
        texts = ["Text one", "Text two", "Text three"]

        results = self.encoder.encode_batch(texts)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.success)

    def test_embedding_similarity(self):
        """测试嵌入相似度"""
        result1 = self.encoder.encode("The cat is sitting on the mat")
        result2 = self.encoder.encode("A cat is on the floor")
        result3 = self.encoder.encode("Something completely different here")

        sim12 = result1.embedding.similarity(result2.embedding)
        sim13 = result1.embedding.similarity(result3.embedding)

        self.assertIsInstance(sim12, float)
        self.assertIsInstance(sim13, float)


class TestImageEncoder(unittest.TestCase):
    """测试图像编码器"""

    def setUp(self):
        from hive.core.multimodal_encoder import EncoderType, ImageEncoder, ModalityType

        self.ImageEncoder = ImageEncoder
        self.ModalityType = ModalityType
        self.EncoderType = EncoderType

        self.encoder = self.ImageEncoder()

    def test_methods_exist(self):
        """测试核心方法存在"""
        self.assertTrue(hasattr(self.encoder, "encode"))
        self.assertTrue(hasattr(self.encoder, "analyze_architecture"))
        self.assertTrue(hasattr(self.encoder, "analyze_screenshot"))

    def test_analyze_architecture_method(self):
        """测试架构图分析方法存在"""
        self.assertTrue(callable(self.encoder.analyze_architecture))

    def test_analyze_screenshot_method(self):
        """测试截图分析方法存在"""
        self.assertTrue(callable(self.encoder.analyze_screenshot))

    def test_nonexistent_image(self):
        """测试不存在的图像"""
        result = self.encoder.encode("/nonexistent/path/to/image.png")

        self.assertFalse(result.success)


class TestAudioEncoder(unittest.TestCase):
    """测试音频编码器"""

    def setUp(self):
        from hive.core.multimodal_encoder import AudioEncoder, EncoderType, ModalityType

        self.AudioEncoder = AudioEncoder
        self.ModalityType = ModalityType
        self.EncoderType = EncoderType

        self.encoder = self.AudioEncoder()

    def test_methods_exist(self):
        """测试核心方法存在"""
        self.assertTrue(hasattr(self.encoder, "encode"))
        self.assertTrue(hasattr(self.encoder, "transcribe_and_summarize"))

    def test_transcribe_and_summarize_method(self):
        """测试转录总结方法存在"""
        self.assertTrue(callable(self.encoder.transcribe_and_summarize))

    def test_nonexistent_audio(self):
        """测试不存在的音频"""
        result = self.encoder.encode("/nonexistent/path/to/audio.mp3")

        self.assertFalse(result.success)


class TestMultimodalEncoder(unittest.TestCase):
    """测试多模态统一编码器"""

    def setUp(self):
        from hive.core.multimodal_encoder import (
            EncoderType,
            ModalityEmbedding,
            ModalityType,
            MultimodalEncoder,
        )

        self.MultimodalEncoder = MultimodalEncoder
        self.ModalityType = ModalityType
        self.EncoderType = EncoderType
        self.ModalityEmbedding = ModalityEmbedding

        self.encoder = self.MultimodalEncoder(unified_dim=128)

    def test_detect_modality_text(self):
        """测试文本模态检测"""
        modality = self.encoder.detect_modality("Hello world")
        self.assertEqual(modality, self.ModalityType.UNKNOWN)

    def test_detect_modality_with_metadata(self):
        """测试带元数据的模态检测"""
        modality = self.encoder.detect_modality("Hello", metadata={"modality": "text"})
        self.assertEqual(modality, self.ModalityType.TEXT)

    def test_detect_modality_mime_type(self):
        """测试 MIME 类型模态检测"""
        modality = self.encoder.detect_modality("test", metadata={"mime_type": "image/png"})
        self.assertEqual(modality, self.ModalityType.IMAGE)

    def test_encode_text(self):
        """测试文本编码"""
        result = self.encoder.encode("Hello world", self.ModalityType.TEXT)

        self.assertTrue(result.success)
        self.assertEqual(result.modality_type, self.ModalityType.TEXT)

    def test_fuse_embeddings_average(self):
        """测试平均融合"""
        emb1 = self.ModalityEmbedding(
            embedding_id="test1",
            modality_type=self.ModalityType.TEXT,
            encoder_type=self.EncoderType.UNIFIED,
            embedding=[0.1, 0.2, 0.3, 0.4],
            embedding_dim=4,
        )

        emb2 = self.ModalityEmbedding(
            embedding_id="test2",
            modality_type=self.ModalityType.IMAGE,
            encoder_type=self.EncoderType.UNIFIED,
            embedding=[0.5, 0.6, 0.7, 0.8],
            embedding_dim=4,
        )

        fused = self.encoder.fuse_embeddings([emb1, emb2], strategy="average")

        self.assertIsNotNone(fused)
        self.assertEqual(fused.embedding_dim, 4)

    def test_fuse_embeddings_weighted(self):
        """测试加权融合"""
        emb1 = self.ModalityEmbedding(
            embedding_id="test1",
            modality_type=self.ModalityType.TEXT,
            encoder_type=self.EncoderType.UNIFIED,
            embedding=[0.1, 0.2, 0.3, 0.4],
            embedding_dim=4,
        )

        emb2 = self.ModalityEmbedding(
            embedding_id="test2",
            modality_type=self.ModalityType.IMAGE,
            encoder_type=self.EncoderType.UNIFIED,
            embedding=[0.5, 0.6, 0.7, 0.8],
            embedding_dim=4,
        )

        fused = self.encoder.fuse_embeddings(
            [emb1, emb2], strategy="weighted_average", weights={"text": 2.0, "image": 1.0}
        )

        self.assertIsNotNone(fused)

    def test_fuse_embeddings_concatenation(self):
        """测试拼接融合"""
        emb1 = self.ModalityEmbedding(
            embedding_id="test1",
            modality_type=self.ModalityType.TEXT,
            encoder_type=self.EncoderType.UNIFIED,
            embedding=[0.1, 0.2],
            embedding_dim=2,
        )

        emb2 = self.ModalityEmbedding(
            embedding_id="test2",
            modality_type=self.ModalityType.IMAGE,
            encoder_type=self.EncoderType.UNIFIED,
            embedding=[0.5, 0.6],
            embedding_dim=2,
        )

        fused = self.encoder.fuse_embeddings([emb1, emb2], strategy="concatenation")

        self.assertEqual(fused.embedding_dim, 4)

    def test_get_encoder_for_modality(self):
        """测试获取对应模态的编码器"""
        text_encoder = self.encoder.get_encoder_for_modality(self.ModalityType.TEXT)
        self.assertIsNotNone(text_encoder)

        image_encoder = self.encoder.get_encoder_for_modality(self.ModalityType.IMAGE)
        self.assertIsNotNone(image_encoder)

        audio_encoder = self.encoder.get_encoder_for_modality(self.ModalityType.AUDIO)
        self.assertIsNotNone(audio_encoder)


class TestCrossModalAttention(unittest.TestCase):
    """测试跨模态注意力"""

    def setUp(self):
        from hive.core.cross_modal_association import CrossModalAttention

        self.CrossModalAttention = CrossModalAttention

        self.attention = self.CrossModalAttention(embedding_dim=4)

    def test_self_attention(self):
        """测试自注意力"""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [-0.1, -0.2, -0.3, -0.4],
        ]

        output, weights = self.attention.self_attention(embeddings)

        self.assertEqual(len(output), 3)
        self.assertEqual(len(weights), 3)
        self.assertEqual(len(weights[0]), 3)

    def test_self_attention_empty(self):
        """测试空自注意力"""
        output, weights = self.attention.self_attention([])
        self.assertEqual(len(output), 0)
        self.assertEqual(len(weights), 0)

    def test_cross_attention(self):
        """测试互注意力"""
        source = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]

        target = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

        output, weights = self.attention.cross_attention(source, target)

        self.assertEqual(len(output), 2)
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(weights[0]), 3)

    def test_guided_attention(self):
        """测试引导注意力"""
        source = [0.1, 0.2, 0.3, 0.4]
        targets = [
            [0.5, 0.6, 0.7, 0.8],
            [-0.1, -0.2, -0.3, -0.4],
            [0.9, 0.8, 0.7, 0.6],
        ]

        aligned, weights, best_idx = self.attention.guided_attention(
            source_embedding=source,
            target_embeddings=targets,
        )

        self.assertEqual(len(aligned), 4)
        self.assertEqual(len(weights), 3)
        self.assertIsInstance(best_idx, int)
        self.assertGreaterEqual(best_idx, 0)
        self.assertLess(best_idx, 3)

    def test_guided_attention_with_weights(self):
        """测试带权重的引导注意力"""
        source = [0.1, 0.2, 0.3, 0.4]
        targets = [
            [0.5, 0.6, 0.7, 0.8],
            [-0.1, -0.2, -0.3, -0.4],
        ]

        aligned, weights, best_idx = self.attention.guided_attention(
            source_embedding=source,
            target_embeddings=targets,
            guidance_weights=[0.8, 0.2],
        )

        self.assertEqual(len(aligned), 4)

    def test_guided_attention_empty_targets(self):
        """测试空目标的引导注意力"""
        source = [0.1, 0.2, 0.3, 0.4]

        aligned, weights, best_idx = self.attention.guided_attention(
            source_embedding=source,
            target_embeddings=[],
        )

        self.assertEqual(aligned, source)
        self.assertEqual(len(weights), 0)
        self.assertEqual(best_idx, -1)


class TestUnifiedRepresentationSpace(unittest.TestCase):
    """测试统一表示空间"""

    def setUp(self):
        from hive.core.cross_modal_association import UnifiedRepresentationSpace

        self.UnifiedRepresentationSpace = UnifiedRepresentationSpace

        self.space = self.UnifiedRepresentationSpace(dimension=128)

    def test_project_to_unified_same_dim(self):
        """测试相同维度的投影"""
        embedding = [0.1] * 128

        projected = self.space.project_to_unified(embedding, "text")

        self.assertEqual(len(projected), 128)

    def test_project_to_unified_larger_dim(self):
        """测试更大维度的投影"""
        embedding = [0.1] * 256

        projected = self.space.project_to_unified(embedding, "text")

        self.assertEqual(len(projected), 128)

    def test_project_to_unified_smaller_dim(self):
        """测试更小维度的投影"""
        embedding = [0.1] * 64

        projected = self.space.project_to_unified(embedding, "text")

        self.assertEqual(len(projected), 128)

    def test_add_embedding(self):
        """测试添加嵌入"""
        result = self.space.add_embedding(
            embedding_id="test_001",
            embedding=[0.1] * 64,
            modality="text",
            metadata={"source": "test"},
        )

        self.assertTrue(result)

    def test_remove_embedding(self):
        """测试移除嵌入"""
        self.space.add_embedding("to_remove", [0.1] * 64, "text")

        result = self.space.remove_embedding("to_remove")
        self.assertTrue(result)

        result2 = self.space.remove_embedding("nonexistent")
        self.assertFalse(result2)

    def test_similarity_search(self):
        """测试相似性搜索"""
        self.space.add_embedding("doc1", [0.1] * 64, "text", {"label": "cat"})
        self.space.add_embedding("doc2", [0.2] * 64, "image", {"label": "dog"})
        self.space.add_embedding("doc3", [0.15] * 64, "text", {"label": "bird"})

        results = self.space.similarity_search(
            query_embedding=[0.12] * 64,
            query_modality="text",
            top_k=2,
        )

        self.assertEqual(len(results), 2)

    def test_similarity_search_with_modality_filter(self):
        """测试带模态过滤的相似性搜索"""
        self.space.add_embedding("text1", [0.1] * 64, "text")
        self.space.add_embedding("image1", [0.1] * 64, "image")
        self.space.add_embedding("text2", [0.2] * 64, "text")

        results = self.space.similarity_search(
            query_embedding=[0.15] * 64,
            query_modality="text",
            target_modalities=["text"],
            top_k=10,
        )

        for result in results:
            self.assertEqual(result[2], "text")

    def test_cross_modal_similarity(self):
        """测试跨模态相似度"""
        self.space.add_embedding("text1", [0.1, 0.2, 0.3, 0.4], "text")
        self.space.add_embedding("image1", [0.1, 0.2, 0.3, 0.5], "image")

        sim = self.space.cross_modal_similarity("text1", "image1")

        self.assertIsInstance(sim, float)

    def test_get_stats(self):
        """测试获取统计信息"""
        self.space.add_embedding("t1", [0.1] * 64, "text")
        self.space.add_embedding("i1", [0.2] * 64, "image")

        stats = self.space.get_stats()

        self.assertIn("total_embeddings", stats)
        self.assertEqual(stats["total_embeddings"], 2)
        self.assertIn("dimension", stats)
        self.assertIn("by_modality", stats)


class TestCrossModalRetriever(unittest.TestCase):
    """测试跨模态检索器"""

    def setUp(self):
        from hive.core.cross_modal_association import (
            CrossModalRetriever,
            UnifiedRepresentationSpace,
        )

        self.CrossModalRetriever = CrossModalRetriever
        self.UnifiedRepresentationSpace = UnifiedRepresentationSpace

        self.space = self.UnifiedRepresentationSpace(dimension=128)
        self.space.add_embedding("img_cat", [0.1] * 128, "image", {"label": "cat"})
        self.space.add_embedding("img_dog", [0.2] * 128, "image", {"label": "dog"})
        self.space.add_embedding("text_animal", [0.15] * 128, "text", {"label": "animal"})

        self.retriever = self.CrossModalRetriever(self.space, top_k_default=5)

    def test_text_to_image(self):
        """测试文本检索图像"""
        result = self.retriever.text_to_image("Find images of cats")

        self.assertIsNotNone(result)
        self.assertEqual(result.query_modality, "text")

    def test_image_to_text(self):
        """测试图像检索文本"""
        result = self.retriever.image_to_text([0.15] * 128)

        self.assertIsNotNone(result)
        self.assertEqual(result.query_modality, "image")

    def test_cross_modal_search(self):
        """测试通用跨模态搜索"""
        result = self.retriever.cross_modal_search(
            query_embedding=[0.15] * 128,
            query_modality="text",
            target_modalities=["image", "text"],
            top_k=3,
        )

        self.assertIsNotNone(result)

    def test_get_recent_retrievals(self):
        """测试获取最近检索历史"""
        self.retriever.text_to_image("test query 1")
        self.retriever.text_to_image("test query 2")

        recent = self.retriever.get_recent_retrievals(count=2)

        self.assertEqual(len(recent), 2)


class TestTaskTypeClassifier(unittest.TestCase):
    """测试任务类型分类器"""

    def setUp(self):
        from hive.core.multimodal_task_decoder import OutputFormat, TaskType, TaskTypeClassifier

        self.TaskTypeClassifier = TaskTypeClassifier
        self.TaskType = TaskType
        self.OutputFormat = OutputFormat

        self.classifier = self.TaskTypeClassifier()

    def test_classify_code_generation(self):
        """测试代码生成任务分类"""
        task_type, confidence = self.classifier.classify(
            "请帮我写一个 Python 函数来计算斐波那契数列"
        )

        self.assertEqual(task_type, self.TaskType.CODE_GENERATION)

    def test_classify_summarization(self):
        """测试摘要任务分类"""
        task_type, confidence = self.classifier.classify("总结以下内容的要点")

        self.assertEqual(task_type, self.TaskType.TEXT_SUMMARIZATION)

    def test_classify_question_answering(self):
        """测试问答任务分类"""
        task_type, confidence = self.classifier.classify("什么是 Python 中的列表推导式？")

        self.assertEqual(task_type, self.TaskType.QUESTION_ANSWERING)

    def test_classify_with_modalities(self):
        """测试带模态的分类"""
        task_type, confidence = self.classifier.classify(
            "分析这张架构图",
            input_modalities=["image"],
        )

        self.assertIn(
            task_type, [self.TaskType.IMAGE_UNDERSTANDING, self.TaskType.MULTIMODAL_REASONING]
        )

    def test_classify_empty(self):
        """测试空输入分类"""
        task_type, confidence = self.classifier.classify("")

        # R11 契约核对：空 prompt 且无输入模态时显式返回 (UNKNOWN, 0.0)
        self.assertEqual(task_type, self.TaskType.UNKNOWN)
        self.assertEqual(confidence, 0.0)

    def test_detect_output_format_json(self):
        """测试 JSON 输出格式检测"""
        output_format = self.classifier.detect_output_format(
            "请以 JSON 格式返回结果",
            self.TaskType.QUESTION_ANSWERING,
        )

        self.assertEqual(output_format, self.OutputFormat.JSON)

    def test_detect_output_format_code(self):
        """测试代码输出格式检测"""
        output_format = self.classifier.detect_output_format(
            "实现这个功能",
            self.TaskType.CODE_GENERATION,
        )

        self.assertEqual(output_format, self.OutputFormat.CODE)

    def test_detect_output_format_markdown(self):
        """测试 Markdown 输出格式检测"""
        output_format = self.classifier.detect_output_format(
            "使用标题和列表格式",
            self.TaskType.TEXT_SUMMARIZATION,
        )

        self.assertEqual(output_format, self.OutputFormat.MARKDOWN)

    def test_detect_output_format_default(self):
        """测试默认输出格式检测"""
        output_format = self.classifier.detect_output_format(
            "普通文本",
            self.TaskType.QUESTION_ANSWERING,
        )

        self.assertEqual(output_format, self.OutputFormat.PLAIN_TEXT)


class TestMultimodalTaskDecoder(unittest.TestCase):
    """测试多模态任务解码器"""

    def setUp(self):
        from hive.core.multimodal_task_decoder import MultimodalTaskDecoder, OutputFormat, TaskType

        self.MultimodalTaskDecoder = MultimodalTaskDecoder
        self.TaskType = TaskType
        self.OutputFormat = OutputFormat

        self.decoder = self.MultimodalTaskDecoder(embedding_dim=128)

    def test_decode_task_basic(self):
        """测试基本任务解码"""
        unified_embedding = [0.1] * 128

        output = self.decoder.decode_task(
            prompt="Test decoding",
            unified_embedding=unified_embedding,
            input_modalities=["text"],
        )

        self.assertIsNotNone(output.output_id)
        self.assertEqual(output.source_modalities, ["text"])

    def test_decode_task_with_expected_type(self):
        """测试带期望类型的任务解码"""
        unified_embedding = [0.1] * 128

        output = self.decoder.decode_task(
            prompt="Generate Python code",
            unified_embedding=unified_embedding,
            input_modalities=["text"],
            expected_task_type=self.TaskType.CODE_GENERATION,
            expected_output_format=self.OutputFormat.CODE,
        )

        self.assertIsNotNone(output)

    def test_decode_task_with_raw_inputs(self):
        """测试带原始输入的任务解码"""
        unified_embedding = [0.1] * 128

        output = self.decoder.decode_task(
            prompt="Analyze this",
            unified_embedding=unified_embedding,
            input_modalities=["text", "image"],
            raw_inputs={
                "text": ["This is test input"],
                "images": ["/path/to/image.png"],
            },
        )

        self.assertIsNotNone(output)

    def test_reason_with_modalities(self):
        """测试多模态推理"""
        result = self.decoder.reason_with_modalities(
            question="What is the relationship between these inputs?",
            text_inputs=["Text context here"],
            image_inputs=["/path/to/image.png"],
            reasoning_type="comprehensive",
        )

        self.assertIn("question", result)
        self.assertIn("modalities_used", result)
        self.assertIn("conclusion", result)

    def test_generate_code_from_multimodal(self):
        """测试从多模态生成代码"""
        result = self.decoder.generate_code_from_multimodal(
            requirement="Create a simple function",
            text_inputs=["Function to add two numbers"],
            language="python",
        )

        self.assertIn("status", result)
        self.assertIn("code", result)

    def test_extract_knowledge_from_multimodal(self):
        """测试从多模态提取知识"""
        result = self.decoder.extract_knowledge_from_multimodal(
            text_inputs=["Python is a programming language. It was created by Guido van Rossum."],
            focus="programming languages",
        )

        self.assertIn("status", result)
        self.assertIn("extraction", result)

    def test_get_stats(self):
        """测试获取统计信息"""
        self.decoder.decode_task(
            prompt="Test 1",
            unified_embedding=[0.1] * 128,
            input_modalities=["text"],
        )

        stats = self.decoder.get_stats()

        self.assertIn("total_tasks_processed", stats)
        self.assertGreater(stats["total_tasks_processed"], 0)
        self.assertIn("embedding_dimension", stats)


class TestCrossModalReasoningEngine(unittest.TestCase):
    """测试跨模态推理引擎"""

    def setUp(self):
        from hive.core.multimodal_task_decoder import CrossModalReasoningEngine

        self.CrossModalReasoningEngine = CrossModalReasoningEngine

        self.engine = self.CrossModalReasoningEngine(embedding_dim=128)

    def test_reason_basic(self):
        """测试基本推理"""
        result = self.engine.reason(
            question="What can we learn from this data?",
            modality_inputs={
                "text": ["Sample text input"],
            },
            reasoning_type="comprehensive",
        )

        self.assertIn("question", result)
        self.assertIn("modalities_used", result)
        self.assertIn("conclusion", result)

    def test_reason_with_multiple_modalities(self):
        """测试多模态推理"""
        result = self.engine.reason(
            question="Analyze the relationship between these inputs.",
            modality_inputs={
                "text": ["The user is asking about architecture."],
                "image": ["architecture_diagram.png"],
            },
        )

        self.assertIn("text", result["modalities_used"])
        self.assertIn("image", result["modalities_used"])

    def test_causal_reasoning(self):
        """测试因果推理"""
        result = self.engine.causal_reasoning(
            cause_inputs=["Event A occurred", "Action was taken"],
            effect_inputs=["Result B happened"],
            question="What is the causal relationship?",
        )

        self.assertIn("conclusion", result)

    def test_analogical_reasoning(self):
        """测试类比推理"""
        result = self.engine.analogical_reasoning(
            source_inputs=["A : B = C : ?"],
            target_inputs=["Pattern matching scenario"],
            question="Solve the analogy.",
        )

        self.assertIn("conclusion", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
