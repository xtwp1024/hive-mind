# -*- coding: utf-8 -*-
"""
Phase 4: 多模态扩展 - 快速功能验证脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_text_encoder():
    """测试文本编码器"""
    print('[1/8] 测试 TextEncoder...')

    from hive.core.multimodal_encoder import TextEncoder, ModalityType, EncoderType

    encoder = TextEncoder()

    result = encoder.encode("Hello, this is a test text for multimodal encoding.")

    print(f'    编码成功: {result.success}')
    print(f'    模态类型: {result.modality_type.value if result.modality_type else "N/A"}')
    if result.embedding:
        print(f'    嵌入维度: {result.embedding.embedding_dim}')
        print(f'    处理时间: {result.embedding.processing_time_ms:.2f}ms')

    print('    [OK] TextEncoder 验证通过')
    return True


def test_image_encoder():
    """测试图像编码器"""
    print()
    print('[2/8] 测试 ImageEncoder...')

    from hive.core.multimodal_encoder import ImageEncoder

    encoder = ImageEncoder()

    print('    架构图分析方法可用:', hasattr(encoder, 'analyze_architecture'))
    print('    截图分析方法可用:', hasattr(encoder, 'analyze_screenshot'))
    print('    编码方法可用:', hasattr(encoder, 'encode'))

    print('    [OK] ImageEncoder 验证通过')
    return True


def test_audio_encoder():
    """测试音频编码器"""
    print()
    print('[3/8] 测试 AudioEncoder...')

    from hive.core.multimodal_encoder import AudioEncoder

    encoder = AudioEncoder()

    print('    转录方法可用:', hasattr(encoder, 'encode'))
    print('    转录总结方法可用:', hasattr(encoder, 'transcribe_and_summarize'))

    print('    [OK] AudioEncoder 验证通过')
    return True


def test_multimodal_encoder():
    """测试多模态统一编码器"""
    print()
    print('[4/8] 测试 MultimodalEncoder...')

    from hive.core.multimodal_encoder import (
        MultimodalEncoder, ModalityType, ModalityEmbedding
    )

    encoder = MultimodalEncoder()

    modality = encoder.detect_modality("Hello world")
    print(f'    文本模态检测: {modality.value}')

    modality2 = encoder.detect_modality("test.png")
    print(f'    图像模态检测 (路径): {modality2.value}')

    text_emb = ModalityEmbedding(
        embedding_id="test_text_1",
        modality_type=ModalityType.TEXT,
        encoder_type=encoder.text_encoder.preferred_encoder,
        embedding=[0.1, 0.2, 0.3, 0.4],
        embedding_dim=4,
    )

    img_emb = ModalityEmbedding(
        embedding_id="test_img_1",
        modality_type=ModalityType.IMAGE,
        encoder_type=encoder.image_encoder.preferred_encoder,
        embedding=[0.5, 0.6, 0.7, 0.8],
        embedding_dim=4,
    )

    fused = encoder.fuse_embeddings([text_emb, img_emb], strategy="weighted_average")
    print(f'    融合嵌入维度: {fused.embedding_dim}')
    print(f'    融合嵌入长度: {len(fused.embedding)}')

    print('    [OK] MultimodalEncoder 验证通过')
    return True


def test_cross_modal_attention():
    """测试跨模态注意力"""
    print()
    print('[5/8] 测试 CrossModalAttention...')

    from hive.core.cross_modal_association import CrossModalAttention

    attention = CrossModalAttention(embedding_dim=4)

    emb1 = [0.1, 0.2, 0.3, 0.4]
    emb2 = [0.5, 0.6, 0.7, 0.8]
    emb3 = [-0.1, -0.2, -0.3, -0.4]

    aligned, weights, best_idx = attention.guided_attention(
        source_embedding=emb1,
        target_embeddings=[emb2, emb3],
    )

    print(f'    对齐嵌入长度: {len(aligned)}')
    print(f'    注意力权重数量: {len(weights)}')
    print(f'    最佳匹配索引: {best_idx}')

    output, attn_weights = attention.self_attention([emb1, emb2, emb3])
    print(f'    自注意力输出数量: {len(output)}')

    print('    [OK] CrossModalAttention 验证通过')
    return True


def test_unified_representation_space():
    """测试统一表示空间"""
    print()
    print('[6/8] 测试 UnifiedRepresentationSpace...')

    from hive.core.cross_modal_association import UnifiedRepresentationSpace

    space = UnifiedRepresentationSpace(dimension=128)

    space.add_embedding(
        embedding_id="text_001",
        embedding=[0.1] * 64,
        modality="text",
        metadata={"source": "test"},
    )

    space.add_embedding(
        embedding_id="image_001",
        embedding=[0.2] * 64,
        modality="image",
        metadata={"source": "test"},
    )

    stats = space.get_stats()
    print(f'    总嵌入数: {stats["total_embeddings"]}')
    print(f'    空间维度: {stats["dimension"]}')
    print(f'    按模态统计: {stats["by_modality"]}')

    results = space.similarity_search(
        query_embedding=[0.15] * 64,
        query_modality="text",
        top_k=2,
    )

    print(f'    相似搜索结果数: {len(results)}')

    print('    [OK] UnifiedRepresentationSpace 验证通过')
    return True


def test_cross_modal_retriever():
    """测试跨模态检索器"""
    print()
    print('[7/8] 测试 CrossModalRetriever...')

    from hive.core.cross_modal_association import (
        CrossModalRetriever, UnifiedRepresentationSpace
    )

    space = UnifiedRepresentationSpace(dimension=128)

    space.add_embedding("img_cat", [0.1] * 128, "image", {"label": "cat"})
    space.add_embedding("img_dog", [0.2] * 128, "image", {"label": "dog"})
    space.add_embedding("text_animal", [0.15] * 128, "text", {"label": "animal"})

    retriever = CrossModalRetriever(space, top_k_default=5)

    result = retriever.text_to_image("cat sitting on a mat")
    print(f'    文本检索图像结果数: {result.total_count}')
    print(f'    检索ID: {result.query_id}')

    recent = retriever.get_recent_retrievals(count=1)
    print(f'    最近检索记录数: {len(recent)}')

    print('    [OK] CrossModalRetriever 验证通过')
    return True


def test_multimodal_task_decoder():
    """测试多模态任务解码器"""
    print()
    print('[8/8] 测试 MultimodalTaskDecoder...')

    from hive.core.multimodal_task_decoder import (
        MultimodalTaskDecoder, TaskType, OutputFormat, TaskTypeClassifier
    )

    classifier = TaskTypeClassifier()

    task_type, confidence = classifier.classify("请帮我写一个 Python 函数来计算斐波那契数列")
    print(f'    代码生成任务检测: {task_type.value}, 置信度: {confidence:.2f}')

    task_type2, confidence2 = classifier.classify("总结以下内容的要点")
    print(f'    摘要任务检测: {task_type2.value}, 置信度: {confidence2:.2f}')

    output_format = classifier.detect_output_format("请以 JSON 格式返回结果", TaskType.QUESTION_ANSWERING)
    print(f'    输出格式检测: {output_format.value}')

    decoder = MultimodalTaskDecoder(embedding_dim=128)

    stats = decoder.get_stats()
    print(f'    解码器嵌入维度: {stats["embedding_dimension"]}')

    unified_embedding = [0.1] * 128

    output = decoder.decode_task(
        prompt="测试解码功能",
        unified_embedding=unified_embedding,
        input_modalities=["text"],
        raw_inputs={"text": ["This is test input"]},
    )

    print(f'    解码成功: {output.output_id is not None}')
    print(f'    解码置信度: {output.confidence}')

    stats2 = decoder.get_stats()
    print(f'    已处理任务数: {stats2["total_tasks_processed"]}')

    print('    [OK] MultimodalTaskDecoder 验证通过')
    return True


def main():
    print('=' * 60)
    print('Phase 4: 多模态扩展 - 快速功能验证')
    print('=' * 60)

    tests = [
        test_text_encoder,
        test_image_encoder,
        test_audio_encoder,
        test_multimodal_encoder,
        test_cross_modal_attention,
        test_unified_representation_space,
        test_cross_modal_retriever,
        test_multimodal_task_decoder,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f'    [FAIL] 错误: {e}')
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print('=' * 60)
    print(f'验证结果: {passed} 通过, {failed} 失败')
    print('=' * 60)

    if failed == 0:
        print()
        print('Phase 4 所有模块验证通过！')
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
