"""R12 回归：分诊 #18 —— ImageEncoder 对不存在的路径必须显式失败。

背景：``encode()`` 旧实现只用 ``Path(x).exists()`` 判定「路径」语义，不存在的
文件路径字符串会落进 else 分支被当作 base64 原样处理，最终返回 success=True。
R12 经用户拍板，对齐 ``AudioEncoder.encode()`` 既有的 ``path.exists()`` 前置
校验：路径语义输入不存在时返回 success=False；base64 / data URI 输入保持原有
数据语义，不因修复被误伤。
"""
from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from hive.core.multimodal_encoder import (
    ImageEncoder,
    ModalityType,
    ModalityEmbedding,
    MultimodalEncoder,
    _is_path_like_image_input,
)

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "hive" / "core" / "multimodal_encoder.py"
MISSING = "/nonexistent/path/to/image.png"


@pytest.mark.parametrize(
    "value, expected",
    [
        (MISSING, True),
        ("relative/missing/photo.JPG", True),
        (r"C:\missing\screenshot.png", True),
        (Path("/definitely/missing/a.png"), True),
        ("not-an-image.txt", True),
        # 既无扩展名又恰好是合法 base64 的路径字符串：无法与数据区分，
        # 有意保留旧的数据语义，避免误伤含 '/' 的真实 base64 图像数据。
        ("a/path/without/extension", False),
        ("plain short prompt", False),
        ("Hello world.", False),
        ("", False),
        ("data:image/png;base64,iVBORw0KGgo=", False),
        (base64.b64encode(b"contains/a/slash").decode(), False),
        ("aGVsbG8gd29ybGQ=", False),
    ],
)
def test_path_like_classification(value, expected):
    """路径语义与数据语义的判定表；含 '/' 的合法 base64 不能被误判为路径。"""
    assert _is_path_like_image_input(value) is expected


def test_missing_path_returns_failure_with_reason():
    encoder = ImageEncoder()
    result = encoder.encode(MISSING)
    assert result.success is False
    assert "not found" in result.message
    assert result.modality_type == ModalityType.IMAGE
    assert result.embedding is None


def test_missing_path_as_pathlib_object_returns_failure():
    result = ImageEncoder().encode(Path("/definitely/missing/a.png"))
    assert result.success is False
    assert "not found" in result.message


def test_existing_file_still_succeeds(tmp_path):
    """真实存在的文件仍走路径分支并成功；外部模型被 patch 掉以免联网。"""
    image = tmp_path / "tiny.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"payload")
    encoder = ImageEncoder()
    with patch.object(encoder, "_get_brain", return_value=None), patch.object(
        encoder, "_get_ollama", return_value=None
    ):
        result = encoder.encode(image)
    assert result.success is True
    assert result.embedding is not None


@pytest.mark.parametrize(
    "data",
    [
        base64.b64encode(b"a/b/c/d").decode(),
        "data:image/png;base64,iVBORw0KGgo=",
        "plain short prompt",
    ],
)
def test_data_inputs_keep_original_success_semantics(data):
    """修复只针对路径语义输入，base64 / data URI / 短文本不得被连带改成失败。"""
    encoder = ImageEncoder()
    with patch.object(encoder, "_get_brain", return_value=None), patch.object(
        encoder, "_get_ollama", return_value=None
    ):
        result = encoder.encode(data)
    assert result.success is True


def test_unified_router_propagates_failure():
    """MultimodalEncoder 显式 IMAGE 路由到不存在的路径也必须失败。"""
    result = MultimodalEncoder().encode(MISSING, ModalityType.IMAGE)
    assert result.success is False


def test_legacy_classification_injection_reproduces_silent_success():
    """旧实现注入反证：把判定注回「永远不是路径」，静默成功必须复现。

    全程只在内存里 patch 判定函数，不改写产品源码（末尾校验字节级 SAME）。
    """
    before = hashlib.sha256(PRODUCT.read_bytes()).hexdigest()
    encoder = ImageEncoder()
    with patch(
        "hive.core.multimodal_encoder._is_path_like_image_input", return_value=False
    ), patch.object(encoder, "_get_brain", return_value=None), patch.object(
        encoder, "_get_ollama", return_value=None
    ):
        legacy = encoder.encode(MISSING)
    assert legacy.success is True, "旧判定下不存在的路径会被当作 base64 静默成功"
    assert hashlib.sha256(PRODUCT.read_bytes()).hexdigest() == before


def test_guard_is_structural_not_hardcoded():
    """AST 守卫：产品里必须真有判定函数，且 ImageEncoder.encode 真的调用它。"""
    tree = ast.parse(PRODUCT.read_text(encoding="utf-8"))
    helper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_is_path_like_image_input"]
    assert helper, "路径语义判定函数缺失"
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ImageEncoder")
    encode = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "encode")
    called = [
        c for c in ast.walk(encode)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "_is_path_like_image_input"
    ]
    assert called, "encode() 必须显式调用路径语义判定，不能退回 exists() 硬编码"

    # 与 AudioEncoder 的既有语义对齐：音频编码器必须仍保留 path.exists() 前置校验
    audio_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AudioEncoder")
    audio_encode = next(n for n in audio_cls.body if isinstance(n, ast.FunctionDef) and n.name == "encode")
    assert any("not path.exists()" in ast.unparse(n) for n in ast.walk(audio_encode))


def test_encoding_result_shape_unchanged():
    """对外返回类型不变：仍是 EncodingResult / ModalityEmbedding 那一套。"""
    assert hasattr(ModalityEmbedding, "__dataclass_fields__")
    result = ImageEncoder().encode(MISSING)
    assert type(result).__name__ == "EncodingResult"
    assert result.modality_type == ModalityType.IMAGE
