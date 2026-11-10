# -*- coding: utf-8 -*-
"""R1 验收口径修订的回归测试：镜像项允许「本地领先待推送」，但仍拦分叉。

旧口径要求三方严格相等，导致每轮 commit 后、push 前误报 FAIL（把正常
推进态判成基线已坏）。修订后：镜像必须 == origin/master，主工作区允许
领先，但落后/分叉必须仍然 FAIL。
"""
from unittest.mock import patch

from tools import loop_check


def _fake_git(head: str, origin: str, mirror: str, ancestor_ok: bool = True):
    """按 git 参数返回预设值；merge-base 用退出语义模拟。"""
    def _git(*args, cwd=None):
        cmd = args[0]
        if cmd == "rev-parse":
            if cwd is not None:
                return mirror
            return origin if args[1] == "origin/master" else head
        if cmd == "merge-base":
            return "" if ancestor_ok else "__ERROR__exit 1"
        if cmd == "rev-list":
            return "2"
        return ""
    return _git


def test_aligned_is_pass():
    with patch.object(loop_check, "git", side_effect=_fake_git("a" * 40, "a" * 40, "a" * 40)), \
         patch.object(loop_check.MIRROR.__class__, "exists", return_value=True):
        issues, status = loop_check.check_mirror()
    assert issues == []
    assert "三方对齐" in status


def test_local_ahead_is_not_a_failure():
    """核心修订：本地领先 2 个提交待推送，不应被判成基线已坏。"""
    with patch.object(loop_check, "git", side_effect=_fake_git("b" * 40, "a" * 40, "a" * 40)), \
         patch.object(loop_check.MIRROR.__class__, "exists", return_value=True):
        issues, status = loop_check.check_mirror()
    assert issues == [], issues
    assert "本地领先" in status and "待推送" in status


def test_mirror_lagging_still_fails():
    with patch.object(loop_check, "git", side_effect=_fake_git("a" * 40, "a" * 40, "c" * 40)), \
         patch.object(loop_check.MIRROR.__class__, "exists", return_value=True):
        issues, _ = loop_check.check_mirror()
    assert any("镜像与远端不一致" in x for x in issues)


def test_diverged_still_fails():
    """反证：修订不能放宽到把分叉也放过。"""
    with patch.object(loop_check, "git",
                      side_effect=_fake_git("b" * 40, "a" * 40, "a" * 40, ancestor_ok=False)), \
         patch.object(loop_check.MIRROR.__class__, "exists", return_value=True):
        issues, _ = loop_check.check_mirror()
    assert any("落后或分叉" in x for x in issues)
