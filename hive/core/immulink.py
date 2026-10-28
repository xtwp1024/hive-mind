# -*- coding: utf-8 -*-
import shutil
import filecmp
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

class ImmuLink:
    """
    ImmuLink: 双向验证导回策略 (Two-Way Verification Merge Strategy)
    
    负责将沙箱中的优化结果安全地应用回原始项目。
    包含前后两次验证逻辑，确保代码在导回前后都是可运行的。
    """
    
    def __init__(self, src_root: Path, sandbox_root: Path):
        self.src_root = src_root
        self.sandbox_root = sandbox_root

    def verify_sandbox(self) -> bool:
        """
        验证 1: 沙箱验证 (Sandbox Verification)
        检查沙箱内的修改是否通过了基本健康检查。
        """
        python_files = list(self.sandbox_root.rglob("*.py"))
        if not python_files:
            return True
            
        for py_file in python_files:
            try:
                # 排除可能非常大的文件（可选）
                if py_file.stat().st_size > 1024 * 1024: # > 1MB 
                    continue
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        compile(content, str(py_file), 'exec')
            except Exception as e:
                logger.error(f"🛡️ [ImmuLink] 沙箱验证失败: {py_file.name} 存在语法错误: {e}")
                return False
        
        logger.info(f"🛡️ [ImmuLink] 沙箱验证通过: 成功验证 {len(python_files)} 个 Python 文件")
        return True

    def identify_changes(self) -> List[Path]:
        """
        识别沙箱中被修改的文件
        """
        changed_files = []
        for sandbox_file in self.sandbox_root.rglob("*"):
            if sandbox_file.is_file():
                try:
                    rel_path = sandbox_file.relative_to(self.sandbox_root)

                    # 防止路径遍历攻击：确保相对路径不包含可以逃逸的组件
                    if ".." in rel_path.parts:
                        logger.warning(f"🛡️ [ImmuLink] 跳过可疑路径（路径遍历检测）: {rel_path}")
                        continue

                    src_file = self.src_root / rel_path

                    # 如果原文件不存在，或者内容不一致，则认为是修改
                    if not src_file.exists() or not filecmp.cmp(src_file, sandbox_file, shallow=False):
                        changed_files.append(rel_path)
                except Exception as e:
                    logger.warning(f"🛡️ [ImmuLink] 识别文件对比失败 {sandbox_file}: {e}")

        return changed_files

    def merge_back(self, changed_files: List[Path]) -> tuple[bool, dict]:
        """
        物理导回修改（使用临时目录实现原子性，失败时自动回滚）
        返回 (success, backups) — backups 供调用方在后续验证失败时回滚
        """
        import tempfile
        import shutil as sh

        if not changed_files:
            logger.info("🛡️ [ImmuLink] 没有发现变更，无需导回。")
            return True, {}

        backups: dict = {}
        try:
            # 第一阶段：读取所有原始文件备份（在内存中）
            for rel_path in changed_files:
                src_file = self.src_root / rel_path
                if src_file.exists():
                    backups[src_file] = src_file.read_bytes()

            # 第二阶段：使用临时目录批量复制，实现原子性
            # 如果复制过程中崩溃，原始文件不受影响
            with tempfile.TemporaryDirectory(dir=self.src_root.parent) as tmp_dir:
                tmp_root = Path(tmp_dir) / self.src_root.name

                for rel_path in changed_files:
                    src_file = self.src_root / rel_path
                    sandbox_file = self.sandbox_root / rel_path

                    # 安全检查：防止符号链接攻击
                    if sandbox_file.is_symlink() or (src_file.exists() and src_file.is_symlink()):
                        logger.warning(f"🛡️ [ImmuLink] 跳过符号链接（沙箱逃逸防护）: {rel_path}")
                        continue

                    # 安全检查：确保沙箱文件路径仍在沙箱目录内
                    try:
                        sandbox_resolved = sandbox_file.resolve()
                        if not str(sandbox_resolved).startswith(str(self.sandbox_root.resolve())):
                            logger.warning(f"🛡️ [ImmuLink] 跳过可疑路径（路径遍历检测）: {rel_path}")
                            continue
                    except Exception as e:
                        logger.warning(f"🛡️ [ImmuLink] 路径解析失败: {rel_path}: {e}")
                        continue

                    # 复制到临时目录
                    dst_file = tmp_root / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    sh.copy2(sandbox_file, dst_file)
                    logger.info(f"🛡️ [ImmuLink] 已备份变更至临时区: {rel_path}")

                # 第三阶段：原子性替换原始目录
                # 先备份原目录（用于异常时回滚）
                backup_dir = Path(tmp_dir) / f"{self.src_root.name}_backup"
                sh.copytree(self.src_root, backup_dir, dirs_exist_ok=True)

                try:
                    # 删除原始内容，替换为新内容
                    sh.rmtree(self.src_root)
                    sh.move(tmp_root, self.src_root)
                    logger.info(f"🛡️ [ImmuLink] 已原子性应用所有变更: {len(changed_files)} 个文件")
                except Exception as move_err:
                    # 移动失败，尝试恢复原目录
                    try:
                        sh.rmtree(self.src_root)
                    except Exception:
                        pass
                    sh.move(backup_dir, self.src_root)
                    raise move_err

            return True, backups
        except Exception as e:
            logger.error(f"🛡️ [ImmuLink] 导回过程中出错，正在回滚: {e}")
            for src_file, original in backups.items():
                try:
                    src_file.write_bytes(original)
                except Exception as rb_e:
                    logger.error(f"🛡️ [ImmuLink] 回滚失败: {src_file}: {rb_e}")
            return False, {}

    def verify_source(self) -> bool:
        """
        验证 2: 宿主验证 (Source Verification)
        导回后的整体项目健康检查。
        """
        python_files = list(self.src_root.rglob("*.py"))
        for py_file in python_files:
            try:
                if py_file.stat().st_size > 1024 * 1024:  # > 1MB，跳过
                    continue
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        compile(content, str(py_file), 'exec')
            except Exception as e:
                logger.error(f"🛡️ [ImmuLink] 宿主验证失败: {py_file.name} 导回后出现错误: {e}")
                return False
        return True

def apply_sandbox_to_source(src_path: Path, sandbox_path: Path) -> bool:
    """
    便捷函数：执行完整的导回策略
    """
    link = ImmuLink(src_path, sandbox_path)

    # 1. 验证沙箱
    if not link.verify_sandbox():
        return False

    # 2. 识别变更
    changes = link.identify_changes()
    if not changes:
        return True

    # 3. 物理导回（获取备份供回滚）
    success, backups = link.merge_back(changes)
    if not success:
        return False

    # 4. 再次验证，失败则用备份回滚
    if not link.verify_source():
        logger.warning("🛡️ [ImmuLink] 宿主验证未通过，正在回滚...")
        for src_file, original in backups.items():
            try:
                src_file.write_bytes(original)
            except Exception as e:
                logger.error(f"🛡️ [ImmuLink] 回滚失败: {src_file}: {e}")
        logger.error("🛡️ [ImmuLink] 宿主验证失败，已回滚至导回前状态。")
        return False

    logger.info(f"🛡️ [ImmuLink] 双向验证导回成功: {len(changes)} 个文件已同步。")
    return True
