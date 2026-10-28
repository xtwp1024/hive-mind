# -*- coding: utf-8 -*-
"""
批量翻译 GitHub 项目 README 并更新 Obsidian 笔记
功能：
1. 读取每个克隆仓库的 README 文件
2. 使用 GLM-4-Flash API 翻译为中文
3. 更新对应 Obsidian 笔记，添加翻译后的 README 内容
"""

import os
import sys
import time
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_TIMEOUT = 60  # 秒
DEEPSEEK_MODEL = "deepseek-chat"


# 翻译提示词
TRANSLATE_PROMPT = """你是一个专业的技术文档翻译助手。请将以下 GitHub 项目的 README 文档翻译成中文（简体中文）。

翻译要求：
1. 保持技术术语的准确性，常用术语（如 README、API、CLI、SDK等）可直接使用英文
2. 保持代码块的原样输出，不翻译代码
3. 保持 Markdown 格式
4. 标题翻译后加括号标注英文原文，如 "Getting Started (快速开始)"
5. 翻译要准确、通顺，符合中文技术文档习惯
6. 如果描述部分过于简单，可以适当补充解释

以下是需要翻译的 README 内容：

"""

SUMMARY_PROMPT = """请为以下 GitHub 项目的 README 内容生成一个简短的中文摘要（200字以内），
包括：项目功能、主要特性、适用场景。

README 内容：

"""


class READMETranslator:
    """README 翻译器"""

    def __init__(self, clone_dir: str, obsidian_dir: str, api_key: str = None):
        self.clone_dir = Path(clone_dir)
        self.obsidian_dir = Path(obsidian_dir)
        self.api_key = api_key or DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY", "")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.lock = Lock()
        self.translated_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.rate_limit_count = 0

        # 加载已处理的笔记
        self.processed_notes = set()
        self._load_processed_notes()

    def _load_processed_notes(self):
        """加载已处理的笔记"""
        if self.obsidian_dir.exists():
            for note in self.obsidian_dir.glob("*.md"):
                self.processed_notes.add(note.stem)
        logger.info(f"已扫描到 {len(self.processed_notes)} 个笔记")

    def _find_readme(self, repo_path: Path) -> Optional[Path]:
        """查找 README 文件"""
        readme_names = [
            "README.md", "README.rst", "README.txt", "README",
            "readme.md", "README.MD", "Readme.md"
        ]

        for name in readme_names:
            readme_path = repo_path / name
            if readme_path.exists() and readme_path.is_file():
                try:
                    size = readme_path.stat().st_size
                    if size > 0:
                        return readme_path
                except:
                    pass

        # 检查根目录下的 README
        for item in repo_path.iterdir():
            if item.is_file() and item.name.upper().startswith("README"):
                try:
                    if item.stat().st_size > 0:
                        return item
                except:
                    pass

        return None

    def _read_readme(self, readme_path: Path, max_chars: int = 8000) -> str:
        """读取 README 内容"""
        try:
            # 尝试不同编码
            for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']:
                try:
                    with open(readme_path, 'r', encoding=encoding) as f:
                        content = f.read(max_chars)
                    return content
                except:
                    continue

            # 如果都失败，尝试二进制读取后解码
            with open(readme_path, 'rb') as f:
                content = f.read(max_chars)
            return content.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.debug(f"读取失败 {readme_path}: {e}")
            return ""

    def _translate_text(self, text: str, prompt_prefix: str = TRANSLATE_PROMPT) -> str:
        """调用 DeepSeek API 翻译文本"""
        if not self.api_key:
            logger.warning("未设置 DEEPSEEK_API_KEY，无法翻译")
            return text

        if not text or len(text.strip()) < 10:
            return text

        # 截断过长的文本
        if len(text) > 6000:
            text = text[:6000]

        prompt = prompt_prefix + text

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.3,
        }

        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=self.headers,
                json=payload,
                timeout=DEEPSEEK_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                translated = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return translated.strip()
            elif response.status_code == 429:
                self.rate_limit_count += 1
                logger.warning("API 限流，等待 30 秒...")
                time.sleep(30)
                return None
            else:
                logger.warning(f"翻译 API 错误: {response.status_code} - {response.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("翻译超时")
            return None
        except Exception as e:
            logger.warning(f"翻译异常: {e}")
            return None

    def _translate_readme(self, readme_content: str) -> tuple:
        """翻译 README 内容"""
        if not readme_content or len(readme_content.strip()) < 50:
            return ("", "empty")

        # 翻译主内容
        translated = self._translate_text(readme_content)
        if translated is None:
            return ("", "api_error")

        if not translated:
            return ("", "empty_result")

        return (translated, "success")

    def _get_summary(self, readme_content: str) -> str:
        """生成中文摘要"""
        if not self.api_key or not readme_content:
            return ""

        summary = self._translate_text(readme_content[:3000], SUMMARY_PROMPT)
        return summary or ""

    def _update_obsidian_note(self, note_path: Path, translated_content: str, summary: str = ""):
        """更新 Obsidian 笔记，添加翻译后的 README"""
        try:
            existing_content = ""
            if note_path.exists():
                with open(note_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()

            # 检查是否已有翻译
            if "## 📖 中文自述 (README)" in existing_content:
                logger.info(f"  ⏭️ 已翻译，跳过: {note_path.name}")
                return "already_translated"

            # 构建更新后的内容
            readme_section = f"""

---

## 📖 中文自述 (README)

{translated_content}

---

> 🤖 由 Hive 自动翻译 | 翻译时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

            # 追加到文件
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(existing_content)
                f.write(readme_section)

            return "success"

        except Exception as e:
            logger.error(f"  ❌ 更新笔记失败 {note_path.name}: {e}")
            return "failed"

    def process_single_repo(self, repo_name: str) -> tuple:
        """处理单个仓库"""
        repo_path = self.clone_dir / repo_name
        note_path = self.obsidian_dir / f"{repo_name}.md"

        if not repo_path.exists():
            return (repo_name, "skipped", "Repo dir not found")

        # 查找 README
        readme_path = self._find_readme(repo_path)
        if not readme_path:
            return (repo_name, "skipped", "No README found")

        # 检查笔记是否存在
        if not note_path.exists():
            # 尝试其他可能的笔记名
            found_note = False
            for note_file in self.obsidian_dir.glob("*.md"):
                # 模糊匹配
                if repo_name.lower().replace("_", "-") in note_file.stem.lower().replace("_", "-"):
                    note_path = note_file
                    found_note = True
                    break
            if not found_note:
                return (repo_name, "skipped", "No Obsidian note found")

        # 读取 README
        readme_content = self._read_readme(readme_path)
        if not readme_content or len(readme_content.strip()) < 50:
            return (repo_name, "skipped", "README too short or empty")

        logger.info(f"  翻译中: {repo_name} ({len(readme_content)} chars)")

        # 翻译
        translated, status = self._translate_readme(readme_content)
        if status != "success" or not translated:
            return (repo_name, "failed", f"Translation {status}")

        # 生成摘要
        summary = self._get_summary(readme_content)

        # 更新笔记
        update_status = self._update_obsidian_note(note_path, translated, summary)

        return (repo_name, update_status, note_path.name)

    def run(self, max_workers: int = 4, limit: int = 0) -> dict:
        """运行批量翻译"""
        # 获取所有仓库
        repos = sorted([d.name for d in self.clone_dir.iterdir() if d.is_dir()])

        if limit > 0:
            repos = repos[:limit]

        logger.info(f"=" * 60)
        logger.info(f"开始翻译 README，目标: {len(repos)} 个仓库")
        logger.info(f"克隆目录: {self.clone_dir}")
        logger.info(f"笔记目录: {self.obsidian_dir}")
        logger.info(f"API Key: {'已设置' if self.api_key else '未设置'}")
        logger.info(f"=" * 60)

        if not self.api_key:
            logger.error("未设置 GLM_API_KEY，无法进行翻译！")
            logger.error("请设置环境变量: export GLM_API_KEY=your_api_key")
            return {"status": "no_api_key"}

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_repo = {
                executor.submit(self.process_single_repo, repo): repo
                for repo in repos
            }

            for future in as_completed(future_to_repo):
                result = future.result()
                results.append(result)

                if result[1] == "success":
                    with self.lock:
                        self.translated_count += 1
                    logger.info(f"  ✅ 翻译完成: {result[0]} → {result[2]}")
                elif result[1] == "already_translated":
                    with self.lock:
                        self.skipped_count += 1
                elif result[1] == "skipped":
                    with self.lock:
                        self.skipped_count += 1
                    logger.debug(f"  ⏭️ 跳过: {result[0]} - {result[2]}")
                else:
                    with self.lock:
                        self.failed_count += 1
                    logger.warning(f"  ❌ 失败: {result[0]} - {result[2]}")

        elapsed = time.time() - start_time

        summary = {
            "status": "completed",
            "total": len(repos),
            "success": self.translated_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "rate_limit_hits": self.rate_limit_count,
            "elapsed_seconds": elapsed,
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"翻译完成!")
        logger.info(f"成功: {self.translated_count}")
        logger.info(f"跳过: {self.skipped_count}")
        logger.info(f"失败: {self.failed_count}")
        logger.info(f"限流: {self.rate_limit_count}")
        logger.info(f"耗时: {elapsed:.1f} 秒")
        logger.info(f"{'=' * 60}")

        return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="批量翻译 GitHub 项目 README")
    parser.add_argument(
        "--clone-dir", "-c",
        default="D:/Hive/Data/trading_repos",
        help="克隆仓库目录"
    )
    parser.add_argument(
        "--obsidian-dir", "-o",
        default="C:/Users/Administrator/Documents/Obsidian Vault/Trading/GitHub_Repos",
        help="Obsidian 笔记目录"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.getenv("DEEPSEEK_API_KEY", ""),
        help="DeepSeek API Key"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="并发数"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="限制处理数量（0=全部）"
    )

    args = parser.parse_args()

    translator = READMETranslator(
        clone_dir=args.clone_dir,
        obsidian_dir=args.obsidian_dir,
        api_key=args.api_key
    )

    result = translator.run(max_workers=args.workers, limit=args.limit)

    return result


if __name__ == "__main__":
    main()
