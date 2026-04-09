# -*- coding: utf-8 -*-
"""
LibrarianBee - 知识检索与第二大脑
负责从 `AGENT_DEVELOPMENT_KNOWLEDGE_BASE.md` 和 `VectorVault` 中检索高质量情报。
Phase 1: LLM 驱动的即时检索，输出真实代码片段而非文本匹配
"""

import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

try:
    from hive.utils.brain_interface import BrainInterface, get_brain
except ImportError:
    BrainInterface = None
    get_brain = None

# 默认知识库路径
KB_PATH = Path("c:/Users/Administrator/.gemini/antigravity/scratch/AGENT_DEVELOPMENT_KNOWLEDGE_BASE.md")

# 本地代码库路径（用于检索真实代码片段）
CODE_REPOS_PATH = Path("c:/Users/Administrator/Documents/Repos")


class LibrarianBee(BaseBee):
    """
    图书管理员蜂 - 蜂巢的第二大脑接口

    Phase 1 能力：
    1. 关键词搜索 (Fast Retrieval) - 保留原有能力
    2. LLM 即时检索 - 对检索结果进行 LLM 总结
    3. 真实代码片段呈现 - 不再只是文本匹配，而是呈现真实可运行的代码
    4. 免疫警告检索 (Immune Warnings) - 保留原有能力
    """

    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.logger = get_logger("LibrarianBee")
        self.kb_content = ""
        self._brain: Optional[BrainInterface] = None
        self._load_kb()

    def _get_brain(self) -> Optional[BrainInterface]:
        """懒加载 BrainInterface 单例"""
        if self._brain is None and BrainInterface is not None and get_brain is not None:
            self._brain = get_brain()
        return self._brain

    def _load_kb(self):
        """加载本地知识库到内存"""
        if KB_PATH.exists():
            try:
                with open(KB_PATH, 'r', encoding='utf-8') as f:
                    self.kb_content = f.read()
                self.logger.info(f"📚 Librarian 已加载知识库: {len(self.kb_content)} chars")
            except Exception as e:
                self.logger.error(f"❌ 加载知识库失败: {e}")

    def specialized_task(self, query: str) -> Dict[str, Any]:
        """
        执行检索或管理任务
        """
        if isinstance(query, dict) and query.get("action") == "record_failure":
            return self.record_failure(query.get("death_cert"))

        self.logger.info(f"🔍 Librarian 正在检索: {query}")

        # 1. 关键词搜索 (Fast Retrieval)
        keyword_hits = self._search_markdown_kb(str(query))

        # 2. Phase 1: 真实代码片段搜索
        code_snippets = self._search_real_code_snippets(str(query))

        # 3. 免疫警告检索 (Immune Warnings)
        immune_warnings = self.get_immune_warnings(str(query))

        # 4. Phase 1: LLM 即时总结（如果有结果）
        llm_summary = None
        if (keyword_hits or code_snippets) and BrainInterface is not None:
            llm_summary = self._llm_summarize(query, keyword_hits, code_snippets)

        return {
            "status": "success",
            "query": query,
            "keyword_hits": keyword_hits,
            "code_snippets": code_snippets,
            "immune_warnings": immune_warnings,
            "llm_summary": llm_summary,
            "summary": (
                f"Found {len(keyword_hits)} KB sections, "
                f"{len(code_snippets)} code snippets, "
                f"{len(immune_warnings)} warnings"
            )
        }

    def _search_real_code_snippets(self, query: str) -> List[Dict[str, Any]]:
        """
        Phase 1: 搜索本地代码库，返回真实代码片段

        根据查询词，在 CODE_REPOS_PATH 中搜索匹配的文件，
        返回真实可运行的代码片段。
        """
        snippets = []
        if not CODE_REPOS_PATH.exists():
            self.logger.debug(f"📚 代码库路径不存在: {CODE_REPOS_PATH}")
            return snippets

        # 根据查询推断搜索路径和关键词
        search_keywords = self._infer_search_keywords(query)
        max_results = 3

        for keyword in search_keywords:
            if len(snippets) >= max_results:
                break

            # 在代码库中搜索匹配的文件
            for file_path in CODE_REPOS_PATH.rglob('*'):
                if len(snippets) >= max_results:
                    break
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {'.py', '.js', '.ts', '.go', '.java', '.md'}:
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if keyword.lower() not in content.lower():
                        continue

                    # 找到匹配，提取上下文片段
                    snippet = self._extract_code_snippet(content, keyword, file_path)
                    if snippet:
                        snippets.append({
                            "file": str(file_path.relative_to(CODE_REPOS_PATH)),
                            "language": file_path.suffix[1:],
                            "matched_keyword": keyword,
                            "snippet": snippet,
                            "score": content.lower().count(keyword.lower())
                        })
                except Exception as e:
                    self.logger.debug(f"📚 读取文件失败 {file_path}: {e}")

        # 按匹配分数排序
        snippets.sort(key=lambda x: x["score"], reverse=True)
        return snippets[:max_results]

    def _infer_search_keywords(self, query: str) -> List[str]:
        """
        Phase 1: 根据自然语言查询推断搜索关键词
        """
        # 常见模式 → 代码关键词映射
        patterns = {
            r"rag|retrieval|rag.*implement": ["RAG", "retrieval", "vector", "embed", "chunk"],
            r"auth|login|jwt|token": ["auth", "jwt", "token", "login", "oauth"],
            r"api|rest|endpoint": ["api", "rest", "endpoint", "router", "handler"],
            r"database|db|sql|query": ["database", "db", "sql", "query", "repository"],
            r"cache|redis|mem": ["cache", "redis", "memory", "store"],
            r"async|await|concurr": ["async", "await", "concurrent", "asyncio"],
            r"test|pytest|unittest": ["test", "pytest", "unittest", "mock"],
            r"docker|container|k8s": ["docker", "container", "kubernetes", "k8s"],
            r"api.*key|secret|credential": ["api_key", "secret", "credential", "config"],
            r"websocket|sse|realtime": ["websocket", "sse", "realtime", "socket"],
            r"auth.*jwt|bearer|oauth": ["auth", "jwt", "bearer", "oauth", "token"],
        }

        query_lower = query.lower()
        keywords = []

        for pattern, words in patterns.items():
            if re.search(pattern, query_lower):
                keywords.extend(words)

        # 如果没有匹配，使用查询词本身
        if not keywords:
            # 提取英文单词作为关键词
            words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', query)
            keywords = [w for w in words if len(w) > 2][:3]

        return keywords[:3]  # 最多3个关键词

    def _extract_code_snippet(
        self,
        content: str,
        keyword: str,
        file_path: Path
    ) -> Optional[str]:
        """
        从文件中提取包含关键词的代码片段（前8行 + 后8行上下文）
        """
        lines = content.split('\n')
        keyword_lower = keyword.lower()

        for i, line in enumerate(lines):
            if keyword_lower in line.lower():
                start = max(0, i - 2)
                end = min(len(lines), i + 10)
                snippet_lines = lines[start:end]

                # 添加行号
                result = []
                for j, l in enumerate(snippet_lines):
                    line_no = start + j + 1
                    prefix = ">>> " if l.strip().startswith(('#', '//')) else "    "
                    result.append(f"{line_no:4d}: {l.rstrip()}")

                return '\n'.join(result)

        return None

    def _llm_summarize(
        self,
        query: str,
        keyword_hits: List[Dict[str, Any]],
        code_snippets: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Phase 1: 调用 LLM 对检索结果进行总结和回答
        """
        brain = self._get_brain()
        if brain is None:
            return None

        # 构建上下文
        context_parts = [f"用户查询: {query}\n"]

        if code_snippets:
            context_parts.append("## 真实代码片段:\n")
            for snip in code_snippets[:2]:
                context_parts.append(
                    f"文件: {snip['file']} (语言: {snip['language']})\n"
                    f"```\n{snip['snippet']}\n```\n"
                )

        if keyword_hits:
            context_parts.append("## 知识库匹配:\n")
            for hit in keyword_hits[:2]:
                context_parts.append(f"### {hit['title']}\n{hit['preview']}\n")

        context = "\n".join(context_parts)

        system_prompt = """你是一位资深的技术图书馆管理员。
你的职责是根据检索到的真实代码片段和知识库内容，
用中文为用户的问题提供准确、实用的回答。

要求：
1. 直接引用代码片段中的关键代码
2. 解释代码的工作原理
3. 指出使用场景和注意事项
4. 如果代码不完整，给出补充建议

只回答与用户问题相关的内容，不要泛泛而谈。"""

        prompt = f"用户问题: {query}\n\n{context}\n\n请根据以上信息回答用户的问题。"

        try:
            result = brain.consult(prompt, system_prompt, model="glm-4-flash")
            if result:
                self.logger.info(f"📚 Librarian LLM 总结完成")
                return result
        except Exception as e:
            self.logger.error(f"📚 LLM 总结失败: {e}")

        return None

    def record_failure(self, death_cert: Dict[str, Any]) -> Dict[str, Any]:
        """
        记录失败经历到知识库 (免疫记忆存储)
        """
        if not death_cert:
            return {"status": "error", "message": "Missing death certificate"}
            
        target = death_cert.get("target_name", "Unknown")
        error = death_cert.get("error_message", "Unknown Error")
        bee_type = death_cert.get("bee_type", "UnknownBee")
        
        self.logger.info(f"🛡️ [Immune Memory] 记录失败因子: {target} | {bee_type}")
        
        # 构造 Markdown 条目
        entry = (
            f"\n### 🛡️ Pathogen Archive: {target} ({bee_type})\n"
            f"- **Timestamp**: {death_cert.get('timestamp')}\n"
            f"- **Error**: {error}\n"
            f"- **Traceback Snippet**: `{death_cert.get('traceback_str', '')[:200]}...`\n"
            f"- **Immune Response**: Avoid this specific state in future generations.\n"
        )
        
        try:
            # 找到或创建 ## 🛡️ Immune Memory 章节
            kb_text = KB_PATH.read_text(encoding='utf-8')
            immune_header = "## 🛡️ Immune Memory (Pathogen Archive)"
            
            if immune_header not in kb_text:
                kb_text += f"\n\n{immune_header}\n*This section contains records of failed biological mutations to prevent regressions.*\n"
            
            # 追加记录
            kb_text += entry
            KB_PATH.write_text(kb_text, encoding='utf-8')
            
            # 更新内存
            self.kb_content = kb_text
            return {"status": "success", "message": "Immune memory recorded"}
        except Exception as e:
            self.logger.error(f"❌ 记录免疫记忆失败: {e}")
            return {"status": "error", "message": str(e)}

    def get_immune_warnings(self, context: str) -> List[str]:
        """
        检索相关的失败经历，告诫后代
        """
        warnings = []
        if "🛡️ Pathogen Archive" not in self.kb_content:
            return warnings
            
        # 简单匹配：搜索上下文中提到的项目名或错误类型
        immune_section = self.kb_content.split("## 🛡️ Immune Memory")[1]
        archives = immune_section.split("### 🛡️ Pathogen Archive:")
        
        for archive in archives[1:]: # 第一个是标题前的说明
            if any(keyword.lower() in archive.lower() for keyword in context.split()):
                lines = archive.split("\n")
                summary = lines[0].strip()
                error_info = next((l for l in lines if "**Error**" in l), "Unknown Error")
                warnings.append(f"PAST FAILURE [{summary}]: {error_info}")
                
        return warnings[:3] # 最多返回三个警告

    def _search_markdown_kb(self, query: str) -> List[Dict[str, str]]:
        """在 Markdown 知识库中进行简单的块级搜索"""
        hits = []
        if not self.kb_content:
            return hits
            
        # 简单切分：按二级标题切分项目
        sections = re.split(r'\n## ', self.kb_content)
        
        for section in sections:
            if query.lower() in section.lower():
                # 提取标题
                lines = section.split('\n')
                title = lines[0].strip()
                preview = "\n".join(lines[:10]) + "..." # 取前10行作为摘要
                
                hits.append({
                    "title": title,
                    "preview": preview,
                    "full_content": "## " + section
                })
                
                if len(hits) >= 5: # 限制返回数量
                    break
        
        return hits

    def consult_library(self, query: str) -> str:
        """对外的高级接口，直接返回人类可读的答案"""
        results = self.specialized_task(query)
        
        if not results['keyword_hits']:
            return "📚 Librarian: 抱歉，我在知识库中没找到相关信息。"
            
        response = f"📚 Librarian: 找到了 {len(results['keyword_hits'])} 个相关项目：\n\n"
        for hit in results['keyword_hits']:
            response += f"--- **{hit['title']}** ---\n{hit['preview']}\n\n"
            
        return response
