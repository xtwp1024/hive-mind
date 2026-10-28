# -*- coding: utf-8 -*-
"""
HoneyProcessor - 蜂蜜处理器

专注负责处理 Worker 返回的结果、知识消化和免疫响应。
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("Hive.HoneyProcessor")


class HoneyProcessor:
    """蜂蜜处理器"""

    MAX_HONEY_STOMACH_SIZE = 1000
    _digest_executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self, max_stomach_size: int = MAX_HONEY_STOMACH_SIZE):
        self.max_stomach_size = max_stomach_size
        self.honey_stomach: list[dict[str, Any]] = []
        self.antibody_count = 0
        self.processed_count = 0
        self.digested_count = 0

    def process(self, honey: dict[str, Any]) -> dict[str, Any]:
        """处理单个蜂蜜结果"""
        self.processed_count += 1

        if len(self.honey_stomach) >= self.max_stomach_size:
            if honey.get("status") != "SUCCESS":
                self.trigger_immune_response(honey)
                return {"offspring": None}
            else:
                self._digest_knowledge(honey)
                dna = honey.get("dna") or None
                return {"offspring": dna if dna else None}

        self.honey_stomach.append(honey)

        if honey.get("status") != "SUCCESS":
            self.trigger_immune_response(honey)
            return {"offspring": None}

        self._digest_knowledge(honey)
        dna = honey.get("dna") or None
        return {"offspring": dna if dna else None}

    def process_batch(self, payloads: list) -> list[dict[str, Any]]:
        """批量处理队列中的结果（并行消化知识）"""
        from concurrent.futures import as_completed

        processed = 0
        offspring = []
        digest_futures = []

        for honey in payloads:
            result = self.process(honey)
            processed += 1
            if result.get("offspring"):
                offspring.append(result["offspring"])

            # 并行提交知识消化任务（I/O 密集型）
            if honey.get("status") == "SUCCESS":
                future = self._digest_executor.submit(self._digest_knowledge, honey)
                digest_futures.append(future)

        # 等待所有消化任务完成
        for future in as_completed(digest_futures):
            try:
                future.result()  # 收集异常
            except Exception as e:
                logger.debug(f"[HoneyProcessor] 消化任务异常: {e}")

        if processed > 0:
            logger.info(f"[HoneyProcessor] 处理了 {processed} 份花蜜")

        return offspring

    def _digest_knowledge(self, honey: dict[str, Any]) -> None:
        """消化知识到向量库"""
        try:
            from hive.core.vector_vault import get_vector_vault

            vault = get_vector_vault()
            dna = honey.get("dna", {})
            doc_id = f"honey_{dna.get('target_name')}_{int(time.time())}"
            content = f"Project: {dna.get('target_name')}\nMutation: {dna.get('mutation_gene')}\nPayload: {str(honey)}"

            vault.add_document(doc_id, content, metadata=honey)
            self.digested_count += 1
            logger.debug(f"[HoneyProcessor] 消化知识: {dna.get('target_name')}")
        except Exception as e:
            logger.error(f"[HoneyProcessor] 消化失败: {e}")

    def trigger_immune_response(self, bad_honey: dict[str, Any]) -> None:
        """触发免疫响应"""
        self.antibody_count += 1
        source = bad_honey.get("id", "unknown")
        error = bad_honey.get("error", "unknown")
        target = bad_honey.get("dna", {}).get("target_name", "unknown")

        try:
            logger.warning(f"[HoneyProcessor] 免疫响应 #{self.antibody_count}: {source} -> {target}: {error}")
            self._archive_immune_memory(bad_honey)
        except Exception as e:
            logger.warning(f"[HoneyProcessor] 免疫响应错误: {e}")

    def _archive_immune_memory(self, bad_honey: dict[str, Any]) -> None:
        """归档免疫记忆"""
        death_cert = bad_honey.get("death_cert")
        if not death_cert:
            return

        try:
            from hive.utils.immune_archive import archive_failure
            archive_failure(death_cert)
            logger.info("[HoneyProcessor] 免疫记忆已归档")
        except ImportError:
            logger.debug("[HoneyProcessor] 免疫归档模块不可用，跳过归档")
        except Exception as e:
            logger.error(f"[HoneyProcessor] 免疫记忆归档失败: {e}")

    def clear_stomach(self) -> None:
        """清空蜂蜜胃"""
        if self.honey_stomach:
            count = len(self.honey_stomach)
            self.honey_stomach.clear()
            logger.debug(f"[HoneyProcessor] 清空 {count} 条蜂蜜记录")

    def get_status(self) -> dict[str, Any]:
        """获取处理器状态"""
        return {
            "honey_stomach_size": len(self.honey_stomach),
            "processed_count": self.processed_count,
            "digested_count": self.digested_count,
            "antibody_count": self.antibody_count
        }

    def shutdown(self) -> None:
        """关闭处理器，释放资源（供长时运行进程调用）"""
        self._digest_executor.shutdown(wait=True)
        logger.debug("[HoneyProcessor] 消化线程池已关闭")
