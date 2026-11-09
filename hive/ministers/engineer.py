#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Engineer (工程蜂) - 任务与架构指挥官
"""

import logging
import random
import time
import multiprocessing
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from hive.config.settings import get_settings
from hive.core.dna import BeeDNA
from hive.core.auto_feast import get_auto_feast
from hive.core.interfaces import Task, TaskResult
from hive.ministers import BaseMinister
from hive.utils.relevance import RelevanceConfig, RelevanceScorer
from hive.utils.targeting import load_targeting_keywords
from hive.utils.security import is_safe_gene, DANGEROUS_CHARS



logger = logging.getLogger("Hive.Engineer")

class EngineerBee(BaseMinister):
    """工程蜂 - 指挥官"""
    
    def __init__(self, dna: BeeDNA, id: str, result_queue: multiprocessing.Queue):
        super().__init__(dna, id, result_queue)
        self.auto_feast = get_auto_feast()
        self.engineering_db = {"architectures": [], "designs": [], "implementations": [], "technical_debt": []}
        self.mutation_pool = [
            'standard', 'seeker', 'sentinel', 'librarian', 'architect', 
            'miner', 'antibody', 'darwin', 'code_generator',
            'vortex', 'primearchitect', 'cipher', 'oracle', 'prism', 
            'echo', 'ultracipher', 'primeseeker', 'minerantibody', 
            'primeminer', 'primeantibody', 'primeminerlibrarian',
            'validator', 'logger', 'counter', 'pinger', 'searcher', 
            'texter', 'watcher', 'config', 'cyberminer', 'cyberprimeminer',
            'cybersentinel', 'cybersentinelminer', 'quantumcybersentinel',
            'librariansentinel'
        ]
        self._mutation_index = 0
        self.trusted_species = ['vortex', 'primearchitect', 'engineer', 'code_generator']
        logger.info(f"🏗️ 工程蜂 ({self.id}) 已就位 - 变异池: {len(self.mutation_pool)} 种")

    def assign_tasks(self, strategy_topics: List[str] = None, capacity: int = 5) -> List[BeeDNA]:
        """分配任务，生成工蜂 DNA"""
        dna_batch = []
        
        # 0. 优先处理伤亡
        casualty_tasks = self._check_morgue(limit=max(1, capacity // 2))
        if casualty_tasks:
            dna_batch.extend(casualty_tasks)
            logger.info(f"🚑 [救援] 启动了 {len(casualty_tasks)} 个修复任务")
            
        remaining = capacity - len(dna_batch)
        if remaining <= 0: return dna_batch
        
        # 1. 优先处理积压
        self._process_backlog(dna_batch, remaining)
        
        remaining = capacity - len(dna_batch)
        if remaining <= 0: return dna_batch

        # 2. 主动狩猎
        self._perform_active_hunting(dna_batch, remaining, strategy_topics)

        return dna_batch

    def _process_backlog(self, dna_batch: List[BeeDNA], limit: int):
        backlog_dna = self.auto_feast.feast(limit=limit)
        if backlog_dna:
            for dna in backlog_dna:
                dna.mutation_gene = self._roll_diverse_mutation()
            dna_batch.extend(backlog_dna)
            logger.info(f"🏗️ [积压] 转换了 {len(backlog_dna)} 个积压任务")

    def _perform_active_hunting(self, dna_batch: List[BeeDNA], limit: int, strategy_topics: List[str]):
        topics = self._determine_hunting_topics(strategy_topics)
        logger.info(f"🏹 积压不足，发动主动狩猎! 目标: {topics}")
        
        from hive.utils.github_hunter import GithubHunter
        hunter = GithubHunter()
        prey_list = hunter.hunt_sync(topics, count=limit * 2)
        
        unique_prey = self._filter_unique_prey(prey_list, limit)
        if not unique_prey: return

        # 🆕 自适应进化：根据熏物语言分布更新下一轮狩猎策略
        self._update_intel_strategy(unique_prey, topics)
        self._create_hunter_tasks(dna_batch, unique_prey)
        logger.info(f"🏗️ [狩猎] 生成了 {len(unique_prey)} 个新任务")

    def _update_intel_strategy(self, prey_list: List[Dict], used_topics: List[str]) -> None:
        """将狩猎结果写回 intel_strategy.json，下次更聪明地狩猎"""
        strategy_file = Path("intel_strategy.json")
        try:
            existing = {"hunting_keywords": used_topics, "runs": 0, "top_languages": {}}
            if strategy_file.exists():
                with open(strategy_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)

            # 统计语言分布
            for prey in prey_list:
                lang = prey.get("language", "unknown") or "unknown"
                existing["top_languages"][lang] = existing["top_languages"].get(lang, 0) + 1

            # 每 3 次狩猎进化一次关键词
            existing["runs"] = existing.get("runs", 0) + 1
            if existing["runs"] % 3 == 0:
                top_langs = sorted(existing["top_languages"].items(), key=lambda x: -x[1])[:2]
                lang_keywords = [f"{lang} agent" for lang, _ in top_langs if lang != "unknown"]
                base = ["multi-agent system", "llm orchestration", "ai tool use"]
                existing["hunting_keywords"] = (lang_keywords + base)[:5]
                logger.info(f"🧠 [自适应] 狩猎关键词进化为: {existing['hunting_keywords']}")

            with open(strategy_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"更新狩猎策略失败: {e}")

    def _determine_hunting_topics(self, strategy_topics: List[str]) -> List[str]:
        if strategy_topics: return strategy_topics
        
        strategy_file = Path("intel_strategy.json")
        default_topics = ["ai agent", "llm framework", "multi-agent"]
        
        if strategy_file.exists():
            try:
                with open(strategy_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("hunting_keywords", default_topics)
            except (json.JSONDecodeError, IOError): pass
        return default_topics

    def _filter_unique_prey(self, prey_list: List[Dict], limit: int) -> List[Dict]:
        unique_prey = []
        new_urls = []

        settings = get_settings()
        languages = [lang.strip() for lang in settings.RELEVANCE_LANGUAGES.split(",") if lang.strip()]
        scorer = RelevanceScorer(
            RelevanceConfig(
                min_score=settings.GITHUB_RELEVANCE_MIN_SCORE,
                strict=settings.GITHUB_RELEVANCE_STRICT,
                preferred_languages=languages,
            )
        )

        for prey in prey_list:
            url = prey["url"]
            if url in self.auto_feast.history_cache:
                continue

            score = prey.get("relevance_score")
            if score is None:
                score = scorer.score(
                    name=prey.get("name", ""),
                    description=prey.get("description", ""),
                    keywords=load_targeting_keywords(),
                    language=prey.get("language", ""),
                    stars=prey.get("stars", 0),
                )
                prey["relevance_score"] = score


            if scorer.strict and not scorer.is_relevant(score):
                continue

            unique_prey.append(prey)
            new_urls.append(url)
            if len(unique_prey) >= limit:
                break
        
        if new_urls:
            self.auto_feast.persist_to_history(new_urls)
            logger.info(f"💾 [去重] 新增 {len(new_urls)} 个历史记录")
        return unique_prey


    def _create_hunter_tasks(self, dna_batch: List[BeeDNA], unique_prey: List[Dict]):
        from hive.utils.immune_archive import get_immune_warnings

        for i, prey in enumerate(unique_prey):
            warnings = get_immune_warnings(prey["name"])
            if warnings:
                logger.warning(f"🛡️ [免疫警告] {prey['name']} 发现 {len(warnings)} 个警告")

            gene = self._roll_diverse_mutation()
            if not gene or not isinstance(gene, str):
                gene = "standard"
            if not is_safe_gene(gene):
                gene = "standard"
            dna = BeeDNA(
                species="bee", role="worker",
                id=f"Hunter_Gen1_{int(time.time())}_{i}",
                target_name=prey["name"], target_url=prey["url"],
                mutation_gene=gene,
                can_merge_back=(gene.lower() in self.trusted_species),
                metadata={"immune_warnings": warnings}
            )
            dna_batch.append(dna)

    def _check_morgue(self, limit: int = 2) -> List[BeeDNA]:
        """检查停尸房，为最近的死者复仇"""
        from hive.config.settings import get_settings
        tasks = []
        try:
            settings = get_settings()
            morgue_dir = settings.BASE_DIR / "hive" / "morgue"
            if not morgue_dir.exists(): return []
                
            certs = sorted(morgue_dir.glob("death_cert_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for i, cert_path in enumerate(certs):
                if i >= limit: break
                task = self._process_single_death_cert(cert_path, i)
                if task: tasks.append(task)
        except Exception as e:
            logger.error(f"停尸房巡检失败: {e}")
        return tasks

    def _process_single_death_cert(self, cert_path: Path, index: int) -> Optional[BeeDNA]:
        try:
            with open(cert_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            target_name = data.get('target_name', 'Unknown')
            if target_name == 'Unknown': return None
                 
            new_path = cert_path.with_name(cert_path.name.replace("death_cert_", "processing_"))
            cert_path.rename(new_path)
            
            from hive.utils.immune_archive import get_immune_warnings

            return BeeDNA(
                species="bee", role="medic",
                id=f"Medic_{int(time.time())}_{index}",
                target_name=target_name,
                target_url=data.get('dna_snapshot', {}).get('target_url', ''),
                mutation_gene="darwin",
                can_merge_back=True,
                metadata={
                    "mission": "self_healing",
                    "death_cert_path": str(new_path),
                    "error_context": data.get('error_message', ''),
                    "immune_warnings": get_immune_warnings(target_name)
                }
            )
        except Exception as e:
            logger.error(f"处理死亡证明失败 {cert_path}: {e}")
            return None

    async def assign_task(self, task: Task) -> Optional[str]:
        logger.info(f"🏗️ [{self.name}] 收到任务请求: {task.type}")
        return None

    def get_bee_status(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "active": self.is_active,
            "mutation_pool_size": len(self.mutation_pool)
        }

    def get_statistics(self) -> dict[str, Any]:
        stats = self.get_minister_status()
        stats.update({
            "history_cache_size": len(self.auto_feast.history_cache),
            "total_assessments": len(self.engineering_db.get("architectures", [])),
            "total_strategies": len(self.strategies)
        })
        return stats

    def _roll_mutation(self) -> str:
        if random.random() < 0.5:
            return random.choice(self.mutation_pool)
        return "standard"

    def _roll_diverse_mutation(self) -> str:
        gene = self.mutation_pool[self._mutation_index % len(self.mutation_pool)]
        self._mutation_index += 1
        return gene

_engineer_instance = None
_engineer_lock = __import__('threading').Lock()

def get_engineer(result_queue: multiprocessing.Queue = None) -> EngineerBee:
    global _engineer_instance
    if _engineer_instance is None:
        with _engineer_lock:
            if _engineer_instance is None:
                from hive.core.dna import BeeDNA
                virt_dna = BeeDNA(species="Minister", role="Engineer", id="ENGINEER-CORE")
                _engineer_instance = EngineerBee(virt_dna, "Engineer", result_queue)
    return _engineer_instance
