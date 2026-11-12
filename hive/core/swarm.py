# -*- coding: utf-8 -*-
"""
Swarm - 蜂群主类

管理所有蜂群原子、簇，知识库，实现万年繁衍。
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from hive.core.attraction import ClusterFormation
from hive.core.bee_atom import AtomState, BeeAtom
from hive.core.cluster import ClusterManager
from hive.core.knowledge_base import KnowledgeBase


class SwarmConfig:
    """蜂群配置"""

    INITIAL_POPULATION = 100
    MUTATION_RATE = 0.1
    ELITE_COUNT = 10
    MAX_OFFSPRING_PER_GEN = 50
    MIN_FITNESS = 0.1
    MAX_AGE = 100
    AGE_DEATH_CHANCE = 0.05
    ACTIVE_THRESHOLD = 1000
    MAX_CLUSTERS = 100
    CHECKPOINT_INTERVAL = 1000
    MAX_POPULATION = 5000  # 防止种群无界增长

    # 任务配置
    TASKS_PER_GEN = 20
    TASK_TIMEOUT = 60

    # 汇报配置
    HOURLY_REPORT = True  # 开启每小时汇报


@dataclass
class SwarmMetrics:
    """蜂群指标"""

    generation: int = 0
    total_atoms: int = 0
    active_atoms: int = 0
    dormant_atoms: int = 0
    retired_atoms: int = 0
    avg_fitness: float = 0.0
    max_fitness: float = 0.0
    avg_age: float = 0.0
    newborns: int = 0
    deaths: int = 0
    clusters: int = 0
    knowledge_size: int = 0


class Swarm:
    """
    蜂群 - 万年繁衍系统

    核心理念：
    - 最小原子：每个原子独立运作
    - 自由结合：根据规则动态形成簇
    - 涌现智能：从局部交互涌现全局智能
    """

    def __init__(self, config: SwarmConfig | None = None):
        self.config = config or SwarmConfig()
        self.atoms: dict[str, BeeAtom] = {}
        self.cluster_manager = ClusterManager()
        self.knowledge_base = KnowledgeBase()
        self.generation = 0
        self.newborns = 0
        self.deaths = 0
        self.metrics = SwarmMetrics()

        # 汇报系统
        self.reporter = None
        self.last_report_gen = -1

        self._initialize_population()

        # 初始化汇报
        if self.config.HOURLY_REPORT:
            self._init_reporter()

    def _initialize_population(self) -> None:
        """初始化种群"""
        for _ in range(self.config.INITIAL_POPULATION):
            atom = BeeAtom(born_at=0)
            atom.fitness = random.uniform(0.3, 0.7)
            self.atoms[atom.id] = atom
        self.newborns = self.config.INITIAL_POPULATION

    def _init_reporter(self):
        """初始化汇报系统"""
        try:
            from hive.core.reporting.swarm_reporter import SwarmReporter

            self.reporter = SwarmReporter(self)
            self._generate_report()
        except Exception as e:
            print(f"汇报系统初始化失败: {e}")

    def _generate_report(self):
        """生成汇报"""
        if self.reporter and self.generation != self.last_report_gen:
            # 每100代生成一次汇报（模拟每小时）
            if self.generation % 100 == 0:
                self.reporter.save_report()
                self.last_report_gen = self.generation

    def evolve_once(self) -> None:
        """进化一代"""
        self.generation += 1
        self.newborns = 0
        self.deaths = 0

        self.knowledge_base.set_generation(self.generation)

        # 0. 执行任务（采集知识）
        self._execute_tasks()

        # 1. 老化所有原子
        self._aging()
        self._evaluate_fitness()
        self._form_clusters()

        elites = self._select_elites()
        self._reproduce(elites)
        self._update_metrics()  # 在淘汰前统计，确保 retired/dead 计数准确
        self._cull_weak()

        # 生成汇报
        self._generate_report()

    def _execute_tasks(self) -> None:
        """执行任务：让原子去 GitHub 采集代码"""

        # 检查是否有 GitHub Token
        from hive.config.settings import get_settings

        has_token = bool(get_settings().GITHUB_TOKEN)

        if not has_token:
            # 模拟模式 - 不实际调用 GitHub
            self._execute_tasks_mock()
            return

        try:
            from hive.core.tasks.github_harvester import DEFAULT_KEYWORDS, GitHubHarvester
        except ImportError:
            print("GitHub 采集器未安装，跳过任务执行")
            return

        active_atoms = [a for a in self.atoms.values() if a.is_active]
        if not active_atoms:
            return

        workers = active_atoms[: self.config.TASKS_PER_GEN]

    def _execute_tasks_mock(self) -> None:
        """模拟任务执行（无 GitHub Token 时使用）"""
        active_atoms = [a for a in self.atoms.values() if a.is_active]
        if not active_atoms:
            return

        workers = active_atoms[: self.config.TASKS_PER_GEN]

        # 模拟知识 - 每个代表一种技能/语言
        mock_repos = [
            {
                "name": "system-design-primer",
                "language": "system_design",
                "description": "System design",
                "stars": 335663,
            },
            {
                "name": "awesome-python",
                "language": "python",
                "description": "Python libs",
                "stars": 283346,
            },
            {
                "name": "transformers",
                "language": "machine_learning",
                "description": "ML transformers",
                "stars": 125000,
            },
            {"name": "fastapi", "language": "web", "description": "Fast web", "stars": 75000},
            {
                "name": "pytorch",
                "language": "deep_learning",
                "description": "PyTorch ML",
                "stars": 85000,
            },
            {"name": "asyncpg", "language": "async", "description": "Async DB", "stars": 45000},
            {"name": "kubernetes", "language": "kubernetes", "description": "K8s", "stars": 120000},
            {"name": "react", "language": "frontend", "description": "React UI", "stars": 220000},
            {"name": "rust", "language": "rust", "description": "Rust systems", "stars": 95000},
            {"name": "go", "language": "go", "description": "Go services", "stars": 110000},
        ]

        success = 0
        new_species = 0

        for atom in workers:
            try:
                # 慢速吸收：每个原子只采集一个
                repo = random.choice(mock_repos)

                # 更新原子
                atom.tasks_completed += 1
                atom.fitness = min(1.0, atom.fitness + 0.1)

                # 存入知识库
                self.knowledge_base.add_knowledge(
                    content=f"{repo['name']}: {repo['description']}",
                    importance=min(1.0, repo["stars"] / 100000 + 0.3),
                    source_atom_id=atom.id,
                )

                # 关键：每吸收一个，创建对应蜂种
                new_atom = self._create_specialized_atom(repo, atom)
                if new_atom:
                    new_species += 1

                success += 1

            except Exception as e:
                print(f"  任务失败: {e}")

        if success > 0:
            kb_stats = self.knowledge_base.get_stats()
            print(
                f"  慢速吸收: {success}原子, 诞生{new_species}新蜂种, 知识+{kb_stats['total_knowledge']}"
            )

    def _create_specialized_atom(self, repo: dict, parent: BeeAtom) -> BeeAtom | None:
        """根据采集的知识创建对应蜂种"""
        if len(self.atoms) >= self.config.MAX_POPULATION:
            return None

        language = repo.get("language", "unknown")

        # 映射语言到蜂种角色
        role_map = {
            "python": "miner",
            "machine_learning": "analyzer",
            "deep_learning": "analyzer",
            "system_design": "architect",
            "web": "builder",
            "async": "orchestrator",
            "kubernetes": "deployer",
            "frontend": "builder",
            "rust": "engineer",
            "go": "engineer",
        }

        role = role_map.get(language, "worker")

        # 创建新原子（对应蜂种）
        child = BeeAtom(born_at=self.generation)

        # 设置对应的 DNA 角色
        if child.dna:
            child.dna.role = role
            child.dna.species = f"hive-{language}"
            child.dna.target_name = repo.get("name", "")

        # 新蜂种继承父本部分适应度
        child.fitness = parent.fitness * 0.8

        # 添加到种群
        self.atoms[child.id] = child

        return child

    def _aging(self) -> None:
        """老化"""
        for atom in self.atoms.values():
            if not atom.is_alive:
                continue
            atom.aging()
            if atom.age > self.config.MAX_AGE:
                if random.random() < self.config.AGE_DEATH_CHANCE:
                    atom.die()
                    self.deaths += 1

    def _evaluate_fitness(self) -> None:
        """评估适应度"""
        for atom in self.atoms.values():
            if not atom.is_alive:
                continue
            if atom.tasks_completed > 0:
                atom.fitness = min(1.0, atom.fitness + 0.01)
            if atom.state == AtomState.ACTIVE:
                atom.fitness = min(1.0, atom.fitness + 0.005)

    def _form_clusters(self) -> None:
        """自由结合形成簇"""
        active_atoms = [a for a in self.atoms.values() if a.is_active]
        if not active_atoms:
            return

        cluster_lists = ClusterFormation.form_clusters(active_atoms)
        self.cluster_manager.update_clusters(cluster_lists, self.generation)

        for cluster in self.cluster_manager.get_active_clusters():
            for atom_id in cluster.member_ids:
                if atom_id in self.atoms:
                    for other_id in cluster.member_ids:
                        if other_id != atom_id:
                            self.atoms[atom_id].add_connection(other_id)

    def _select_elites(self) -> list[BeeAtom]:
        """选择精英"""
        alive = [a for a in self.atoms.values() if a.is_alive]
        if not alive:
            return []
        alive.sort(key=lambda a: a.fitness, reverse=True)
        return alive[: self.config.ELITE_COUNT]

    def _reproduce(self, elites: list[BeeAtom]) -> None:
        """繁殖"""
        if not elites:
            return

        if len(self.atoms) >= self.config.MAX_POPULATION:
            return

        max_offspring = min(
            self.config.MAX_OFFSPRING_PER_GEN,
            len(elites) * 3,
            self.config.MAX_POPULATION - len(self.atoms),
        )

        for _ in range(max_offspring):
            parent = random.choice(elites)
            child = self._create_offspring(parent)
            self.atoms[child.id] = child
            self.newborns += 1

    def _create_offspring(self, parent: BeeAtom) -> BeeAtom:
        """创建后代"""
        child = BeeAtom(born_at=self.generation)

        if parent.dna:
            from hive.core.dna import BeeDNA

            child.dna = BeeDNA(
                species=parent.dna.species,
                role=parent.dna.role,
                id=child.id,
                target_name=parent.dna.target_name,
                generation=parent.dna.generation + 1,
            )
            if random.random() < self.config.MUTATION_RATE:
                self._mutate_dna(child.dna)

        child.fitness = parent.fitness * 0.9
        parent.offspring_count += 1
        return child

    def _mutate_dna(self, dna) -> None:
        """DNA变异"""
        mutations = ["darwin", "miner", "seeker", "architect"]
        if random.random() < 0.3:
            dna.role = random.choice(mutations)
        for cap in [
            "mining_capability",
            "gathering_capability",
            "architectural_capability",
            "librarian_capability",
        ]:
            current = getattr(dna, cap, 0.5)
            delta = random.gauss(0, 0.1)
            setattr(dna, cap, max(0.0, min(1.0, current + delta)))

    def _cull_weak(self) -> None:
        """淘汰弱者并从种群中移除死亡原子"""
        dead_ids = []
        for atom_id, atom in self.atoms.items():
            if not atom.is_alive:
                dead_ids.append(atom_id)
                continue
            if atom.fitness < self.config.MIN_FITNESS:
                atom.die()
                self.deaths += 1
                dead_ids.append(atom_id)
        for atom_id in dead_ids:
            del self.atoms[atom_id]

    def _update_metrics(self) -> None:
        """更新指标"""
        atoms = list(self.atoms.values())
        alive = [a for a in atoms if a.is_alive]

        self.metrics.generation = self.generation
        self.metrics.total_atoms = len(atoms)
        self.metrics.active_atoms = len([a for a in atoms if a.state == AtomState.ACTIVE])
        self.metrics.dormant_atoms = len([a for a in atoms if a.state == AtomState.DORMANT])
        self.metrics.retired_atoms = len([a for a in atoms if a.state == AtomState.RETIRED])

        if alive:
            fitnesses = [a.fitness for a in alive]
            self.metrics.avg_fitness = sum(fitnesses) / len(fitnesses)
            self.metrics.max_fitness = max(fitnesses)
            ages = [a.age for a in alive]
            self.metrics.avg_age = sum(ages) / len(ages)

        self.metrics.newborns = self.newborns
        self.metrics.deaths = self.deaths
        self.metrics.clusters = len(self.cluster_manager.get_active_clusters())
        self.metrics.knowledge_size = len(self.knowledge_base)

    def get_status(self) -> dict[str, Any]:
        """获取状态"""
        return {
            "generation": self.generation,
            "population": len([a for a in self.atoms.values() if a.is_alive]),
            "metrics": {
                "avg_fitness": self.metrics.avg_fitness,
                "max_fitness": self.metrics.max_fitness,
                "newborns": self.metrics.newborns,
                "deaths": self.metrics.deaths,
                "clusters": self.metrics.clusters,
                "knowledge": self.metrics.knowledge_size,
            },
        }

    def __repr__(self) -> str:
        return f"Swarm(gen={self.generation}, pop={len(self.atoms)}, fitness={self.metrics.avg_fitness:.2f})"
