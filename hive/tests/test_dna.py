# -*- coding: utf-8 -*-
"""测试 DNA 相关功能"""

import pytest

from hive.core.dna import DNA, BeeDNA, DNAPool


class TestDNA:
    """测试基础 DNA 类"""

    def test_dna_creation(self):
        """测试 DNA 创建"""
        dna = DNA(species="test_species")
        assert dna.species == "test_species"
        assert dna.generation == 1
        assert dna.version == "2.0.0"
        assert dna.fitness == 0.0

    def test_dna_validation_valid(self):
        """测试有效的 DNA 验证"""
        dna = DNA(
            species="test",
            mining_capability=0.5,
            gathering_capability=0.7,
            architectural_capability=0.3,
        )
        assert dna.validate() is True

    def test_dna_validation_invalid(self):
        """测试无效的 DNA 验证"""
        dna = DNA(
            species="test",
            mining_capability=1.5,  # 超出范围
        )
        assert dna.validate() is False

    def test_dna_mutate(self):
        """测试 DNA 突变"""
        dna = DNA(species="test", mutation_rate=0.1)
        mutated = dna.mutate()

        assert mutated.species == dna.species
        assert mutated.generation == dna.generation + 1

    def test_dna_crossover(self):
        """测试 DNA 交叉"""
        dna1 = DNA(species="test", fitness=0.8)
        dna2 = DNA(species="test", fitness=0.6)

        child = dna1.crossover(dna2)

        assert child.species == dna1.species
        assert child.generation > dna1.generation
        assert child.fitness == 0.7  # 平均值


class TestBeeDNA:
    """测试 BeeDNA 类"""

    def test_bee_dna_creation(self):
        """测试 BeeDNA 创建"""
        dna = BeeDNA(
            species="miner",
            id="bee_001",
            role="worker",
            target_name="test_target",
        )
        assert dna.species == "miner"
        assert dna.id == "bee_001"
        assert dna.role == "worker"
        assert dna.target_name == "test_target"

    def test_calculate_fitness(self):
        """测试适应度计算"""
        dna = BeeDNA(species="test")
        dna.success_rate = 0.8
        dna.task_completion_time = 10.0
        dna.error_count = 2

        fitness = dna.calculate_fitness()

        assert 0.0 <= fitness <= 1.0
        assert dna.fitness == fitness

    def test_update_performance(self):
        """测试性能更新"""
        dna = BeeDNA(species="test")

        dna.update_performance(success=True, completion_time=5.0, error=False)

        assert dna.success_rate > 0
        assert dna.task_completion_time == 5.0
        assert dna.error_count == 0

    def test_update_performance_with_error(self):
        """测试带错误的性能更新"""
        dna = BeeDNA(species="test")

        dna.update_performance(success=False, completion_time=3.0, error=True)

        assert dna.success_rate < 1.0
        assert dna.error_count == 1

    def test_bee_dna_mutate(self):
        """测试 BeeDNA 突变"""
        dna = BeeDNA(
            species="miner",
            id="original_id",
            role="worker",
        )

        mutated = dna.mutate()

        assert mutated.species == dna.species
        assert mutated.role == dna.role
        assert mutated.id != dna.id
        assert "mut_" in mutated.id
        assert mutated.generation == dna.generation + 1


class TestDNAPool:
    """测试 DNA 池"""

    def test_pool_creation(self):
        """测试 DNA 池创建"""
        pool = DNAPool()
        assert pool.get_pool_size() == 0

    def test_add_dna(self):
        """测试添加 DNA"""
        pool = DNAPool()
        dna = DNA(species="test", fitness=0.8)

        pool.add(dna)

        assert pool.get_pool_size() == 1

    def test_add_invalid_dna(self):
        """测试添加无效 DNA"""
        pool = DNAPool()
        dna = DNA(species="test", mining_capability=2.0)  # 无效

        pool.add(dna)

        # 无效 DNA 不应被添加
        assert pool.get_pool_size() == 0

    def test_get_best(self):
        """测试获取最佳 DNA"""
        pool = DNAPool()

        # 添加多个 DNA
        for i in range(5):
            dna = DNA(species="test", fitness=i * 0.2)
            pool.add(dna)

        best = pool.get_best("test", count=1)

        assert len(best) == 1
        assert best[0].fitness == 0.8

    def test_evolve_generation(self):
        """测试进化下一代"""
        pool = DNAPool()

        # 添加父代
        for i in range(5):
            dna = DNA(species="test", fitness=0.5 + i * 0.1)
            pool.add(dna)

        offspring = pool.evolve_generation("test", population_size=10)

        assert len(offspring) >= 5  # 至少保留父代数量
        assert all(dna.species == "test" for dna in offspring)

    def test_clear_pool(self):
        """测试清空池"""
        pool = DNAPool()
        pool.add(DNA(species="test"))

        assert pool.get_pool_size() == 1

        pool.clear()

        assert pool.get_pool_size() == 0

    def test_evolve_with_empty_pool(self):
        """测试空池进化"""
        pool = DNAPool()

        offspring = pool.evolve_generation("test", population_size=10)

        assert offspring == []
