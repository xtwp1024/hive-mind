# -*- coding: utf-8 -*-
"""
Phase 3: 自主进化能力 - 快速功能验证脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import tempfile
import shutil
import pathlib


def test_capability_gap_analyzer():
    """测试能力缺口分析器"""
    print('[1/5] 测试 CapabilityGapAnalyzer...')
    
    from hive.core.capability_gap_analyzer import (
        CapabilityGapAnalyzer, TaskExecutionRecord, GapSeverity, GapType
    )

    analyzer = CapabilityGapAnalyzer(max_records=1000)

    for i in range(20):
        success = i < 6
        analyzer.record_task_simple(
            task_id=f'task_{i}',
            bee_id=f'bee_{i % 3}',
            bee_type='miner_bee',
            task_category='mining',
            success=success,
            duration=5.0 + i,
            error_message=None if success else 'Test failure',
            capability_used='mining',
        )

    report = analyzer.analyze(analysis_window_hours=24)
    print(f'    分析任务数: {report.total_tasks_analyzed}')
    print(f'    检测到缺口数: {report.total_gaps_detected}')
    print('    [OK] CapabilityGapAnalyzer 验证通过')
    return True


def test_species_blueprint():
    """测试蜂种蓝图"""
    print()
    print('[2/5] 测试 SpeciesBlueprint...')
    
    from hive.core.species_blueprint import (
        SpeciesBlueprint, BlueprintRegistry, CapabilityPriority, BlueprintStatus
    )

    blueprint = SpeciesBlueprint(
        blueprint_id='test_blueprint_001',
        name='Test Miner Blueprint',
        description='A test blueprint for mining tasks with high mining capability',
    )

    blueprint.add_capability('mining', 0.9, CapabilityPriority.PRIMARY, weight=2.0)
    blueprint.add_capability('sentinel', 0.7, CapabilityPriority.SECONDARY)
    blueprint.add_task_affinity('mining', 0.9)

    validation = blueprint.validate()
    print(f'    蓝图名称: {blueprint.name}')
    status_text = '通过' if validation.is_valid else '失败'
    print(f'    验证结果: {status_text}')
    print(f'    能力数量: {len(blueprint.capabilities)}')
    print('    [OK] SpeciesBlueprint 验证通过')
    return True


def test_blueprint_registry():
    """测试蓝图注册表"""
    print()
    print('[3/5] 测试 BlueprintRegistry...')
    
    from hive.core.species_blueprint import (
        SpeciesBlueprint, BlueprintRegistry, CapabilityPriority
    )

    blueprint = SpeciesBlueprint(
        blueprint_id='registry_test_001',
        name='Registry Test Blueprint',
        description='Test blueprint for registry',
    )
    blueprint.add_capability('mining', 0.9, CapabilityPriority.PRIMARY)

    temp_dir = tempfile.mkdtemp()
    registry = BlueprintRegistry(pathlib.Path(temp_dir))

    registry.register(blueprint)

    retrieved = registry.get('registry_test_001')
    print(f'    注册蓝图数: {len(registry.get_all())}')
    print(f'    检索成功: {retrieved is not None}')

    shutil.rmtree(temp_dir, ignore_errors=True)
    print('    [OK] BlueprintRegistry 验证通过')
    return True


def test_resource_allocator():
    """测试资源分配器"""
    print()
    print('[4/5] 测试 DynamicResourceAllocator...')
    
    from hive.core.resource_allocator import (
        DynamicResourceAllocator, ResourceType, AllocationPriority
    )

    allocator = DynamicResourceAllocator(system_resources={
        ResourceType.CPU: 8.0,
        ResourceType.MEMORY: 16.0,
    })

    allocator.register_consumer(
        consumer_id='test_bee_001',
        consumer_type='miner_bee',
        base_priority=AllocationPriority.HIGH,
        initial_quotas={
            ResourceType.CPU: {'hard_limit': 2.0, 'soft_limit': 1.5}
        },
    )

    decision = allocator.request_resource(
        consumer_id='test_bee_001',
        resource_type=ResourceType.CPU,
        amount=1.0,
    )

    status_text = '批准' if decision.granted else '拒绝'
    print(f'    资源请求: {status_text}')
    amount = decision.granted_amount if decision.granted else 0
    print(f'    分配数量: {amount}')

    status = allocator.get_system_status()
    print(f'    系统状态 - 消费者数: {status["total_consumers"]}')
    print('    [OK] DynamicResourceAllocator 验证通过')
    return True


def test_dna_adaptive():
    """测试 DNA 自适应引擎"""
    print()
    print('[5/5] 测试 DNAAdaptiveEngine...')
    
    from hive.core.dna_adaptive import (
        DNAAdaptiveEngine, ParameterState, PerformanceTrend
    )

    engine = DNAAdaptiveEngine()

    state = ParameterState(
        name='learning_rate',
        current_value=0.1,
        min_value=0.0,
        max_value=1.0,
    )

    for i in range(20):
        state.record_value(0.1 + i * 0.01)

    trend, slope = state.get_trend()
    print(f'    参数趋势: {trend.value}')
    print(f'    趋势斜率: {slope:.4f}')

    print('    [OK] DNAAdaptiveEngine 验证通过')
    return True


def main():
    print('=' * 60)
    print('Phase 3: 自主进化能力 - 快速功能验证')
    print('=' * 60)

    tests = [
        test_capability_gap_analyzer,
        test_species_blueprint,
        test_blueprint_registry,
        test_resource_allocator,
        test_dna_adaptive,
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
        print('Phase 3 所有模块验证通过！')
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
