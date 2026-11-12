#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蜜蜂类型注册表
集中管理所有蜜蜂类型
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Type

# 添加hive到Python路径
hive_root = Path(__file__).parent.parent
if str(hive_root) not in sys.path:
    sys.path.insert(0, str(hive_root))

# 导入基础类
from hive.bees.base_bee import BaseBee

# 导入变异蜂种
try:
    from hive.bees.variants import (
        CipherBee,
        ConfigBee,
        CopierBee,
        CounterBee,
        CyberMinerBee,
        CyberMinerMinerAntibodyBee,
        CyberSentinelMinerBee,
        EchoBee,
    )
    from hive.bees.variants import (
        LibrarianBee as VariantLibrarianBee,  # Alias to avoid conflict if core has LibrarianBee
    )
    from hive.bees.variants import (
        LoggerBee,
        MergerBee,
        MinerAntibodyBee,
        OracleBee,
        PingerBee,
        PrimeAntibodyBee,
        PrimeArchitectBee,
        PrimeMinerBee,
        PrimeMinerLibrarianBee,
        PrimeSeekerBee,
        PrismBee,
        RenamerBee,
        SearcherBee,
        TexterBee,
        UltraCipherBee,
        ValidatorBee,
        VortexBee,
        VortexPrimeArchitectBee,
        WatcherBee,
    )

    EVOLVED_BEES_IMPORTED = True
    try:
        print("✓ Evolved bees imported successfully")
    except (UnicodeEncodeError, UnicodeDecodeError):
        print("[OK] Evolved bees imported successfully")
except ImportError as e:
    EVOLVED_BEES_IMPORTED = False
    try:
        print(f"⚠️ Warning: Failed to import evolved bees: {e}")
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(f"[WARNING] Failed to import evolved bees: {e}")

# 导入核心蜜蜂类型（从旧系统迁移）
try:
    from hive.bees.antibody_bee import AntibodyBee
    from hive.bees.architect_bee import ArchitectBee
    from hive.bees.darwin_bee import DarwinBee
    from hive.bees.evolver_bee import EvolverBee
    from hive.bees.librarian_bee import LibrarianBee
    from hive.bees.mcp_tool_bee import MCPToolBee
    from hive.bees.miner_bee import MinerBee
    from hive.bees.seeker_bee import SeekerBee
    from hive.bees.sentinel_bee import SentinelBee
    from hive.bees.warden_bee import WardenBee

    CORE_BEES_IMPORTED = True
    try:
        print("✓ 核心蜜蜂导入成功")
    except (UnicodeEncodeError, UnicodeDecodeError):
        print("[OK] 核心蜜蜂导入成功")
except ImportError as e:
    CORE_BEES_IMPORTED = False
    try:
        print(f"⚠️ 警告: 核心蜜蜂导入失败: {e}")
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(f"[WARNING] 核心蜜蜂导入失败: {e}")

# 蜜蜂类型映射
# 标准蜜蜂类型
STANDARD_BEE_TYPES = {
    "Standard": BaseBee,
    "Gatherer": BaseBee,
    "Seeker": SeekerBee if CORE_BEES_IMPORTED else None,
    "Sentinel": SentinelBee if CORE_BEES_IMPORTED else None,
    "Librarian": LibrarianBee if CORE_BEES_IMPORTED else None,
    "Architect": ArchitectBee if CORE_BEES_IMPORTED else None,
    "Miner": MinerBee if CORE_BEES_IMPORTED else None,
    "Darwin": DarwinBee if CORE_BEES_IMPORTED else None,
    "Antibody": AntibodyBee if CORE_BEES_IMPORTED else None,
    "Warden": WardenBee if CORE_BEES_IMPORTED else None,
    "MCPTool": MCPToolBee if CORE_BEES_IMPORTED else None,
    "Evolver": EvolverBee if CORE_BEES_IMPORTED else None,
}

# 变异蜂种
EVOLVED_BEE_TYPES = {
    "Cipher": CipherBee if EVOLVED_BEES_IMPORTED else None,
    "Config": ConfigBee if EVOLVED_BEES_IMPORTED else None,
    "Copier": CopierBee if EVOLVED_BEES_IMPORTED else None,
    "Counter": CounterBee if EVOLVED_BEES_IMPORTED else None,
    "CyberMiner": CyberMinerBee if EVOLVED_BEES_IMPORTED else None,
    "CyberMinerMinerAntibody": CyberMinerMinerAntibodyBee if EVOLVED_BEES_IMPORTED else None,
    "CyberSentinelMiner": CyberSentinelMinerBee if EVOLVED_BEES_IMPORTED else None,
    "Echo": EchoBee if EVOLVED_BEES_IMPORTED else None,
    "VariantLibrarian": VariantLibrarianBee if EVOLVED_BEES_IMPORTED else None,
    "Logger": LoggerBee if EVOLVED_BEES_IMPORTED else None,
    "Merger": MergerBee if EVOLVED_BEES_IMPORTED else None,
    "MinerAntibody": MinerAntibodyBee if EVOLVED_BEES_IMPORTED else None,
    "Oracle": OracleBee if EVOLVED_BEES_IMPORTED else None,
    "Pinger": PingerBee if EVOLVED_BEES_IMPORTED else None,
    "PrimeAntibody": PrimeAntibodyBee if EVOLVED_BEES_IMPORTED else None,
    "PrimeArchitect": PrimeArchitectBee if EVOLVED_BEES_IMPORTED else None,
    "PrimeMiner": PrimeMinerBee if EVOLVED_BEES_IMPORTED else None,
    "PrimeMinerLibrarian": PrimeMinerLibrarianBee if EVOLVED_BEES_IMPORTED else None,
    "PrimeSeeker": PrimeSeekerBee if EVOLVED_BEES_IMPORTED else None,
    "Prism": PrismBee if EVOLVED_BEES_IMPORTED else None,
    "Renamer": RenamerBee if EVOLVED_BEES_IMPORTED else None,
    "Searcher": SearcherBee if EVOLVED_BEES_IMPORTED else None,
    "Texter": TexterBee if EVOLVED_BEES_IMPORTED else None,
    "UltraCipher": UltraCipherBee if EVOLVED_BEES_IMPORTED else None,
    "Validator": ValidatorBee if EVOLVED_BEES_IMPORTED else None,
    "Vortex": VortexBee if EVOLVED_BEES_IMPORTED else None,
    "VortexPrimeArchitect": VortexPrimeArchitectBee if EVOLVED_BEES_IMPORTED else None,
    "Watcher": WatcherBee if EVOLVED_BEES_IMPORTED else None,
}

# 合并所有蜜蜂类型
BEE_SPECIES = {**STANDARD_BEE_TYPES, **EVOLVED_BEE_TYPES}


# 获取有效蜜蜂类型（过滤None值）
@lru_cache(maxsize=1)
def get_valid_species() -> Dict[str, Type[BaseBee]]:
    """获取所有有效的蜜蜂类型"""
    return {k: v for k, v in BEE_SPECIES.items() if v is not None}


# 获取蜜蜂类型数量
def get_species_count() -> tuple:
    """获取蜜蜂类型统计"""
    valid = get_valid_species()
    standard_count = len([k for k in valid.keys() if k in STANDARD_BEE_TYPES])
    evolved_count = len([k for k in valid.keys() if k in EVOLVED_BEE_TYPES])
    return standard_count, evolved_count


# 打印蜜蜂类型统计
def print_species_stats():
    """Print bee species statistics"""
    standard_count, evolved_count = get_species_count()

    print("=" * 60)
    print("🐝 Bee Species Statistics")
    print("=" * 60)
    print(f"Standard Species: {standard_count}")
    print(f"Evolved Species: {evolved_count}")
    print(f"Total: {standard_count + evolved_count}")

    # List valid species
    valid = get_valid_species()
    print(f"\n✅ Loaded Species:")
    for i, (name, bee_class) in enumerate(valid.items(), 1):
        try:
            status = "✓" if name in STANDARD_BEE_TYPES else "🧬"
            print(f"  {i:2d}. [{status}] {name}: {bee_class.__name__}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            status = "[S]" if name in STANDARD_BEE_TYPES else "[E]"
            print(f"  {i:2d}. [{status}] {name}: {bee_class.__name__}")

    # List pending species
    print(f"\n⏳ Pending Species:")
    for i, (name, bee_class) in enumerate(BEE_SPECIES.items(), 1):
        if bee_class is None:
            print(f"  {i:2d}. [..] {name}")
    print("=" * 60)


if __name__ == "__main__":
    print_species_stats()
