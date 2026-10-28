# Hive Mind Swarm System

<div align="center">

**蜂巢思维群智系统 v1.0.0**

一个基于蜂群本质的多AI协作智能系统，采用生物启发式架构实现分布式任务处理和自动进化

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [开发指南](#开发指南)
- [测试](#测试)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

Hive Mind Swarm System 是一个基于生物启发式架构的多AI协作系统。系统模仿蜜蜂社会的组织方式，通过不同类型的"工蜂"（Worker Bees）协同工作来完成复杂的任务。系统具备自动进化能力，可以通过遗传算法不断优化自身性能。

### 设计理念

- **模块化架构**: 每个工蜂类型都是独立模块，易于扩展和维护
- **生物启发**: 模仿自然界的蜂群组织，实现高效协作
- **自动进化**: 基于遗传算法的DNA系统，实现自我优化
- **安全第一**: 内置影子沙箱和免疫响应机制
- **可观测性**: 完善的监控和指标系统

---

## 核心特性

### 智能协作

- 多种专业化工蜂类型（30+ 种）
- 动态任务分配和调度
- 遗传算法驱动的任务优化

### 安全机制

- 影子沙箱隔离执行环境
- 免疫响应系统防御异常行为
- 自动故障检测和恢复

### 监控系统

- 实时系统资源监控
- 性能指标收集和分析
- 健康检查和告警

### 可扩展性

- 支持自定义工蜂类型
- 灵活的配置系统
- 插件式架构设计

---

## 系统架构

```
Hive Mind Swarm System
│
├── 🐝 QueenBee (蜂后) - 核心协调器
│   ├── 任务分发
│   ├── 结果收集
│   └── 进化管理
│
├── 👷 Worker Bees (工蜂) - 任务执行器
│   ├── Miner Bee - 数据挖掘
│   ├── Architect Bee - 系统架构
│   ├── Librarian Bee - 知识管理
│   ├── Code Generator Bee - 代码生成
│   ├── Darwin Bee - 进化优化
│   └── 30+ 其他专业化工蜂
│
├── 🧬 DNA System - 遗传系统
│   ├── 基因编码
│   ├── 变异机制
│   └── 适应度评估
│
├── 🛡️ Security - 安全系统
│   ├── Shadow Sandbox
│   ├── Immune Response
│   └── Anomaly Detection
│
└── 📊 Monitoring - 监控系统
    ├── System Monitor
    ├── Metrics Collector
    └── Health Checker
```

### 核心组件

| 组件 | 描述 | 文件位置 |
|------|------|----------|
| QueenBee | 核心协调器，负责任务分发和结果收集 | `hive/core/queen.py` |
| BeeDNA | 遗传系统，实现任务编码和进化 | `hive/core/dna.py` |
| BaseBee | 所有工蜂的基类 | `hive/bees/base_bee.py` |
| Settings | 配置管理系统 | `hive/config/settings.py` |
| Monitor | 系统资源监控 | `hive/utils/monitor.py` |
| Metrics | 性能指标收集 | `hive/utils/metrics.py` |

---

## 快速开始

### 环境要求

- Python 3.8 或更高版本
- Windows / Linux / macOS
- 至少 4GB RAM
- 推荐 4 核心以上 CPU

### 安装

1. **克隆仓库**

```bash
git clone https://github.com/antigravity/hive.git
cd hive
```

2. **创建虚拟环境**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

或者使用开发模式安装：

```bash
pip install -e ".[dev]"
```

4. **配置环境变量**

复制 `.env` 文件并根据需要修改：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# AI Provider
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434

# System Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_WORKERS=4
MUTATION_RATE=0.1
```

### 运行

```bash
python main.py
```

或使用安装的命令：

```bash
hive
```

### 输出示例

```
============================================================
Initializing Hive Mind Swarm System v2.0
============================================================
Environment: development
Max workers: 4
Mutation rate: 0.1
CPU: 8 cores
Memory: 16 GB

============================================================
Starting Hive Mind Swarm
============================================================

============================================================
🔄 Cycle 1/10
============================================================
🔥 Step 1: Warming up - Ensuring all bee species are present...
✅ Warmup complete: 30 species activated
🚀 Step 2: Deploying swarm (V2 Dispatcher Mode)...
✅ Batch complete. Processed 30 tasks.
🧬 Step 3: Evolution & Digestion...
✅ System healthy
💤 Resting 2 seconds before next cycle...
```

---

## 配置说明

### 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `ENVIRONMENT` | 运行环境 (development/production) | `development` |
| `LOG_LEVEL` | 日志级别 (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `MAX_WORKERS` | 最大并发工蜂数 | `4` |
| `MUTATION_RATE` | 基因变异率 | `0.1` |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `LOG_DIR` | 日志目录 | `logs` |
| `DATA_DIR` | 数据目录 | `Data` |

### 配置文件

系统支持通过 `pyproject.toml` 或自定义配置文件进行详细配置。

```toml
[tool.hive]
max_workers = 8
mutation_rate = 0.15
log_level = "INFO"
enable_sandbox = true
```

---

## 🧬 自我进化系统

Hive 内置了基于 GEP (Genome Evolution Protocol) 的自我进化引擎，能够自动分析运行历史、识别问题并进行自我改进。

### 快速开始

```bash
# 执行单次进化周期
python -m hive.evolution

# 人工审核模式 (推荐首次使用)
python -m hive.evolution --review

# 持续运行模式
python -m hive.evolution --loop

# 查看进化状态
python -m hive.evolution status

# 列出已存储的基因
python -m hive.evolution genes
```

### 进化策略

通过环境变量 `EVOLVE_STRATEGY` 配置进化策略：

| 策略 | 描述 |
|------|------|
| `balanced` | 平衡模式：修复与创新并重 (默认) |
| `innovate` | 创新模式：优先尝试新方案 |
| `harden` | 加固模式：优先稳定性和安全性 |
| `repair-only` | 仅修复：只处理已知错误 |
| `early-stabilize` | 早期稳定：快速收敛到稳定状态 |
| `steady-state` | 稳态模式：维持现状，最小改动 |
| `auto` | 自动模式：根据系统状态动态选择 |

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `EVOLVE_STRATEGY` | `balanced` | 进化策略 |
| `EVOLVE_LOAD_MAX` | `2.0` | 最大负载阈值，超过时退避 |
| `EVOLVE_REVIEW` | `false` | 是否启用人工审核 |

### GEP 协议

进化系统使用 GEP 协议存储进化资产：

```
hive/evolution/gep/
├── genes.json      # 可复用的基因片段 (修复补丁、优化方案)
├── capsules.json   # 成功的进化方案组合
└── events.jsonl    # 进化事件日志 (append-only)
```

### 与 DNA 系统集成

进化引擎与 Hive 的 DNA 系统深度集成：

- 从 `morgue/` 目录扫描崩溃报告
- 分析错误日志识别改进点
- 将进化成果应用到 DNA 池
- 适应度反馈优化基因质量

```python
from hive.evolution import Evolver
from hive.evolution.dna_bridge import DNABridge

# 运行进化
evolver = Evolver()
result = evolver.run_cycle()

# 将基因应用到 DNA 系统
bridge = DNABridge()
bridge.apply_gene_to_dna(gene, target_species="bee")
```

### 高级功能 (v1.1.0+)

#### 带监控的进化

使用实时系统指标动态调整进化策略：

```python
from hive.evolution import Evolver

evolver = Evolver()
# 根据 CPU/内存/错误率自动调整策略
evolver.run_with_monitoring(interval=300, max_cycles=10)
```

#### 带测试验证的进化

每次进化后自动运行测试，检测退化则告警：

```python
evolver.run_with_tests(interval=300, max_cycles=10)
```

#### EvoMap 网络集成

连接 EvoMap 进化网络，获取热门知识并发布本地成果：

```python
# 与 EvoMap 同步
result = evolver.sync_with_evomap()
print(f"导入: {result['imported']} 个胶囊")
print(f"发布: {result['published']} 个基因")

# 完整进化循环 (含 EvoMap 同步)
evolver.run_full_cycle()
```

环境变量配置：

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `EVOMAP_ENABLED` | `true` | 启用 EvoMap 连接 |
| `EVOMAP_NODE_ID` | - | 节点 ID (自动生成) |
| `EVOMAP_AUTO_PUBLISH` | `false` | 自动发布高价值基因 |
| `EVOMAP_AUTO_FETCH` | `true` | 自动获取热门胶囊 |
| `EVOMAP_MIN_GDI` | `40.0` | 最低 GDI 阈值 |

#### 模块结构

```
hive/evolution/
├── __init__.py        # 模块入口
├── __main__.py        # CLI 入口
├── cli.py             # 命令行接口
├── evolver.py         # 核心进化引擎
├── strategies.py      # 进化策略配置
├── dna_bridge.py      # DNA 系统集成
├── evomap_bridge.py   # EvoMap 网络连接
├── monitor_bridge.py  # 监控系统集成
├── test_runner.py     # 自动测试验证
└── gep/               # GEP 协议数据
    ├── genes.json
    ├── capsules.json
    └── events.jsonl
```

---

## 开发指南

### 项目结构

```
Hive_Standalone/
├── main.py                 # 主入口
├── hive_mind.py           # 快速启动脚本
├── requirements.txt       # 依赖列表
├── pyproject.toml        # 项目配置
├── .env                  # 环境变量
├── .gitignore           # Git 忽略规则
├── hive/                # 核心模块
│   ├── bees/           # 工蜂类型
│   │   ├── base_bee.py
│   │   ├── miner_bee.py
│   │   ├── architect_bee.py
│   │   └── ...
│   ├── core/           # 核心组件
│   │   ├── queen.py
│   │   ├── dna.py
│   │   └── warmup.py
│   ├── ministers/       # 部长级组件
│   │   └── engineer.py
│   ├── config/         # 配置
│   │   └── settings.py
│   └── utils/          # 工具
│       ├── logger.py
│       ├── monitor.py
│       └── metrics.py
├── tests/              # 测试文件
└── Data/              # 数据目录
```

### 创建自定义工蜂

1. **继承 BaseBee 类**

```python
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA

class CustomBee(BaseBee):
    """自定义工蜂"""

    species_name = "custom"
    description = "我的自定义工蜂"

    def execute(self, dna: BeeDNA) -> dict:
        """执行任务"""
        try:
            # 实现你的逻辑
            result = self.do_work(dna)
            return {
                "success": True,
                "data": result,
                "metadata": {}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": {}
            }

    def do_work(self, dna: BeeDNA):
        """实际的工作逻辑"""
        # 实现你的具体功能
        pass
```

2. **注册工蜂类型**

在 `hive/bees/__init__.py` 中导入你的工蜂：

```python
from .custom_bee import CustomBee

__all__ = ['CustomBee', ...]
```

3. **测试工蜂**

```python
import pytest
from hive.bees.custom_bee import CustomBee

def test_custom_bee():
    bee = CustomBee()
    assert bee.species_name == "custom"
    # 添加更多测试
```

### 代码风格

项目使用以下工具进行代码质量控制：

- **Black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

运行代码检查：

```bash
# 格式化代码
black hive/
isort hive/

# 代码检查
flake8 hive/
mypy hive/
```

---

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_queen.py

# 运行带覆盖率的测试
pytest --cov=hive --cov-report=html

# 运行快速测试（跳过慢速测试）
pytest -m "not slow"

# 运行集成测试
pytest -m integration
```

### 测试覆盖率

目标：80% 以上覆盖率

查看覆盖率报告：

```bash
pytest --cov=hive --cov-report=html
open htmlcov/index.html
```

### 编写测试

```python
import pytest
from hive.core.queen import QueenBee
from hive.core.dna import BeeDNA

class TestQueenBee:
    """测试 QueenBee 类"""

    @pytest.fixture
    def queen(self):
        """创建 QueenBee 实例"""
        return QueenBee(result_queue=None)

    def test_deploy_swarm(self, queen):
        """测试部署工蜂"""
        dna_list = [BeeDNA(species="miner")]
        results = queen.deploy_swarm_v2(dna_list)
        assert isinstance(results, list)
```

---

## 常见问题

### Q: 如何增加工蜂数量？

A: 修改 `.env` 文件中的 `MAX_WORKERS` 变量，或通过代码设置：

```python
from hive.config.settings import settings
settings.MAX_WORKERS = 8
```

### Q: 系统支持哪些 AI 提供商？

A: 目前支持：
- OpenAI (GPT-4, GPT-3.5)
- Ollama (本地开源模型)

### Q: 如何添加新的 AI 提供商？

A: 在 `hive/config/settings.py` 中添加新的配置，并创建相应的适配器。

### Q: 系统安全吗？

A: 系统内置多重安全机制：
- 影子沙箱隔离执行环境
- 免疫响应系统检测异常行为
- 资源限制防止过度消耗
- 输入验证防止注入攻击

### Q: 如何调试工蜂执行问题？

A: 启用 DEBUG 日志级别：

```env
LOG_LEVEL=DEBUG
```

查看详细日志：

```bash
tail -f logs/hive.log
```

---

## 贡献指南

我们欢迎所有形式的贡献！

### 贡献方式

1. **报告 Bug**: 在 Issues 中提交问题
2. **提出建议**: 分享你的想法和改进建议
3. **提交代码**: 创建 Pull Request
4. **完善文档**: 帮助改进文档

### 提交代码流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 提交信息规范

```
类型(范围): 简短描述

详细描述（可选）

关联的 Issue（可选）
```

类型：
- `feat`: 新功能
- `fix`: 修复 Bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：

```
feat(bees): add quantum computing bee

Implement a new quantum computing bee that can perform
quantum algorithm tasks using Qiskit.

Closes #123
```

---

## 性能优化建议

1. **调整工蜂数量**: 根据系统资源调整 `MAX_WORKERS`
2. **使用本地模型**: 使用 Ollama 可以减少 API 调用延迟
3. **启用缓存**: 配置适当的缓存策略
4. **监控资源**: 定期检查系统资源使用情况

---

## 路线图

### v2.1 (计划中)
- [ ] 增加更多工蜂类型
- [ ] 支持分布式部署
- [ ] Web UI 管理界面
- [ ] 更多 AI 提供商支持

### v2.2 (计划中)
- [ ] 自动负载均衡
- [ ] 高级调度策略
- [ ] 持久化存储
- [ ] API 服务

### v3.0 (未来)
- [ ] 完全分布式架构
- [ ] 跨平台支持
- [ ] 机器学习优化
- [ ] 自动化部署

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 致谢

- 感谢所有贡献者的支持
- 感谢开源社区提供的优秀工具和库
- 灵感来源于自然界蜜蜂的协作智慧

---

## 联系方式

- **主页**: https://github.com/antigravity/hive
- **文档**: https://hive.readthedocs.io
- **问题反馈**: https://github.com/antigravity/hive/issues
- **邮箱**: jarvis@hive.ai

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！**

Made with ❤️ by the Hive Mind Team

</div>
