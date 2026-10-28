# QueenBee 重构计划

**目标:** 将 QueenBee 拆分为单一职责的组件  
**当前状态:** 421 行，7 种职责混合  
**目标状态:** 5-6 个专注的组件

---

## 当前问题

QueenBee 负责:
1. **V1 任务部署** - `deploy_swarm()` - 同步进程
2. **V2 任务部署** - `deploy_swarm_v2()` - 异步调度
3. **Worker 进程管理** - 启动/停止/监控
4. **蜂蜜消费/结果处理** - `consume_honey()`
5. **DNA 繁殖/进化** - `_reproduce()`
6. **免疫响应** - `trigger_immune_response()`
7. **向量库消化** - `_digest_knowledge()`

---

## 重构方案

### 1. SwarmDeployer (任务分发)

```python
# hive/core/deployer.py
class SwarmDeployer:
    """专注任务分发"""
    
    def deploy(self, dna_list: list[BeeDNA]) -> None:
        """V1 同步部署"""
        
    async def deploy_async(self, dna_list: list[BeeDNA]) -> list[dict]:
        """V2 异步部署"""
```

**职责:**
- 创建任务节点
- 管理部署间隔
- 协调调度器

---

### 2. WorkerManager (进程管理)

```python
# hive/core/worker_manager.py
class WorkerManager:
    """专注 Worker 生命周期"""
    
    def spawn_worker(self, dna: BeeDNA) -> Process:
        """启动 Worker"""
        
    def cleanup_dead_workers(self) -> None:
        """清理死亡进程"""
        
    def shutdown(self, timeout: float) -> None:
        """优雅关闭"""
```

**职责:**
- 进程启动/终止
- 状态跟踪
- 资源限制

---

### 3. EvolutionEngine (DNA 进化)

```python
# hive/core/evolution_engine.py
class EvolutionEngine:
    """专注 DNA 繁殖进化"""
    
    def reproduce(self, parent_dna: dict) -> BeeDNA | None:
        """分裂繁殖"""
        
    def calculate_generation(self, parent_id: str) -> int:
        """计算代数"""
```

**职责:**
- DNA 复制
- 突变逻辑
- 资源检查

---

### 4. HoneyProcessor (结果处理)

```python
# hive/core/honey_processor.py
class HoneyProcessor:
    """专注结果处理"""
    
    def process(self, honey: dict) -> dict:
        """处理单个结果"""
        
    def digest_knowledge(self, honey: dict) -> None:
        """消化知识到向量库"""
        
    def trigger_immune_response(self, bad_honey: dict) -> None:
        """触发免疫响应"""
```

**职责:**
- 蜂蜜消费
- 知识消化
- 免疫响应

---

### 5. QueenFacade (统一入口)

```python
# hive/core/queen.py (重构后)
class QueenBee:
    """蜂后 - 统一门面"""
    
    def __init__(self, ...):
        self.deployer = SwarmDeployer()
        self.worker_manager = WorkerManager()
        self.evolution = EvolutionEngine()
        self.processor = HoneyProcessor()
    
    def deploy_swarm(self, dna_list):
        self.deployer.deploy(dna_list)
    
    def consume_honey(self):
        while honey := self._get_honey():
            result = self.processor.process(honey)
            if result.offspring:
                self.evolution.reproduce(result.dna)
```

---

## 实施步骤

### Phase 1: 创建新组件 (不修改现有代码)

1. 创建 `hive/core/deployer.py`
2. 创建 `hive/core/worker_manager.py`
3. 创建 `hive/core/evolution_engine.py`
4. 创建 `hive/core/honey_processor.py`

### Phase 2: 迁移职责

5. 迁移 V1/V2 部署逻辑到 Deployer
6. 迁移进程管理到 WorkerManager
7. 迁移繁殖逻辑到 EvolutionEngine
8. 迁移结果处理到 HoneyProcessor

### Phase 3: 重构 QueenBee

9. 将 QueenBee 改为门面模式
10. 委托各组件处理
11. 保留接口兼容性

### Phase 4: 测试

12. 单元测试每个新组件
13. 集成测试 QueenBee
14. 性能测试

---

## 文件变更清单

| 操作 | 文件 |
|------|------|
| 新建 | `hive/core/deployer.py` |
| 新建 | `hive/core/worker_manager.py` |
| 新建 | `hive/core/evolution_engine.py` |
| 新建 | `hive/core/honey_processor.py` |
| 修改 | `hive/core/queen.py` (简化为门面) |

---

## 兼容性保证

- 保持 `QueenInterface` 不变
- 保持现有 API 签名
- 添加类型注解
- 添加单元测试

---

## 预期收益

| 指标 | 重构前 | 重构后 |
|------|--------|--------|
| QueenBee 行数 | 421 | ~100 |
| 单一职责 | 7 混合 | 1 专注 |
| 可测试性 | 困难 | 容易 |
| 并行开发 | 阻塞 | 可独立 |
