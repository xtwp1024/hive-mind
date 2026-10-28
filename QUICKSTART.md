# Hive Mind Swarm System - 快速入门示例

## 基本使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行系统

```bash
# 开发模式
python main.py

# 或使用 Makefile
make run
```

### 3. 运行测试

```bash
# 所有测试
pytest hive/tests/ -v

# 单元测试
make test-unit

# 集成测试
make test-integration
```

## 组件使用示例

### APIGateway

```python
from hive.components.api_gateway import APIGateway, APIRoute

gateway = APIGateway()

route = APIRoute(
    path="/api/v1/custom",
    methods=["GET"],
    handler="custom_handler",
    auth_required=True
)

gateway.register_route(route)
```

### Scheduler

```python
from hive.components.scheduler import Scheduler, TaskPriority

scheduler = Scheduler()

async def my_handler(msg):
    return f"Done: {msg}"

scheduler.register_handler("my_task", my_handler)

task_id = scheduler.schedule_task(
    name="example_task",
    handler="my_task",
    args=("hello",),
    priority=TaskPriority.HIGH
)
```

### RateLimiter

```python
from hive.components.rate_limiter import RateLimiter, LimiterType, LimitRule

limiter = RateLimiter()

rule = LimitRule(
    name="api_limit",
    limiter_type=LimiterType.TOKEN_BUCKET,
    capacity=100,
    refill_rate=10
)

limiter.create_limiter("api", rule)

allowed = await limiter.acquire("api")
```

### Prometheus指标

```python
from hive.utils.metrics import get_metrics, export_prometheus

metrics = get_metrics()
metrics.increment("requests_total")

# 导出Prometheus格式
prometheus_output = export_prometheus()
```

## Docker使用

```bash
# 构建镜像
docker build -t hive-mind:latest .

# 运行容器
docker run -it hive-mind:latest

# 或使用 docker-compose
docker-compose up -d
```

## 开发命令

| 命令 | 描述 |
|------|------|
| `make install` | 安装依赖 |
| `make test` | 运行测试 |
| `make lint` | 代码检查 |
| `make format` | 代码格式化 |
| `make run` | 运行系统 |
| `make docker-build` | 构建Docker |

---

更多详细信息请参阅 [README.md](README.md)
