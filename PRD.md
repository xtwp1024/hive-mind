# PRD: Hive Swarm System 优化与可视化控制台 (Vision 2.5)

## 1. 项目背景与目标
**Hive** 是一个高性能的分布式蜂群任务系统，目前已达到 9.83/10 的代码质量评分。
为了提升系统的可观测性和 Ralph Loop 插件的自动化处理效率，本项目旨在：
- **核心优化**: 引入异步 SSE 任务队列，完善数据库版本化迁移。
- **UI 增强**: 构建一个现代化的 Web 控制台，支持实时监控、任务编排和蜂群状态可视化。

---

## 2. 核心功能需求

### 2.1 异步任务引擎 (Backend)
- **SSE (Server-Sent Events) 支持**: `dispatcher.py` 需要支持异步生成，允许 UI 订阅实时任务流。
- **任务队列解耦**: 引入后台工作进程，实现任务的非阻塞执行与状态回传。
- **DB 迁移机制**: 在 `gep_store.py` 基础上增加版本控制（Schema Versioning），确保数据库平滑升级。

### 2.2 现代化可视化控制台 (UI/UX)
- **实时监控面板 (Dashboard)**:
    - 动态展示 CPU/内存占用。
    - 活跃蜜蜂 (Active Bees) 的拓扑结构图。
    - 实时任务进度条与成功率统计。
- **蜂群指挥中心 (Command Center)**:
    - 支持手动启动/停止特定类型的 Bee。
    - 任务 DAG (有向无环图) 的可视化查看与调试。
- **知识库浏览器 (Knowledge Base)**:
    - `shared_brain.py` 存储内容的搜索与可视化展示。

---

## 3. 技术策略 (给 Ralph Loop 的指令)

### 3.1 架构设计
- **前端**: 使用 HTML5 + Vanilla JS (追求极致加载速度) 或轻量级 React。
- **通信**: 基于 Flask/FastAPI 扩展现有的 `web_dashboard.py`，增加 WebSocket 或 SSE 终结点。
- **API 规范**: 统一使用 `Authorization: Bearer {API_KEY}` 进行跨模块交互。

### 3.2 优化红线
- **性能**: UI 响应延迟必须 < 200ms。
- **并发**: 支持至少 50 个 Bee 的高频数据同步不阻塞。
- **安全**: UI 必须具备基础鉴权，敏感字段（如私钥/API Key）禁止在前端展示。

---

## 4. 任务清单 (Ralph Loop 待办)

### 第一阶段：基础设施增强 (Infrastructure)
- [ ] 在 `hive/core/dispatcher.py` 中实现 SSE 消息总线。
- [ ] 编写 `hive/evolution/migrations/` 初始 Schema 版本。
- [ ] 优化 `ClusterManager` 的 `CLUSTER_TIMEOUT` 评估逻辑。

### 第二阶段：UI 核心开发 (Frontend)
- [ ] 升级 `web_dashboard.py` 使用响应式网格布局。
- [ ] 集成 `Chart.js` 或 `D3.js` 用于蜂群拓扑展示。
- [ ] 实现任务历史记录的滚动查询。

### 第三阶段：集成与冒烟测试
- [ ] 编写 UI 自动化测试脚本，验证核心接口调用连通性。
- [ ] 执行压力测试，确保 UI 开启状态下蜂群 QPS 不下降超过 5%。

---

## 5. 验收标准
1.  **Ralph Loop** 能够通过 CLI 一键启动 UI 并看到实时数据流。
2.  所有 `Cluster` 超时和解散行为均能在 UI 上即时反馈。
3.  数据库升级后数据不丢失，版本号正确更新。
