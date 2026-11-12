import logging

from .cursor import Cursor

logger = logging.getLogger("Hive.DB_API.Connection")


class Connection:
    """
    符合 PEP 249 标准的 Connection 对象。
    本质上，这个连接对象维护了对 Hive 系统单例（或服务池）的连通性。
    """

    def __init__(self, host="localhost", port=10000, database="default", **kwargs):
        # 实际上的 Hive-Agent 是本地跑的单体，这里只是模仿远程连接的语义
        self.host = host
        self.port = port
        self.database = database
        self._closed = False

        logger.info(f"🔗 [LLM-DB] 成功贯通至智能体集群 '{database}' 无图计算节点。")

    def close(self):
        self._closed = True
        logger.info("🔌 [LLM-DB] 连接管道已拆除。")

    def commit(self):
        # 暂时为空：蜂巢操作不具备事务的严格一致性概念
        pass

    def rollback(self):
        # 暂时不支持回滚大模型记忆
        pass

    def cursor(self):
        if self._closed:
            raise Exception("Cannot operate on a closed Connection")
        return Cursor(self)
