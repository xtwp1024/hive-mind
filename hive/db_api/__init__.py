"""
Hive DB-API v2.0 兼容门面包。
设计灵感来源：dropbox/PyHive。
将 Hive 面向 Agent 的智能系统包装为标准的关系型数据库连接驱动接口。
用户只需编写 SQL 语句，游标会自动把 SQL 翻译为大模型的提取需求并获取结果。
"""

from .connection import Connection


# PEP-249 规定的入口标准函数
def connect(*args, **kwargs):
    return Connection(*args, **kwargs)


# PEP-249 标准属性
apilevel = "2.0"
threadsafety = 2  # 线程安全的游标跨连接
paramstyle = "pyformat"
