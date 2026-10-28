#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AgentQL (智能体查询控制分发变体)
致敬了 Apache Hive 的 HiveQL 设计哲学。
允许使用者使用类似 SQL 的自然语义与正则包装系统下发群集指令，而不是写繁琐的 DNA 对象。
"""

import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("Hive.AgentQL")

class AgentQLParser:
    """极其原始和硬核的原型词法解析器"""
    
    # 支持语句: SELECT extract(information) FROM search(internet) WHERE topic='量子跃迁'
    QUERY_PATTERN = re.compile(
        r"^\s*SELECT\s+(?P<action>[a-zA-Z_]\w*(?:\(.*?\))?)\s+FROM\s+(?P<source>[a-zA-Z_]\w*(?:\(.*?\))?)\s+WHERE\s+(?P<condition>.*?)(?:\s+LIMIT\s+(?P<limit>\d+))?\s*$",
        re.IGNORECASE
    )

    @classmethod
    def parse_query(cls, query: str) -> Dict[str, Any]:
        """将类 SQL 的控制语句转译为 MapReduce 或者普通工蜂可以执行的底层架构 DNA 参数"""
        match = cls.QUERY_PATTERN.match(query)
        if not match:
            raise ValueError(f"语法错误 (Syntax Error) 无法解析 AgentQL: '{query}'")
            
        d = match.groupdict()
        action = d.get("action", "")
        source = d.get("source", "")
        condition = d.get("condition", "").strip()
        limit_str = d.get("limit")
        limit = int(limit_str) if limit_str else 1
        
        logger.info(f"📜 [AgentQL] 成功转译指令树! Action={action}, Source={source}, Condition={condition}, Limit={limit}")
        
        # 将其转化为可以被 MapReduceDispatcher (或者蜂后直接派发) 的配置格式
        # 取一个简单的隐喻规则：
        # 如果 source 带有 (internet)，则调用 SeekerBee。如果带有 (local)，调用 GathererBee
        target_species = "Standard"
        if "search" in source.lower() or "internet" in source.lower():
            target_species = "Seeker"
        elif "local" in source.lower() or "file" in source.lower():
            target_species = "Gatherer"
            
        return {
            "mr_base_task": f"按照条件进行内容汲取: {condition}",
            "mr_sub_items": [f"分页包_部分_{i+1}" for i in range(limit)], # 如果有 LIMIT，强行打碎成并发 Map 发给 Limit 个蜂计算
            "target_species": target_species,
            "reduce_action": action  # 将 SELECT 后面的操作当作 Reduce 动作（譬如 extract 归缩）
        }

    @classmethod
    def execute_query(cls, query: str, queen_instance) -> str:
        """解析并直接向蜂群下压任务！它是蜂后对用户的又一层代理薄膜"""
        try:
            plan = cls.parse_query(query)
            
            mr_dispatcher = queen_instance.mr_dispatcher
            job_id = mr_dispatcher.submit_map_job(
                base_task=plan["mr_base_task"],
                sub_items=plan["mr_sub_items"],
                target_species=plan["target_species"],
                reduce_action=plan["reduce_action"]
            )
            return f"✅ AgentQL 语法树执行就绪！已部署集团军作战单: {job_id}"
        except Exception as e:
            logger.error(f"❌ [AgentQL] 编译崩溃: {e}")
            return str(e)

