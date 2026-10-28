# -*- coding: utf-8 -*-
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class WardenBee(BaseBee):
    """
    WardenBee (看守蜂)
    
    安全审计与人类交互专员。当系统中其他工蜂遇到高风险操作，或主动申请 Human-in-the-loop 时，
    将由 WardenBee 接管并在控制台阻止进程，等待人类许可。
    """
    
    species_name = "warden"
    description = "处理高危权限审查与人类介入交互 (Human-in-the-loop)"
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        
    def execute_task(self, task) -> dict:
        """这个方法是旧架构的执行方式（但目前在devour里运行统一逻辑），我们重写特殊的交互逻辑"""
        return self._handle_human_request(task)

    def specialized_task(self, target_path) -> dict:
        """重写专属任务"""
        return self._handle_human_request(self.dna.metadata)

    def _handle_human_request(self, context_metadata: dict) -> dict:
        """处理人类交互请求的核心逻辑"""
        risk_level = context_metadata.get("risk_level", "UNKNOWN")
        reason = context_metadata.get("reason", "未提供具体原因")
        source_bee = context_metadata.get("source_bee", "UNKNOWN")
        
        # 终端阻塞提醒
        print("\n" + "="*60)
        print(f"🛑 [HUMAN-IN-THE-LOOP REQUIRED] 🛑")
        print(f"⚠️ 来源工蜂: {source_bee}")
        print(f"⚠️ 风险等级: {risk_level}")
        print(f"⚠️ 请求原因: {reason}")
        print("="*60)
        
        # 挂起进程等待输入（由于 multiprocessing 的限制，如果部署在后台，这里可能会获取不到 stdin）
        # 但如果是 TUI 模式或前台进程，这里可以正常阻塞
        sys.stdout.flush()
        try:
            # 这是一个极度简化的控制台输入。如果在实际异步/Web UI 中，应替换为等待 WebSocket / Queue 信号
            decision = input("👉 请审核该操作 [A: 允许(Approve) / R: 拒绝(Reject) / S: 挂起(Suspend)]: ").strip().upper()
            
            if decision == 'A':
                logger.warning(f"👑 人类已授权通过: {reason}")
                return {"status": "APPROVED", "decision": "Approve", "human_note": ""}
            elif decision == 'R':
                logger.error(f"🚫 人类已拒绝该操作: {reason}")
                return {"status": "REJECTED", "decision": "Reject", "human_note": "Risk denied by human"}
            else:
                logger.warning(f"⏸️ 操作被挂起: {reason}")
                return {"status": "SUSPENDED", "decision": "Suspend"}
        except Exception as e:
            logger.error(f"交互获取失败，默认拒绝: {e}")
            return {"status": "REJECTED", "decision": "Error", "error": str(e)}
