#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DB-API 原生测试脚本。
展示如何像连接 MySQL 一样，通过 PyHive/DB-API 的哲学连接我们庞大的 AI 集群系统！
"""

import logging
# 配置一个基本的控制台日志好让我们看到大军出动的踪迹
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 注意这里！不再需要 import QueenBee，不需要理解 BeeDNA！就用最原生的 Database API 驱动！
from hive.db_api import connect

def run_db_demo():
    print("\n======= LLM-as-a-DB 工业级兼容接口展示 =======")
    
    # 建立一条到 AI 的隐形连线
    conn = connect(host="hive-cluster", database="intelligence")
    cursor = conn.cursor()
    
    # 向系统发送一句 SQL
    sql = "SELECT name, sentiment, intention FROM minds WHERE task='请分析并提取出这条评论中的联系人、情绪和真实图谋'"
    print(f"\n[Client] Sending Query: {sql}")
    
    cursor.execute(sql)
    
    rows = cursor.fetchall()
    
    print("\n[Client] Received Output Table:")
    for row in rows:
        print(f"Row: {row}")
        
    cursor.close()
    conn.close()
    print("=============================================\n")

if __name__ == "__main__":
    import traceback
    try:
        run_db_demo()
    except Exception as e:
        with open("error_dump.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise
