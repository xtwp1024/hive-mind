"""
CyberMinerMinerAntibodyBee - 混合变异蜂种
结合赛博化资产提取和数据库错误修复能力
"""

from pathlib import Path
import json
import csv
import sqlite3
from datetime import datetime
import hashlib

from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger

logger = get_logger(__name__)


class CyberMinerMinerAntibodyBee(BaseBee):
    """
    混合变异：赛博化：提取非代码资产 + 错误修复和数据库演进
    
    特殊能力：杂交能力：自动化增强 - 扫描JSON/CSV/权重文件 + 生成SQL补丁
    变异率：7.55%
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        self.reports_root = Path("reports")
        self.report_dir = self.reports_root / "cyberminerminerantibody"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录发现的资产和数据库问题
        self.discovered_assets = []
        self.database_issues = []
        self.sql_patches = []
    
    def specialized_task(self, target_path: Path):
        """
        执行混合任务：
        1. 扫描并提取JSON/CSV/权重文件资产
        2. 识别数据库问题并生成SQL补丁
        """
        logger.info(f"[{self.dna.target_name}] 开始执行混合任务：资产提取 + 数据库修复")
        
        # Update internal target path reference if needed, or just use the arg
        # self.dna.target_path might be the URL, target_path is the local dir
        self.local_target_path = target_path # Save for helper methods
        
        # 第一阶段：扫描和提取资产
        self._scan_and_extract_assets()
        
        # 第二阶段：分析数据库问题
        self._analyze_database_issues()
        
        # 第三阶段：生成综合报告
        self._generate_comprehensive_report()
        
        logger.info(f"[{self.dna.target_name}] 混合任务完成")
    
    def _scan_and_extract_assets(self):
        """扫描并提取非代码资产（JSON/CSV/权重文件）"""
        logger.info(f"[{self.dna.target_name}] 扫描项目资产文件...")
        
        target_path = self.local_target_path
        
        # 查找所有资产文件
        asset_extensions = ['.json', '.csv', '.yaml', '.yml', '.toml', '.weights', '.bin', '.pkl', '.pth']
        
        for ext in asset_extensions:
            pattern = f"*{ext}"
            for file_path in target_path.rglob(pattern):
                try:
                    asset_info = self._analyze_asset_file(file_path)
                    if asset_info:
                        self.discovered_assets.append(asset_info)
                        logger.debug(f"[{self.dna.target_name}] 发现资产: {file_path.relative_to(target_path)}")
                except Exception as e:
                    logger.warning(f"[{self.dna.target_name}] 分析资产文件失败 {file_path}: {e}")
        
        logger.info(f"[{self.dna.target_name}] 共发现 {len(self.discovered_assets)} 个资产文件")
    
    def _analyze_asset_file(self, file_path: Path) -> dict:
        """分析单个资产文件"""
        file_size = file_path.stat().st_size
        file_hash = self._calculate_file_hash(file_path)
        
        asset_info = {
            'path': str(file_path.relative_to(self.local_target_path)),
            'type': file_path.suffix.lower(),
            'size': file_size,
            'hash': file_hash,
            'metadata': {}
        }
        
        # 根据文件类型进行特定分析
        if file_path.suffix.lower() == '.json':
            asset_info['metadata'].update(self._analyze_json_file(file_path))
        elif file_path.suffix.lower() == '.csv':
            asset_info['metadata'].update(self._analyze_csv_file(file_path))
        
        return asset_info
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _analyze_json_file(self, file_path: Path) -> dict:
        """分析JSON文件结构"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = {
                'structure': type(data).__name__,
                'keys_count': len(data) if isinstance(data, dict) else 0,
                'items_count': len(data) if isinstance(data, (list, dict)) else 1
            }
            
            # 检查是否包含敏感配置
            if isinstance(data, dict):
                sensitive_keys = ['password', 'secret', 'api_key', 'token', 'key']
                found_sensitive = [k for k in data.keys() if any(s in k.lower() for s in sensitive_keys)]
                if found_sensitive:
                    metadata['sensitive_keys'] = found_sensitive
            
            return metadata
        except Exception as e:
            logger.warning(f"[{self.dna.target_name}] JSON解析失败 {file_path}: {e}")
            return {'error': str(e)}
    
    def _analyze_csv_file(self, file_path: Path) -> dict:
        """分析CSV文件结构"""
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            metadata = {
                'header': rows[0] if rows else [],
                'row_count': len(rows) - 1 if rows else 0,
                'column_count': len(rows[0]) if rows else 0
            }
            
            return metadata
        except Exception as e:
            logger.warning(f"[{self.dna.target_name}] CSV解析失败 {file_path}: {e}")
            return {'error': str(e)}
    
    def _analyze_database_issues(self):
        """分析数据库问题并生成SQL补丁"""
        logger.info(f"[{self.dna.target_name}] 分析数据库问题...")
        
        target_path = self.local_target_path
        
        # 查找数据库文件
        db_files = list(target_path.rglob('*.db')) + list(target_path.rglob('*.sqlite'))
        
        for db_file in db_files:
            try:
                issues = self._analyze_database_file(db_file)
                if issues:
                    self.database_issues.extend(issues)
                    logger.debug(f"[{self.dna.target_name}] 数据库问题: {db_file.relative_to(target_path)}")
            except Exception as e:
                logger.warning(f"[{self.dna.target_name}] 分析数据库失败 {db_file}: {e}")
        
        # 检查代码中的SQL语句
        self._analyze_sql_statements()
        
        logger.info(f"[{self.dna.target_name}] 共发现 {len(self.database_issues)} 个数据库问题")
    
    def _analyze_database_file(self, db_file: Path) -> list:
        """分析单个数据库文件"""
        issues = []
        
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                
                # 检查表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # 检查是否有索引
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()
                
                # 检查行数
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # 检查是否有主键
                has_primary_key = any(col[5] for col in columns)
                
                if not has_primary_key:
                    issues.append({
                        'database': str(db_file.relative_to(self.local_target_path)),
                        'table': table_name,
                        'type': 'missing_primary_key',
                        'severity': 'warning',
                        'description': f"表 {table_name} 缺少主键"
                    })
                    self.sql_patches.append({
                        'type': 'add_primary_key',
                        'database': str(db_file),
                        'table': table_name,
                        'sql': f"-- 建议为表 {table_name} 添加主键\nALTER TABLE {table_name} ADD COLUMN id INTEGER PRIMARY KEY;"
                    })
                
                if not indexes and row_count > 1000:
                    issues.append({
                        'database': str(db_file.relative_to(self.local_target_path)),
                        'table': table_name,
                        'type': 'missing_index',
                        'severity': 'info',
                        'description': f"表 {table_name} 有 {row_count} 行但缺少索引",
                        'suggestion': f"考虑为常用查询字段创建索引"
                    })
            
            conn.close()
        except Exception as e:
            logger.warning(f"[{self.dna.target_name}] 数据库分析失败: {e}")
        
        return issues
    
    def _analyze_sql_statements(self):
        """分析代码中的SQL语句，查找潜在问题"""
        target_path = self.local_target_path
        
        python_files = list(target_path.rglob('*.py'))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查SQL注入风险
                if '%s' in content or 'format(' in content:
                    if 'SELECT' in content or 'INSERT' in content or 'UPDATE' in content:
                        self.database_issues.append({
                            'database': str(py_file.relative_to(self.local_target_path)),
                            'table': 'code_file',
                            'type': 'sql_injection_risk',
                            'severity': 'critical',
                            'description': f"文件 {py_file.name} 可能存在SQL注入风险",
                            'suggestion': "使用参数化查询代替字符串格式化"
                        })
            except Exception as e:
                logger.debug(f"[{self.dna.target_name}] 分析SQL语句失败 {py_file}: {e}")
    
    def _generate_comprehensive_report(self):
        """生成综合报告"""
        report = {
            'bee_type': 'CyberMinerMinerAntibody',
            'target_name': self.dna.target_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'assets_discovered': len(self.discovered_assets),
                'database_issues': len(self.database_issues),
                'sql_patches': len(self.sql_patches)
            },
            'assets': self.discovered_assets,
            'database_issues': self.database_issues,
            'sql_patches': self.sql_patches,
            'recommendations': self._generate_recommendations()
        }
        
        # 保存报告
        report_file = self.report_dir / f"{self.dna.target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[{self.dna.target_name}] 综合报告已生成: {report_file}")
    
    def _generate_recommendations(self) -> list:
        """生成改进建议"""
        recommendations = []
        
        # 资产管理建议
        if len(self.discovered_assets) > 0:
            recommendations.append({
                'category': 'asset_management',
                'priority': 'medium',
                'action': '建立资产清单',
                'description': f"发现 {len(self.discovered_assets)} 个资产文件，建议建立集中化的资产管理系统"
            })
        
        # 数据库优化建议
        critical_issues = [i for i in self.database_issues if i.get('severity') == 'critical']
        if critical_issues:
            recommendations.append({
                'category': 'database_security',
                'priority': 'high',
                'action': '修复关键安全问题',
                'description': f"发现 {len(critical_issues)} 个关键数据库安全问题，需要立即处理"
            })
        
        # 性能优化建议
        index_issues = [i for i in self.database_issues if i.get('type') == 'missing_index']
        if index_issues:
            recommendations.append({
                'category': 'performance',
                'priority': 'medium',
                'action': '优化数据库索引',
                'description': f"发现 {len(index_issues)} 个表缺少索引，建议添加索引以提高查询性能"
            })
        
        return recommendations
