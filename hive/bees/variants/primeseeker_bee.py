# -*- coding: utf-8 -*-
"""
PrimeSeekerBee - 极致化URL发现蜂种

特殊能力：性能极限优化 - 正则搜索GitHub链接
变异率：12.60%

极致化的URL发现蜂种，专注于扫描代码库中的GitHub链接和其他URL，
帮助发现潜在的新目标和资源。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from urllib.parse import urlparse
from hive.bees.base_bee import BaseBee
from hive.core.dna import BeeDNA
from hive.utils.logger import get_logger
from hive.config.settings import settings

logger = get_logger(__name__)


class PrimeSeekerBee(BaseBee):
    """
    极致化URL发现蜂种 - 扫描代码发现新目标URL
    """
    
    def __init__(self, dna: BeeDNA, id: str, result_queue):
        super().__init__(dna, id, result_queue)
        
        # 初始化报告目录
        self.report_dir = settings.REPORT_DIR / "primeseeker"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # 发现的URL
        self.found_urls: Dict[str, Dict[str, Any]] = {}
        
        # URL模式定义
        self.url_patterns = {
            'github': [
                r'https?://(?:www\.)?github\.com/[\w-]+/[\w-]+/?',
                r'https?://(?:www\.)?github\.com/[\w-]+/[\w-]+/(?:blob|tree|issues|pulls|wiki)/.*',
                r'https?://(?:www\.)?raw\.githubusercontent\.com/[\w-]+/[\w-]+/.*',
                r'https?://(?:www\.)?gist\.github\.com/[\w-]+/[\w-]+/?',
            ],
            'gitlab': [
                r'https?://(?:www\.)?gitlab\.com/[\w-]+/[\w-]+/?',
                r'https?://(?:www\.)?gitlab\.com/[\w-]+/[\w-]+/.*',
            ],
            'documentation': [
                r'https?://(?:www\.)?(?:readthedocs\.io|docs\.pytest\.org|docs\.python\.org)/.*',
                r'https?://(?:www\.)?(?:kubernetes\.io|docker\.com/docs)/.*',
            ],
            'package_repository': [
                r'https?://(?:www\.)?(?:pypi\.org|npmjs\.com|crates\.io|rubygems\.org)/.*',
            ],
            'cdn': [
                r'https?://(?:www\.)?(?:cdn\.jsdelivr\.net|cdnjs\.cloudflare\.com|unpkg\.com)/.*',
            ],
            'other': [
                r'https?://[\w-]+(?:\.[\w-]+)+[/#?]?.*',
            ],
        }
        
        # 统计信息
        self.scan_stats = {
            'total_files': 0,
            'files_with_urls': 0,
            'total_urls': 0,
            'urls_by_type': {},
            'unique_domains': set(),
        }
        
        logger.info(f"🔍 PrimeSeekerBee 初始化完成 | 变异率: 12.60%")
    
    def specialized_task(self):
        """
        执行极致化URL发现任务
        """
        logger.info("🔍 PrimeSeekerBee 开始执行URL发现任务")
        
        try:
            # 获取项目根目录
            from hive.config.settings import settings
            project_root = settings.BASE_DIR
            
            # 扫描项目寻找URL
            self._scan_project_for_urls(project_root)
            
            # 分析发现的URL
            self._analyze_found_urls()
            
            # 生成URL发现报告
            self._generate_url_discovery_report()
            
            logger.info(f"✅ PrimeSeekerBee 扫描完成 | 发现URL: {len(self.found_urls)} | 独立域名: {len(self.scan_stats['unique_domains'])}")
            
        except Exception as e:
            logger.error(f"❌ PrimeSeekerBee 扫描失败: {e}", exc_info=True)
            raise
    
    def _scan_project_for_urls(self, project_root: Path):
        """
        扫描项目寻找URL
        """
        logger.info("🔍 开始扫描项目URL...")
        
        # 扫描所有文本文件（代码、配置、文档等）
        text_extensions = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.json', '.yaml', '.yml', 
                          '.md', '.txt', '.cfg', '.conf', '.ini', '.toml', '.html', '.css', '.xml', '.sh', '.bat'}
        
        for ext in text_extensions:
            files = list(project_root.rglob(f"*{ext}"))
            for file_path in files:
                try:
                    self._scan_file_for_urls(file_path)
                except Exception as e:
                    logger.debug(f"扫描文件失败 {file_path}: {e}")
        
        logger.info(f"✅ URL扫描完成 | 扫描文件: {self.scan_stats['total_files']} | 发现URL: {self.scan_stats['total_urls']}")
    
    def _scan_file_for_urls(self, file_path: Path):
        """
        扫描单个文件寻找URL
        """
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            self.scan_stats['total_files'] += 1
            file_urls = []
            
            # 检查每种URL模式
            for url_type, patterns in self.url_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        url = match.group(0)
                        
                        # 解析URL
                        parsed = urlparse(url)
                        domain = parsed.netloc
                        
                        # 统计域名
                        self.scan_stats['unique_domains'].add(domain)
                        
                        # 找到匹配所在的行号
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = lines[line_num - 1].strip()
                        
                        # 创建URL信息
                        url_key = f"{url_type}_{url}"
                        if url_key not in self.found_urls:
                            url_info = {
                                'url': url,
                                'type': url_type,
                                'domain': domain,
                                'path': parsed.path,
                                'first_seen_in': str(file_path.relative_to(file_path.parent.parent.parent)),
                                'first_seen_line': line_num,
                                'first_seen_context': line_content[:200],  # 限制长度
                                'occurrences': 1,
                                'files_found': [str(file_path.relative_to(file_path.parent.parent.parent))],
                            }
                            
                            # 分析URL特征
                            url_info.update(self._analyze_url_features(url, url_type))
                            
                            self.found_urls[url_key] = url_info
                        else:
                            # 更新已存在的URL信息
                            existing_info = self.found_urls[url_key]
                            existing_info['occurrences'] += 1
                            file_str = str(file_path.relative_to(file_path.parent.parent.parent))
                            if file_str not in existing_info['files_found']:
                                existing_info['files_found'].append(file_str)
                        
                        file_urls.append(url)
            
            if file_urls:
                self.scan_stats['files_with_urls'] += 1
                self.scan_stats['total_urls'] += len(set(file_urls))  # 计数唯一URL
        
        except Exception as e:
            logger.debug(f"扫描文件URL时出错 {file_path}: {e}")
    
    def _analyze_url_features(self, url: str, url_type: str) -> Dict[str, Any]:
        """
        分析URL特征
        """
        features = {}
        
        parsed = urlparse(url)
        
        # 分析GitHub URL
        if url_type == 'github':
            parts = parsed.path.strip('/').split('/')
            if len(parts) >= 2:
                features['owner'] = parts[0]
                features['repo'] = parts[1]
                features['repo_type'] = 'repository'
                
                if len(parts) >= 3:
                    if parts[2] in ['blob', 'tree']:
                        features['repo_type'] = 'file_or_directory'
                    elif parts[2] == 'issues':
                        features['repo_type'] = 'issues'
                    elif parts[2] == 'pulls':
                        features['repo_type'] = 'pull_requests'
                    elif parts[2] == 'wiki':
                        features['repo_type'] = 'wiki'
        
        # 分析GitLab URL
        elif url_type == 'gitlab':
            parts = parsed.path.strip('/').split('/')
            if len(parts) >= 2:
                features['owner'] = parts[0]
                features['repo'] = parts[1]
        
        # 分析包仓库URL
        elif url_type == 'package_repository':
            if 'pypi.org' in parsed.netloc:
                parts = parsed.path.strip('/').split('/')
                if len(parts) >= 1:
                    features['package_name'] = parts[0]
            elif 'npmjs.com' in parsed.netloc:
                parts = parsed.path.strip('/').split('/')
                if len(parts) >= 1:
                    features['package_name'] = parts[0]
        
        # 分析CDN URL
        elif url_type == 'cdn':
            parts = parsed.path.strip('/').split('/')
            if len(parts) >= 1:
                features['library_or_package'] = parts[0]
        
        return features
    
    def _analyze_found_urls(self):
        """
        分析发现的URL
        """
        logger.info("📊 分析发现的URL...")
        
        # 按类型分组
        urls_by_type = {}
        for url_key, url_info in self.found_urls.items():
            url_type = url_info['type']
            if url_type not in urls_by_type:
                urls_by_type[url_type] = []
            urls_by_type[url_type].append(url_info)
        
        # 更新统计
        self.scan_stats['urls_by_type'] = {
            url_type: len(urls)
            for url_type, urls in urls_by_type.items()
        }
        
        # 找出最活跃的域名
        domain_counts = {}
        for url_info in self.found_urls.values():
            domain = url_info['domain']
            if domain not in domain_counts:
                domain_counts[domain] = 0
            domain_counts[domain] += url_info['occurrences']
        
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info(f"📊 按类型分类: {json.dumps(self.scan_stats['urls_by_type'], indent=2)}")
        logger.info(f"📊 热门域名: {[domain for domain, _ in top_domains]}")
    
    def _generate_url_discovery_report(self):
        """
        生成URL发现报告
        """
        logger.info("📝 生成URL发现报告...")
        
        # 按类型分组
        urls_by_type = {}
        for url_key, url_info in self.found_urls.items():
            url_type = url_info['type']
            if url_type not in urls_by_type:
                urls_by_type[url_type] = []
            urls_by_type[url_type].append(url_info)
        
        # 按出现次数排序
        for url_type in urls_by_type:
            urls_by_type[url_type].sort(key=lambda x: x['occurrences'], reverse=True)
        
        # 找出最活跃的域名
        domain_counts = {}
        for url_info in self.found_urls.values():
            domain = url_info['domain']
            if domain not in domain_counts:
                domain_counts[domain] = 0
            domain_counts[domain] += url_info['occurrences']
        
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # 生成建议
        recommendations = self._generate_recommendations()
        
        # 构建报告
        report = {
            'scan_metadata': {
                'bee_type': 'PrimeSeekerBee',
                'mutation_rate': '12.60%',
                'scan_date': self._get_current_timestamp(),
            },
            'scan_statistics': {
                'total_files_scanned': self.scan_stats['total_files'],
                'files_with_urls': self.scan_stats['files_with_urls'],
                'total_unique_urls': len(self.found_urls),
                'total_url_occurrences': self.scan_stats['total_urls'],
                'unique_domains': len(self.scan_stats['unique_domains']),
                'urls_by_type': self.scan_stats['urls_by_type'],
            },
            'urls_by_type': {
                url_type: urls[:20]  # 限制数量
                for url_type, urls in urls_by_type.items()
            },
            'top_domains': [
                {
                    'domain': domain,
                    'occurrences': count,
                    'url_count': len([u for u in self.found_urls.values() if u['domain'] == domain])
                }
                for domain, count in top_domains
            ],
            'github_repositories': self._extract_github_repos(),
            'recommendations': recommendations,
            'summary': {
                'urls_found': len(self.found_urls),
                'github_repos_found': len(self._extract_github_repos()),
                'has_github_repos': len(self._extract_github_repos()) > 0,
                'potential_targets': len([u for u in self.found_urls.values() if u['type'] in ['github', 'gitlab']]),
            }
        }
        
        # 保存报告
        report_path = self.report_dir / f"url_discovery_report_{self._get_current_timestamp()}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ URL发现报告已生成: {report_path}")
    
    def _extract_github_repos(self) -> List[Dict[str, Any]]:
        """
        提取GitHub仓库信息
        """
        repos = {}
        
        for url_info in self.found_urls.values():
            if url_info['type'] == 'github' and 'owner' in url_info and 'repo' in url_info:
                repo_key = f"{url_info['owner']}/{url_info['repo']}"
                if repo_key not in repos:
                    repos[repo_key] = {
                        'owner': url_info['owner'],
                        'repo': url_info['repo'],
                        'full_name': repo_key,
                        'url': f"https://github.com/{repo_key}",
                        'occurrences': 0,
                        'files_found': [],
                    }
                repos[repo_key]['occurrences'] += url_info['occurrences']
                repos[repo_key]['files_found'].extend(url_info['files_found'])
        
        # 去重files_found
        for repo in repos.values():
            repo['files_found'] = list(set(repo['files_found']))
        
        # 按出现次数排序
        sorted_repos = sorted(repos.values(), key=lambda x: x['occurrences'], reverse=True)
        
        return sorted_repos
    
    def _generate_recommendations(self) -> List[str]:
        """
        生成建议
        """
        recommendations = []
        
        if len(self.found_urls) > 0:
            recommendations.append(f"发现 {len(self.found_urls)} 个URL，建议审查外部依赖")
        
        github_repos = self._extract_github_repos()
        if len(github_repos) > 0:
            recommendations.append(f"发现 {len(github_repos)} 个GitHub仓库，建议检查依赖的活跃度")
            recommendations.append(f"建议关注以下仓库: {[repo['full_name'] for repo in github_repos[:3]]}")
        
        if 'github.com' in self.scan_stats['unique_domains']:
            recommendations.append("项目依赖GitHub，建议考虑镜像或本地备份")
        
        if 'cdn.jsdelivr.net' in self.scan_stats['unique_domains']:
            recommendations.append("项目使用CDN，建议检查CDN可用性和稳定性")
        
        return recommendations
    
    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳
        """
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
