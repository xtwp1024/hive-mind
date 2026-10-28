# -*- coding: utf-8 -*-
"""
阶段 C: API 文档生成脚本

功能:
1. 自动分析所有核心模块
2. 提取类、函数、方法的 docstring
3. 生成 HTML 和 Markdown 格式的 API 文档
4. 创建 API 索引页面

使用方法:
    python hive/scripts/generate_api_docs.py
    python hive/scripts/generate_api_docs.py --format html
    python hive/scripts/generate_api_docs.py --format md
    python hive/scripts/generate_api_docs.py --format all
"""

import sys
import os
import re
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from enum import Enum
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_OUTPUT = PROJECT_ROOT / "docs" / "api"


@dataclass
class ParameterInfo:
    """参数信息"""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    description: Optional[str] = None


@dataclass
class MethodInfo:
    """方法/函数信息"""
    name: str
    docstring: Optional[str] = None
    parameters: List[ParameterInfo] = field(default_factory=list)
    return_annotation: Optional[str] = None
    is_async: bool = False
    is_public: bool = True
    is_classmethod: bool = False
    is_staticmethod: bool = False


@dataclass
class ClassInfo:
    """类信息"""
    name: str
    docstring: Optional[str] = None
    bases: List[str] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    is_public: bool = True
    is_dataclass: bool = False
    is_enum: bool = False


@dataclass
class ModuleInfo:
    """模块信息"""
    name: str
    full_path: str
    docstring: Optional[str] = None
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[MethodInfo] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)


# 核心模块列表 (需要生成文档的模块)
CORE_MODULES = [
    # Phase 1-2
    "hive.bees.cognitive_bee",
    "hive.core.swarm_communication",
    "hive.core.consensus",
    
    # Phase 3
    "hive.core.capability_gap_analyzer",
    "hive.core.species_blueprint",
    "hive.core.auto_species_generator",
    "hive.core.resource_allocator",
    "hive.core.dna_adaptive",
    
    # Phase 4
    "hive.core.multimodal_encoder",
    "hive.core.cross_modal_association",
    "hive.core.multimodal_task_decoder",
    
    # Phase 5
    "hive.core.hive_federation",
    "hive.core.knowledge_sync",
    "hive.core.node_auth",
    "hive.core.conflict_resolution",
    "hive.core.load_balancer",
    
    # PRD UI
    "hive.core.dispatcher",
    "hive.evolution.migrations",
    
    # 存储
    "hive.core.memory_store",
    "hive.core.memory_box",
]


def parse_docstring(docstring: Optional[str]) -> Dict[str, Any]:
    """
    解析 docstring
    
    支持 Google 风格、NumPy 风格、Epydoc 风格
    """
    result = {
        "description": "",
        "parameters": {},
        "returns": "",
        "raises": [],
        "examples": [],
        "notes": "",
    }
    
    if not docstring:
        return result
    
    lines = docstring.strip().split('\n')
    
    # 简单解析
    current_section = "description"
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        # 检查常见的 section 标签
        lower_line = line.lower()
        
        if lower_line.startswith("args:") or lower_line.startswith("arguments:") or lower_line.startswith("parameters:"):
            current_section = "parameters"
            continue
        elif lower_line.startswith("returns:") or lower_line.startswith("return:"):
            current_section = "returns"
            continue
        elif lower_line.startswith("raises:") or lower_line.startswith("raise:"):
            current_section = "raises"
            continue
        elif lower_line.startswith("example:") or lower_line.startswith("examples:"):
            current_section = "examples"
            continue
        elif lower_line.startswith("note:") or lower_line.startswith("notes:"):
            current_section = "notes"
            continue
        
        # 根据当前 section 处理
        if current_section == "description":
            if result["description"]:
                result["description"] += " " + line
            else:
                result["description"] = line
        elif current_section == "parameters":
            # 解析参数: "name: description"
            if ":" in line:
                name, desc = line.split(":", 1)
                result["parameters"][name.strip()] = desc.strip()
        elif current_section == "returns":
            if result["returns"]:
                result["returns"] += " " + line
            else:
                result["returns"] = line
        elif current_section == "raises":
            result["raises"].append(line)
        elif current_section == "examples":
            result["examples"].append(line)
        elif current_section == "notes":
            if result["notes"]:
                result["notes"] += " " + line
            else:
                result["notes"] = line
    
    return result


def is_public_name(name: str) -> bool:
    """检查是否是公开名称"""
    return not name.startswith("_")


def get_type_annotation(annotation) -> Optional[str]:
    """获取类型注解的字符串表示"""
    if annotation is inspect.Signature.empty:
        return None
    try:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)
    except Exception:
        return str(annotation)


def analyze_function(func, name: Optional[str] = None) -> MethodInfo:
    """分析函数/方法"""
    func_name = name or func.__name__
    
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        sig = None
    
    # 获取 docstring
    docstring = inspect.getdoc(func)
    
    # 解析参数
    params = []
    if sig:
        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name == "cls":
                continue
            
            default = None
            if param.default is not inspect.Signature.empty:
                try:
                    default = str(param.default)
                except Exception:
                    default = "<unrepresentable>"
            
            params.append(ParameterInfo(
                name=param_name,
                type_annotation=get_type_annotation(param.annotation),
                default_value=default,
            ))
    
    # 返回值注解
    return_ann = None
    if sig and sig.return_annotation is not inspect.Signature.empty:
        return_ann = get_type_annotation(sig.return_annotation)
    
    # 检查是否是异步
    is_async = inspect.iscoroutinefunction(func)
    
    return MethodInfo(
        name=func_name,
        docstring=docstring,
        parameters=params,
        return_annotation=return_ann,
        is_async=is_async,
        is_public=is_public_name(func_name),
    )


def analyze_class(cls: Type) -> ClassInfo:
    """分析类"""
    # 获取 docstring
    docstring = inspect.getdoc(cls)
    
    # 获取基类
    bases = []
    for base in cls.__bases__:
        if base is not object:
            bases.append(base.__name__)
    
    # 分析方法
    methods = []
    attributes = []
    
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") and not name.startswith("__"):
            continue
        
        if inspect.ismethod(member) or inspect.isfunction(member):
            # 普通方法/函数
            method_info = analyze_function(member, name)
            method_info.is_public = is_public_name(name)
            
            # 检查装饰器
            if isinstance(inspect.getattr_static(cls, name, None), classmethod):
                method_info.is_classmethod = True
            if isinstance(inspect.getattr_static(cls, name, None), staticmethod):
                method_info.is_staticmethod = True
            
            methods.append(method_info)
        elif not name.startswith("__"):
            # 可能是属性
            if not callable(member):
                attributes.append(name)
    
    # 检查是否是 dataclass 或 enum
    is_dataclass = hasattr(cls, "__dataclass_fields__")
    is_enum = issubclass(cls, Enum) if inspect.isclass(cls) else False
    
    return ClassInfo(
        name=cls.__name__,
        docstring=docstring,
        bases=bases,
        methods=methods,
        attributes=attributes,
        is_public=is_public_name(cls.__name__),
        is_dataclass=is_dataclass,
        is_enum=is_enum,
    )


def analyze_module(module_name: str) -> Optional[ModuleInfo]:
    """分析模块"""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"  [警告] 无法导入模块 {module_name}: {e}")
        return None
    
    # 获取模块 docstring
    docstring = inspect.getdoc(module)
    
    # 获取文件路径
    try:
        full_path = inspect.getfile(module)
    except TypeError:
        full_path = module.__name__
    
    # 分析类
    classes = []
    for name, member in inspect.getmembers(module, inspect.isclass):
        if member.__module__ == module_name:
            class_info = analyze_class(member)
            classes.append(class_info)
    
    # 分析函数
    functions = []
    for name, member in inspect.getmembers(module, inspect.isfunction):
        if member.__module__ == module_name:
            func_info = analyze_function(member, name)
            func_info.is_public = is_public_name(name)
            functions.append(func_info)
    
    # 导出列表 (从 __all__)
    exports = getattr(module, "__all__", [])
    
    return ModuleInfo(
        name=module_name,
        full_path=full_path,
        docstring=docstring,
        classes=classes,
        functions=functions,
        exports=exports,
    )


def generate_markdown_doc(module_info: ModuleInfo) -> str:
    """生成 Markdown 格式文档"""
    md = f"# {module_info.name}\n\n"
    
    # 模块描述
    if module_info.docstring:
        parsed = parse_docstring(module_info.docstring)
        if parsed["description"]:
            md += f"{parsed['description']}\n\n"
    
    # 导航
    if module_info.classes or module_info.functions:
        md += "## 目录\n\n"
        if module_info.classes:
            md += "- [类](#类)\n"
            for cls in module_info.classes:
                if cls.is_public:
                    md += f"  - [{cls.name}](#{cls.name.lower()})\n"
        if module_info.functions:
            public_funcs = [f for f in module_info.functions if f.is_public]
            if public_funcs:
                md += "- [函数](#函数)\n"
                for func in public_funcs:
                    md += f"  - [{func.name}](#{func.name.lower()})\n"
        md += "\n"
    
    # 类
    if module_info.classes:
        md += "## 类\n\n"
        
        for cls in module_info.classes:
            if not cls.is_public:
                continue
            
            # 类标题
            md += f"### {cls.name}\n\n"
            
            # 类装饰器标记
            tags = []
            if cls.is_dataclass:
                tags.append("@dataclass")
            if cls.is_enum:
                tags.append("Enum")
            if tags:
                md += f"**{', '.join(tags)}**\n\n"
            
            # 基类
            if cls.bases:
                md += f"**继承:** `{', '.join(cls.bases)}`\n\n"
            
            # 类描述
            if cls.docstring:
                parsed = parse_docstring(cls.docstring)
                if parsed["description"]:
                    md += f"{parsed['description']}\n\n"
            
            # 方法
            public_methods = [m for m in cls.methods if m.is_public]
            if public_methods:
                md += "#### 方法\n\n"
                
                for method in public_methods:
                    # 方法签名
                    sig_parts = []
                    for param in method.parameters:
                        param_str = param.name
                        if param.type_annotation:
                            param_str += f": {param.type_annotation}"
                        if param.default_value:
                            param_str += f" = {param.default_value}"
                        sig_parts.append(param_str)
                    
                    return_str = ""
                    if method.return_annotation:
                        return_str = f" -> {method.return_annotation}"
                    
                    async_prefix = "async " if method.is_async else ""
                    md += f"```python\n{async_prefix}def {method.name}({', '.join(sig_parts)}){return_str}\n```\n\n"
                    
                    # 方法描述
                    if method.docstring:
                        parsed = parse_docstring(method.docstring)
                        if parsed["description"]:
                            md += f"{parsed['description']}\n\n"
                        
                        # 参数说明
                        if parsed["parameters"]:
                            md += "**参数:**\n\n"
                            for param_name, desc in parsed["parameters"].items():
                                md += f"- `{param_name}`: {desc}\n"
                            md += "\n"
                        
                        # 返回值
                        if parsed["returns"]:
                            md += f"**返回:** {parsed['returns']}\n\n"
                        
                        # 示例
                        if parsed["examples"]:
                            md += "**示例:**\n\n```python\n"
                            md += "\n".join(parsed["examples"])
                            md += "\n```\n\n"
            
            # 属性
            if cls.attributes:
                md += "#### 属性\n\n"
                for attr in cls.attributes:
                    md += f"- `{attr}`\n"
                md += "\n"
    
    # 函数
    public_funcs = [f for f in module_info.functions if f.is_public]
    if public_funcs:
        md += "## 函数\n\n"
        
        for func in public_funcs:
            # 函数签名
            sig_parts = []
            for param in func.parameters:
                param_str = param.name
                if param.type_annotation:
                    param_str += f": {param.type_annotation}"
                if param.default_value:
                    param_str += f" = {param.default_value}"
                sig_parts.append(param_str)
            
            return_str = ""
            if func.return_annotation:
                return_str = f" -> {func.return_annotation}"
            
            async_prefix = "async " if func.is_async else ""
            md += f"### {func.name}\n\n"
            md += f"```python\n{async_prefix}def {func.name}({', '.join(sig_parts)}){return_str}\n```\n\n"
            
            # 函数描述
            if func.docstring:
                parsed = parse_docstring(func.docstring)
                if parsed["description"]:
                    md += f"{parsed['description']}\n\n"
                
                # 参数说明
                if parsed["parameters"]:
                    md += "**参数:**\n\n"
                    for param_name, desc in parsed["parameters"].items():
                        md += f"- `{param_name}`: {desc}\n"
                    md += "\n"
                
                # 返回值
                if parsed["returns"]:
                    md += f"**返回:** {parsed['returns']}\n\n"
                
                # 示例
                if parsed["examples"]:
                    md += "**示例:**\n\n```python\n"
                    md += "\n".join(parsed["examples"])
                    md += "\n```\n\n"
    
    # 导出列表
    if module_info.exports:
        md += "## 导出\n\n"
        md += f"```python\n__all__ = {module_info.exports}\n```\n\n"
    
    md += f"---\n\n*文档生成于: {datetime.now().isoformat()}*\n"
    
    return md


def generate_html_doc(module_info: ModuleInfo) -> str:
    """生成简化的 HTML 文档 (用于 pdoc3 格式)"""
    # 这里我们主要依赖 pdoc3 来生成完整的 HTML
    # 这个函数只生成一个简单的版本
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{module_info.name} - API Documentation</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 2rem; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 2rem; }}
        h3 {{ color: #5d6d7e; margin-top: 1.5rem; }}
        pre {{ background: #f4f4f4; padding: 1rem; border-radius: 4px; overflow-x: auto; }}
        code {{ font-family: 'Consolas', monospace; }}
        .class-section {{ margin: 1rem 0; padding: 1rem; border-left: 3px solid #3498db; }}
        .method-section {{ margin: 0.5rem 0; padding: 0.5rem; background: #fafafa; }}
    </style>
</head>
<body>
    <h1>{module_info.name}</h1>
    <p>文档生成于: {datetime.now().isoformat()}</p>
    
    <h2>类</h2>
    {''.join(f'''
    <div class="class-section">
        <h3>{cls.name}</h3>
        <p>继承: {', '.join(cls.bases) if cls.bases else 'object'}</p>
        {f'<p>{cls.docstring}</p>' if cls.docstring else ''}
        <h4>方法</h4>
        {''.join(f'''
        <div class="method-section">
            <pre><code>{'async ' if m.is_async else ''}def {m.name}(...){' -> ' + m.return_annotation if m.return_annotation else ''}</code></pre>
            {f'<p>{m.docstring}</p>' if m.docstring else ''}
        </div>
        ''' for m in cls.methods if m.is_public)}
    </div>
    ''' for cls in module_info.classes if cls.is_public)}
    
    <h2>函数</h2>
    {''.join(f'''
    <div class="method-section">
        <h3>{f.name}</h3>
        <pre><code>{'async ' if f.is_async else ''}def {f.name}(...){' -> ' + f.return_annotation if f.return_annotation else ''}</code></pre>
        {f'<p>{f.docstring}</p>' if f.docstring else ''}
    </div>
    ''' for f in module_info.functions if f.is_public)}
</body>
</html>
"""


def generate_index_page(module_infos: List[ModuleInfo]) -> str:
    """生成 API 索引页面"""
    md = "# Hive API 文档索引\n\n"
    md += f"> 生成时间: {datetime.now().isoformat()}\n\n"
    md += "## 模块列表\n\n"
    
    # 按阶段分组
    phase_groups = {
        "Phase 1-2: 认知觉醒与群体智慧": [],
        "Phase 3: 自主进化能力": [],
        "Phase 4: 多模态扩展": [],
        "Phase 5: 分布式网络": [],
        "PRD UI 控制台": [],
        "存储系统": [],
    }
    
    for info in module_infos:
        name = info.name
        
        if "cognitive" in name or "swarm" in name or "consensus" in name:
            phase_groups["Phase 1-2: 认知觉醒与群体智慧"].append(info)
        elif "gap" in name or "blueprint" in name or "species" in name or "resource" in name or "dna" in name:
            phase_groups["Phase 3: 自主进化能力"].append(info)
        elif "multimodal" in name or "cross_modal" in name:
            phase_groups["Phase 4: 多模态扩展"].append(info)
        elif "federation" in name or "sync" in name or "auth" in name or "conflict" in name or "balancer" in name:
            phase_groups["Phase 5: 分布式网络"].append(info)
        elif "dispatcher" in name or "migration" in name:
            phase_groups["PRD UI 控制台"].append(info)
        elif "memory" in name or "store" in name or "box" in name:
            phase_groups["存储系统"].append(info)
        else:
            if "其他" not in phase_groups:
                phase_groups["其他"] = []
            phase_groups["其他"].append(info)
    
    for phase_name, modules in phase_groups.items():
        if not modules:
            continue
        
        md += f"### {phase_name}\n\n"
        
        for info in modules:
            # 简短描述
            desc = ""
            if info.docstring:
                parsed = parse_docstring(info.docstring)
                if parsed["description"]:
                    desc = " - " + parsed["description"][:100]
                    if len(parsed["description"]) > 100:
                        desc += "..."
            
            md += f"- **{info.name}**{desc}\n"
            
            # 类和函数计数
            class_count = sum(1 for c in info.classes if c.is_public)
            func_count = sum(1 for f in info.functions if f.is_public)
            
            if class_count > 0 or func_count > 0:
                details = []
                if class_count > 0:
                    details.append(f"{class_count} 个类")
                if func_count > 0:
                    details.append(f"{func_count} 个函数")
                md += f"  - 包含: {', '.join(details)}\n"
        
        md += "\n"
    
    md += "## 统计\n\n"
    md += f"- **分析模块数:** {len(module_infos)}\n"
    md += f"- **总类数:** {sum(len(m.classes) for m in module_infos)}\n"
    md += f"- **总函数数:** {sum(len(m.functions) for m in module_infos)}\n\n"
    
    md += "---\n\n"
    md += "*本文档由 `generate_api_docs.py` 自动生成*\n"
    
    return md


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Hive API 文档生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 生成所有格式的文档
    python hive/scripts/generate_api_docs.py
    
    # 只生成 Markdown
    python hive/scripts/generate_api_docs.py --format md
    
    # 只生成 HTML (使用 pdoc3)
    python hive/scripts/generate_api_docs.py --format html
    
    # 指定模块
    python hive/scripts/generate_api_docs.py --module hive.core.dispatcher
        """
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["all", "md", "markdown", "html"],
        help="输出格式 (all, md, html)",
    )
    
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="指定单个模块生成文档",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(DOCS_OUTPUT),
        help="输出目录",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )
    
    args = parser.parse_args()
    
    # 确定输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定要分析的模块
    if args.module:
        modules = [args.module]
    else:
        modules = CORE_MODULES
    
    # 打印头部
    print()
    print("=" * 60)
    print("Hive API 文档生成器")
    print("=" * 60)
    print()
    print(f"输出目录: {output_dir}")
    print(f"模块数: {len(modules)}")
    print(f"格式: {args.format}")
    print()
    
    # 分析模块
    module_infos = []
    
    for module_name in modules:
        print(f"  分析模块: {module_name}")
        
        info = analyze_module(module_name)
        if info:
            module_infos.append(info)
            
            # 生成 Markdown
            if args.format in ["all", "md", "markdown"]:
                md_content = generate_markdown_doc(info)
                
                # 创建子目录结构
                module_path = module_name.replace(".", "/")
                md_file = output_dir / f"{module_path}.md"
                md_file.parent.mkdir(parents=True, exist_ok=True)
                
                md_file.write_text(md_content, encoding="utf-8")
                
                if args.verbose:
                    print(f"    生成: {md_file}")
                    class_count = sum(1 for c in info.classes if c.is_public)
                    func_count = sum(1 for f in info.functions if f.is_public)
                    print(f"    包含: {class_count} 个类, {func_count} 个函数")
        print()
    
    # 生成索引页面
    if module_infos:
        index_md = generate_index_page(module_infos)
        index_file = output_dir / "README.md"
        index_file.write_text(index_md, encoding="utf-8")
        print(f"  生成索引: {index_file}")
    
    # 生成 HTML (使用 pdoc3)
    if args.format in ["all", "html"]:
        print()
        print("  尝试使用 pdoc3 生成 HTML 文档...")
        
        try:
            import pdoc
            print(f"    pdoc 版本: {pdoc.__version__}")
            
            # 使用 pdoc 命令行
            import subprocess
            
            cmd = [
                sys.executable, "-m", "pdoc",
                "--html",
                "-o", str(output_dir / "html"),
                "--force",
            ]
            
            # 添加模块
            if args.module:
                cmd.append(args.module)
            else:
                # 只添加核心包
                cmd.extend([
                    "hive.core",
                    "hive.bees",
                    "hive.evolution",
                ])
            
            if args.verbose:
                print(f"    命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                print(f"    [OK] HTML 文档生成成功")
                print(f"    输出: {output_dir / 'html'}")
            else:
                print(f"    [警告] pdoc 执行出错")
                if args.verbose:
                    print(f"    stdout: {result.stdout}")
                    print(f"    stderr: {result.stderr}")
                    
        except ImportError:
            print("    [跳过] pdoc3 未安装")
            print("    提示: pip install pdoc3")
        except Exception as e:
            print(f"    [错误] {e}")
    
    # 汇总
    print()
    print("=" * 60)
    print("文档生成完成")
    print("=" * 60)
    print()
    print(f"  分析模块数: {len(module_infos)}")
    print(f"  总类数: {sum(len(m.classes) for m in module_infos)}")
    print(f"  总函数数: {sum(len(m.functions) for m in module_infos)}")
    print()
    print(f"  输出位置: {output_dir}")
    print()
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
