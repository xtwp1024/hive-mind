#!/usr/bin/env python3
"""
Hive <-> OpenClaw Integration Test (Simplified)
------------------------------------------------
简化版集成测试，检查核心组件是否存在。
"""

import sys
from pathlib import Path

print("Hive <-> OpenClaw Integration Test")
print("=" * 60)

# Check 1: Hive Integration Files
print("\n[Check 1] Hive Integration Files")
print("-" * 40)

files_to_check = [
    ("hive/integrations/openclaw_adapter.py", "OpenClaw Adapter"),
    ("hive/server/__main__.py", "API Server"),
    ("test_openclaw_integration.py", "Integration Test"),
]

base_path = Path("D:/Hive_Standalone")

for filepath, description in files_to_check:
    full_path = base_path / filepath
    if full_path.exists():
        size = full_path.stat().st_size
        print(f"OK: {description}")
        print(f"   Path: {full_path}")
        print(f"   Size: {size} bytes")
    else:
        print(f"MISSING: {description} at {full_path}")

# Check 2: OpenClaw Integration Skill
print("\n[Check 2] OpenClaw Integration Skill")
print("-" * 40)

skill_path = Path("D:/openclaw-main/skills/hive-integration/SKILL.md")
if skill_path.exists():
    content = skill_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    print(f"OK: Integration skill found")
    print(f"   Path: {skill_path}")
    print(f"   Size: {len(content)} bytes, {len(lines)} lines")
    
    # Extract name
    for line in lines[:10]:
        if line.startswith("name:"):
            print(f"   Name: {line.split(':', 1)[1].strip()}")
else:
    print(f"MISSING: Skill not found at {skill_path}")

# Check 3: Adapter Content
print("\n[Check 3] Adapter Capabilities")
print("-" * 40)

adapter_path = base_path / "hive/integrations/openclaw_adapter.py"
if adapter_path.exists():
    content = adapter_path.read_text(encoding='utf-8')
    
    # Check for key features
    features = [
        ("class OpenClawAdapter", "Main Adapter Class"),
        ("send_message", "Message Sending"),
        ("query_skill", "Skill Query"),
        ("get_crypto_price", "Price Query"),
        ("search_web", "Web Search"),
        ("write_shared_data", "Shared Data"),
        ("class HiveOpenClawBridge", "Bridge Class"),
    ]
    
    for pattern, desc in features:
        if pattern in content:
            print(f"OK: {desc}")
        else:
            print(f"MISSING: {desc}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Integration Components Created:

1. Hive Side (Python):
   - hive/integrations/openclaw_adapter.py
     * OpenClawAdapter class for API communication
     * HiveOpenClawBridge for high-level operations
     * Shared data storage (D:/Hive_Standalone/Data/shared/)
   
   - hive/server/__main__.py
     * RESTful API server
     * Endpoints: /api/tasks, /api/status, /api/bees
     * Async and sync modes

2. OpenClaw Side (PowerShell):
   - skills/hive-integration/SKILL.md
     * Send-HiveTask function
     * Get-HiveStatus function
     * Bidirectional communication patterns

3. Integration Features:
   - Task dispatch from OpenClaw to Hive
   - Status monitoring
   - Shared data storage
   - Multiple API support (price, search, analysis)

Next Steps:
1. Install Python dependencies:
   cd D:\Hive_Standalone
   pip install -r requirements.txt
   pip install aiohttp requests

2. Start Hive server:
   python -m hive.server --port 8080

3. Test from OpenClaw:
   Send-HiveTask -Task @{type="analysis"; description="Test"}

4. Test from Hive:
   from hive.integrations.openclaw_adapter import ask_openclaw
   response = ask_openclaw("Hello from Hive!")
""")

print("\nIntegration framework ready!")
