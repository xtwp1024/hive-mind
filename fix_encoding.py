import os
import glob

def fix_file_encoding(filepath, encoding_out='utf-8'):
    """Reads a file and rewrites it in the target encoding."""
    try:
        # Try reading with utf-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Try reading with gbk (common cause of issues on CN Windows)
            with open(filepath, 'r', encoding='gbk') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Could not read {filepath}: {e}")
            return False

    try:
        with open(filepath, 'w', encoding=encoding_out) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"❌ Could not write {filepath}: {e}")
        return False

def main():
    print("🔍 Starting Global Encoding Scan...")
    
    # 1. Fix log_bug.ps1 specifically (Needs BOM for PowerShell 5.1)
    target_ps1 = 'log_bug.ps1'
    if os.path.exists(target_ps1):
        print(f"🔧 Fixing {target_ps1} (To UTF-8-SIG/BOM)...")
        if fix_file_encoding(target_ps1, 'utf-8-sig'):
            print(f"✅ {target_ps1} fixed.")
    
    # 2. Scan other text files (Python, Markdown, JSON, YAML)
    extensions = ['*.py', '*.md', '*.json', '*.yaml', '*.txt']
    files = []
    for ext in extensions:
        files.extend(glob.glob(f"**/{ext}", recursive=True))
    
    for fw in files:
        if 'venv' in fw or '.git' in fw or '__pycache__' in fw:
            continue
            
        # Standardize to utf-8 (no BOM usually best for code, unless PS1)
        # We just verify we can read it as utf-8. If not, we convert.
        try:
            with open(fw, 'r', encoding='utf-8') as f:
                f.read()
            # print(f"✅ {fw} is valid UTF-8")
        except UnicodeDecodeError:
            print(f"⚠️  {fw} is NOT UTF-8. Attempting conversion from GBK...")
            if fix_file_encoding(fw, 'utf-8'):
                print(f"✅ {fw} converted to UTF-8.")

    print("🏁 Encoding scan complete.")

if __name__ == "__main__":
    main()
