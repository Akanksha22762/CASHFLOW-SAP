"""
Comprehensive indentation fix for app1.py
Fixes all indentation errors at once
"""

import re

def fix_all_indentation():
    print("🔧 Starting comprehensive indentation fix...")
    
    with open('app1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix common indentation patterns
    fixes_applied = 0
    
    # Fix 1: Remove extra spaces before if statements
    pattern1 = r'(\s{8,})if use_cache:'
    replacement1 = r'        if use_cache:'
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        fixes_applied += 1
        print("✅ Fixed: Extra spaces before 'if use_cache:'")
    
    # Fix 2: Fix import statements inside functions
    pattern2 = r'(\s{8,})import time'
    replacement2 = r'    import time'
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        fixes_applied += 1
        print("✅ Fixed: Import statement indentation")
    
    # Fix 3: Fix grid_search.fit(X) indentation
    pattern3 = r'(\s{4,})grid_search\.fit\(X\)'
    replacement3 = r'                    grid_search.fit(X)'
    if re.search(pattern3, content):
        content = re.sub(pattern3, replacement3, content)
        fixes_applied += 1
        print("✅ Fixed: grid_search.fit(X) indentation")
    
    # Fix 4: Fix for loops after else statements
    pattern4 = r'(\s{4,})else:\s*\n(\s{4,})for '
    replacement4 = r'\1else:\n\2    for '
    if re.search(pattern4, content):
        content = re.sub(pattern4, replacement4, content)
        fixes_applied += 1
        print("✅ Fixed: For loops after else statements")
    
    # Fix 5: Fix category assignment indentation
    pattern5 = r'(\s{12,})category = '
    replacement5 = r'            category = '
    if re.search(pattern5, content):
        content = re.sub(pattern5, replacement5, content)
        fixes_applied += 1
        print("✅ Fixed: Category assignment indentation")
    
    # Fix 6: Fix any remaining inconsistent indentation
    # Look for lines that should be indented but aren't
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            continue
        
        # Check for specific problematic patterns
        if 'for idx, row in df_processed.iterrows():' in line and i > 0:
            prev_line = lines[i-1].strip()
            if prev_line == 'else:':
                # This line should be indented
                if not line.startswith('    '):
                    line = '    ' + line.lstrip()
                    fixes_applied += 1
                    print(f"✅ Fixed: Line {i+1} - for loop after else")
        
        fixed_lines.append(line)
    
    # Write the fixed content back
    with open('app1.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"🎉 Comprehensive fix completed! Applied {fixes_applied} fixes.")
    print("📝 File has been updated with correct indentation.")

if __name__ == "__main__":
    fix_all_indentation() 