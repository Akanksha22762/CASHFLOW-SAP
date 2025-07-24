"""
Fix specific indentation errors in app1.py
"""

def fix_specific_indentation():
    print("🔧 Fixing specific indentation errors...")
    
    with open('app1.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixes_applied = 0
    
    # Fix line 765 - the malformed line
    if len(lines) >= 765:
        line_765 = lines[764]  # 0-indexed
        if 'ai_cache_manager.set(cache_key, result)' in line_765:
            # Check if this line is properly indented
            if not line_765.startswith('        '):
                lines[764] = '        ai_cache_manager.set(cache_key, result)\n'
                fixes_applied += 1
                print("✅ Fixed: Line 765 - ai_cache_manager.set indentation")
    
    # Fix the line before it (line 764) - the if statement
    if len(lines) >= 764:
        line_764 = lines[763]  # 0-indexed
        if 'if use_cache:' in line_764:
            # Check if this line is properly indented
            if not line_764.startswith('        '):
                lines[763] = '        if use_cache:\n'
                fixes_applied += 1
                print("✅ Fixed: Line 764 - if use_cache: indentation")
    
    # Fix any other similar issues
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Fix category assignment lines that are over-indented
        if 'category = categories[j]' in line and line.count(' ') > 12:
            lines[i] = '            category = categories[j] if j < len(categories) else \'Operating Activities\'\n'
            fixes_applied += 1
            print(f"✅ Fixed: Line {line_num} - category assignment over-indentation")
        
        # Fix any other malformed lines
        if '        if use_cache:' in line and line.count(' ') > 8:
            lines[i] = '        if use_cache:\n'
            fixes_applied += 1
            print(f"✅ Fixed: Line {line_num} - if use_cache: over-indentation")
    
    # Write back to file
    with open('app1.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"🎉 Specific fixes completed! Applied {fixes_applied} fixes.")

if __name__ == "__main__":
    fix_specific_indentation() 