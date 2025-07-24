"""
Fix indentation issues in app1.py
"""

def fix_indentation():
    with open('app1.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix specific indentation issues
    fixed_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Fix line 766 - remove extra spaces before "if use_cache:"
        if line_num == 766 and 'if use_cache:' in line:
            # Remove extra spaces, keep proper indentation
            line = '        if use_cache:\n'
            print(f"Fixed line {line_num}: {line.strip()}")
        
        # Fix line 283 - ensure proper indentation for grid_search.fit(X)
        elif line_num == 283 and 'grid_search.fit(X)' in line:
            # Ensure proper indentation
            if not line.startswith('                    '):
                line = '                    grid_search.fit(X)\n'
                print(f"Fixed line {line_num}: {line.strip()}")
        
        # Fix line 2854 - remove extra spaces before "import time"
        elif line_num == 2854 and 'import time' in line:
            # Remove extra spaces, keep proper indentation
            line = '    import time\n'
            print(f"Fixed line {line_num}: {line.strip()}")
        
        fixed_lines.append(line)
    
    # Write back to file
    with open('app1.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Indentation fixes applied!")

if __name__ == "__main__":
    fix_indentation() 