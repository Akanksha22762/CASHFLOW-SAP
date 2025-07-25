#!/usr/bin/env python3
"""
Script to fix all indentation issues in app1.py
"""

def fix_indentation():
    with open('app1.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix the problematic section around line 7275-7285
        if i == 7274:  # Line 7275 (0-indexed)
            fixed_lines.append("        # MUCH MORE LENIENT MATCHING THRESHOLD\n")
        elif i == 7275:  # Line 7276 (0-indexed)
            fixed_lines.append("        if best_payment_match is not None and best_match_score >= 0.15:  # Only 15% threshold!\n")
        elif i == 7276:  # Line 7277 (0-indexed)
            # Skip the duplicate comment line
            i += 1
            continue
        elif i == 7277:  # Line 7278 (0-indexed)
            # Skip the duplicate if statement
            i += 1
            continue
        elif i == 7278:  # Line 7279 (0-indexed)
            # Skip the duplicate comment line
            i += 1
            continue
        elif i == 7279:  # Line 7280 (0-indexed)
            # Skip the duplicate matched_invoice_payments.append line
            i += 1
            continue
        elif i == 7280:  # Line 7281 (0-indexed)
            # This should be the correct matched_invoice_payments.append line
            fixed_lines.append("            matched_invoice_payments.append({\n")
        # Fix line 7306 - else statement indentation
        elif i == 7305:  # Line 7306 (0-indexed)
            fixed_lines.append("        else:\n")
        # Fix line 7717 - description_column assignment indentation
        elif i == 7716:  # Line 7717 (0-indexed)
            fixed_lines.append("            if 'desc' in col_lower or 'note' in col_lower or 'memo' in col_lower:\n")
        elif i == 7717:  # Line 7718 (0-indexed)
            fixed_lines.append("                description_column = col\n")
        elif i == 7718:  # Line 7719 (0-indexed)
            fixed_lines.append("                print(f\"🔍 FALLBACK DESCRIPTION: {col}\")\n")
        elif i == 7719:  # Line 7720 (0-indexed)
            fixed_lines.append("                break\n")
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content back
    with open('app1.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed all indentation issues!")

if __name__ == "__main__":
    fix_indentation() 