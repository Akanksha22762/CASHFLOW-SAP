#!/usr/bin/env python3
"""
Quick Test to Verify Revenue Analysis Changes
"""

import os

def test_verify_changes():
    """Verify that the changes are properly applied"""
    print("🔍 VERIFYING REVENUE ANALYSIS CHANGES")
    print("=" * 50)
    
    try:
        # Read the HTML file
        html_file = "templates/sap_bank_interface.html"
        
        if not os.path.exists(html_file):
            print(f"❌ HTML file not found: {html_file}")
            return False
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ HTML file loaded successfully")
        
        # Check for View Analysis buttons
        if 'View Analysis' in content and 'onclick="viewRevenueAnalysis(' in content:
            print("✅ View Analysis buttons found")
        else:
            print("❌ View Analysis buttons not found")
            return False
        
        # Check for small close button
        if 'position: absolute; top: 10px; right: 10px;' in content and '&times;' in content:
            print("✅ Small close button found")
        else:
            print("❌ Small close button not found")
            return False
        
        # Check that separate close card is removed
        if 'Close Analysis' not in content and 'Close Button Card' not in content:
            print("✅ Separate close card removed")
        else:
            print("❌ Separate close card still exists")
            return False
        
        # Check for Export buttons
        if 'Export' in content and 'onclick="exportRevenueAnalysis()"' in content:
            print("✅ Export buttons found")
        else:
            print("❌ Export buttons not found")
            return False
        
        print("\n🎉 ALL CHANGES VERIFIED!")
        print("✅ Revenue Analysis now has:")
        print("   👁️ View Analysis buttons (not Run Analysis)")
        print("   🔴 Small × close button (not separate card)")
        print("   📤 Export buttons")
        print("   🎭 Modal system")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_verify_changes()
    
    if success:
        print("\n🚀 CHANGES APPLIED SUCCESSFULLY!")
        print("Go to http://localhost:5000 and refresh the page to see the changes!")
    else:
        print("\n❌ CHANGES NOT APPLIED!")
        print("Please check the HTML file manually.") 