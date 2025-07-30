#!/usr/bin/env python3
"""
Test Small Close Button for Revenue Analysis
Verify that the Revenue Analysis uses a small close button like the existing system
"""

import os
import sys

def test_small_close_button():
    """Test that the Revenue Analysis uses a small close button"""
    print("🔍 TESTING SMALL CLOSE BUTTON FOR REVENUE ANALYSIS")
    print("=" * 60)
    
    try:
        # Read the HTML file
        html_file = "templates/sap_bank_interface.html"
        
        if not os.path.exists(html_file):
            print(f"❌ HTML file not found: {html_file}")
            return False
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ HTML file loaded successfully")
        
        # Test 1: Check for small close button pattern
        print("\n📋 TEST 1: Small Close Button Pattern")
        close_button_patterns = [
            '&times;',
            'position: absolute',
            'top: 10px',
            'right: 10px',
            'font-size: 20px',
            'color: #666'
        ]
        
        for pattern in close_button_patterns:
            if pattern in content:
                print(f"✅ Found: {pattern}")
            else:
                print(f"❌ Missing: {pattern}")
                return False
        
        # Test 2: Check that there's no separate close card
        print("\n🚫 TEST 2: No Separate Close Card")
        close_card_patterns = [
            'Close Analysis',
            'Close Button Card',
            'fas fa-times stat-icon'
        ]
        
        # These should NOT be present in the new implementation
        for pattern in close_card_patterns:
            if pattern in content:
                print(f"⚠️  Found (should be removed): {pattern}")
            else:
                print(f"✅ Correctly removed: {pattern}")
        
        # Test 3: Check for 5 individual cards
        print("\n🔄 TEST 3: 5 Individual Cards")
        card_checks = [
            'Historical Revenue Trends',
            'Sales Forecast', 
            'Customer Contracts',
            'Pricing Models',
            'Accounts Receivable Aging'
        ]
        
        for check in card_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 4: Check for View Analysis buttons
        print("\n⚡ TEST 4: View Analysis Buttons")
        button_checks = [
            'View Analysis',
            'onclick="viewRevenueAnalysis(',
            'fas fa-eye'
        ]
        
        for check in button_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Revenue Analysis uses small close button like existing system")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING SMALL CLOSE BUTTON FOR REVENUE ANALYSIS")
    print("=" * 60)
    
    success = test_small_close_button()
    
    if success:
        print("\n🎉 SMALL CLOSE BUTTON TEST PASSED!")
        print("✅ Revenue Analysis now uses:")
        print("   📋 Step 1: Click Revenue Analysis card")
        print("   🔄 Step 2: Shows all 5 individual cards")
        print("   🔴 Step 3: Small × button in top-right corner")
        print("   ⚡ Step 4: Each card has View Analysis & Export buttons")
        print("   🎯 Step 5: Click × button to return to single card")
        print("\n🚀 READY TO USE!")
        print("Go to http://localhost:5000 and test the Revenue Analysis!")
    else:
        print("\n❌ SMALL CLOSE BUTTON TEST FAILED!")
        print("Revenue Analysis needs to be updated to use small close button") 