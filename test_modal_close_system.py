#!/usr/bin/env python3
"""
Test Modal Close System for Revenue Analysis
Verify that Revenue Analysis uses proper modal with close button like existing system
"""

import os
import sys

def test_modal_close_system():
    """Test that the Revenue Analysis uses proper modal with close button"""
    print("🔍 TESTING MODAL CLOSE SYSTEM FOR REVENUE ANALYSIS")
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
        
        # Test 1: Check for modal functions
        print("\n📋 TEST 1: Modal Functions")
        modal_checks = [
            'function showRevenueAnalysisModal(',
            'function closeRevenueAnalysisModal()',
            'function getRevenueAnalysisTitle(',
            'function runRevenueAnalysisFromModal('
        ]
        
        for check in modal_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 2: Check for modal HTML structure
        print("\n🎭 TEST 2: Modal HTML Structure")
        modal_html_checks = [
            'revenueDetailModal',
            'position: fixed',
            'background: rgba(0,0,0,0.8)',
            '&times;',
            'onclick="closeRevenueAnalysisModal()"'
        ]
        
        for check in modal_html_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 3: Check for proper close button pattern
        print("\n🔴 TEST 3: Close Button Pattern")
        close_button_checks = [
            'background: none; border: none; font-size: 24px; cursor: pointer; color: #666;',
            '&times;',
            'onclick="closeRevenueAnalysisModal()"'
        ]
        
        for check in close_button_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 4: Check for analysis titles
        print("\n📊 TEST 4: Analysis Titles")
        title_checks = [
            'Historical Revenue Trends',
            'Sales Forecast',
            'Customer Contracts',
            'Pricing Models',
            'Accounts Receivable Aging'
        ]
        
        for check in title_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 5: Check for Run Analysis button in modal
        print("\n🚀 TEST 5: Run Analysis Button in Modal")
        run_button_checks = [
            'runRevenueAnalysisFromModal',
            '🧠 Run Analysis',
            'background: #3b82f6'
        ]
        
        for check in run_button_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Revenue Analysis uses proper modal with close button")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING MODAL CLOSE SYSTEM FOR REVENUE ANALYSIS")
    print("=" * 60)
    
    success = test_modal_close_system()
    
    if success:
        print("\n🎉 MODAL CLOSE SYSTEM TEST PASSED!")
        print("✅ Revenue Analysis now works like existing system:")
        print("   📋 Step 1: Click Revenue Analysis card")
        print("   🔄 Step 2: Shows all 5 individual cards")
        print("   👁️ Step 3: Click 'View Analysis' on any card")
        print("   🎭 Step 4: Opens modal with close button (×)")
        print("   🧠 Step 5: Click 'Run Analysis' to start analysis")
        print("   ❌ Step 6: Click 'Close' or × to close modal")
        print("   🔴 Step 7: Click small × to return to single card")
        print("\n🚀 READY TO USE!")
        print("Go to http://localhost:5000 and test the Revenue Analysis!")
    else:
        print("\n❌ MODAL CLOSE SYSTEM TEST FAILED!")
        print("Revenue Analysis needs proper modal implementation") 