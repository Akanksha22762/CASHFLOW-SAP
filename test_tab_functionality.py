#!/usr/bin/env python3
"""
Test Tab Functionality
Verify that the Revenue Analysis modal tabs work correctly
"""

import os
import sys

def test_tab_functionality():
    """Test that the tab functionality works correctly"""
    print("🔍 TESTING TAB FUNCTIONALITY")
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
        
        # Test 1: Check tab buttons exist
        print("\n📋 TEST 1: Tab Buttons")
        tab_buttons = [
            'onclick="showRevenueTab(\'historical\')"',
            'onclick="showRevenueTab(\'forecast\')"',
            'onclick="showRevenueTab(\'contracts\')"',
            'onclick="showRevenueTab(\'pricing\')"',
            'onclick="showRevenueTab(\'aging\')"'
        ]
        
        for button in tab_buttons:
            if button in content:
                print(f"✅ Found: {button}")
            else:
                print(f"❌ Missing: {button}")
                return False
        
        # Test 2: Check tab content exists
        print("\n📊 TEST 2: Tab Content")
        content_checks = [
            'Historical Revenue Trends',
            'Sales Forecast',
            'Customer Contracts',
            'Pricing Models',
            'Accounts Receivable Aging'
        ]
        
        for check in content_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 3: Check JavaScript function
        print("\n⚡ TEST 3: JavaScript Function")
        if 'function showRevenueTab(tabName)' in content:
            print("✅ Found: showRevenueTab function")
        else:
            print("❌ Missing: showRevenueTab function")
            return False
        
        # Test 4: Check console logging
        print("\n🔍 TEST 4: Debug Logging")
        debug_checks = [
            'console.log(\'Loading tab content for:\', tabName)',
            'console.log(\'Historical tab content loaded\')',
            'console.log(\'Forecast tab content loaded\')',
            'console.log(\'Contracts tab content loaded\')',
            'console.log(\'Pricing tab content loaded\')',
            'console.log(\'Aging tab content loaded\')'
        ]
        
        for check in debug_checks:
            if check in content:
                print(f"✅ Found: {check[:30]}...")
            else:
                print(f"❌ Missing: {check[:30]}...")
                return False
        
        # Test 5: Check tab content area
        print("\n📦 TEST 5: Tab Content Area")
        area_checks = [
            'id="revenueTabContent"',
            'class="tab-content"',
            'min-height: 300px',
            'background: #f8fafc'
        ]
        
        for check in area_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        print("\n🎉 ALL TAB TESTS PASSED!")
        print("✅ Tab functionality should work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING TAB FUNCTIONALITY")
    print("=" * 60)
    
    success = test_tab_functionality()
    
    if success:
        print("\n🎉 TAB TEST PASSED!")
        print("✅ Revenue Analysis modal tabs should work:")
        print("   📋 Click any tab button")
        print("   📊 Tab content should display")
        print("   🔄 Active tab should be highlighted")
        print("   📥 Content should load immediately")
        print("   🔍 Check browser console for debug logs")
        print("\n🚀 READY TO TEST!")
        print("Go to http://localhost:5000 and test the tabs!")
    else:
        print("\n❌ TAB TEST FAILED!")
        print("Tab functionality has issues") 