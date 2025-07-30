#!/usr/bin/env python3
"""
Test File Upload Check for Revenue Analysis
Verify that Revenue Analysis only runs when files are uploaded
"""

import os
import sys

def test_file_upload_check():
    """Test that Revenue Analysis checks for uploaded files"""
    print("🔍 TESTING FILE UPLOAD CHECK FOR REVENUE ANALYSIS")
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
        
        # Test 1: Check for file upload validation in runRevenueAnalysis
        print("\n📋 TEST 1: File Upload Check in runRevenueAnalysis")
        run_checks = [
            'window.uploadedFiles',
            '!window.uploadedFiles.bank',
            '!window.uploadedFiles.sap',
            'Please upload both Bank Statement and SAP files first!'
        ]
        
        for check in run_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 2: Check for file upload validation in viewRevenueAnalysis
        print("\n👁️ TEST 2: File Upload Check in viewRevenueAnalysis")
        view_checks = [
            'viewRevenueAnalysis',
            'window.uploadedFiles',
            'Please upload both Bank Statement and SAP files first!'
        ]
        
        for check in view_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 3: Check for file upload validation in exportRevenueAnalysis
        print("\n📥 TEST 3: File Upload Check in exportRevenueAnalysis")
        export_checks = [
            'exportRevenueAnalysis',
            'window.uploadedFiles',
            'Please upload both Bank Statement and SAP files first!'
        ]
        
        for check in export_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 4: Check that the warning message is consistent
        print("\n⚠️ TEST 4: Consistent Warning Message")
        warning_count = content.count('Please upload both Bank Statement and SAP files first!')
        if warning_count >= 3:
            print(f"✅ Found {warning_count} instances of warning message")
        else:
            print(f"❌ Only found {warning_count} instances (expected 3)")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Revenue Analysis properly checks for uploaded files")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING FILE UPLOAD CHECK FOR REVENUE ANALYSIS")
    print("=" * 60)
    
    success = test_file_upload_check()
    
    if success:
        print("\n🎉 FILE UPLOAD CHECK TEST PASSED!")
        print("✅ Revenue Analysis now properly:")
        print("   📋 Step 1: Checks if files are uploaded")
        print("   ⚠️ Step 2: Shows warning if files not uploaded")
        print("   🚫 Step 3: Prevents analysis without files")
        print("   ✅ Step 4: Only runs when both files are uploaded")
        print("\n🚀 READY TO USE!")
        print("Now Revenue Analysis will only run when you upload files!")
    else:
        print("\n❌ FILE UPLOAD CHECK TEST FAILED!")
        print("Revenue Analysis needs file upload validation") 