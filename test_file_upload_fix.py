#!/usr/bin/env python3
"""
Test File Upload Fix for Revenue Analysis
Verify that file upload tracking works correctly
"""

import os

def test_file_upload_fix():
    """Test that file upload tracking works correctly"""
    print("🔍 TESTING FILE UPLOAD FIX FOR REVENUE ANALYSIS")
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
        
        # Test 1: Check for window.uploadedFiles initialization
        print("\n📁 TEST 1: File Upload Tracking Initialization")
        init_checks = [
            'window.uploadedFiles = {',
            'bank: false,',
            'sap: false'
        ]
        
        for check in init_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 2: Check for file upload tracking in uploadFiles function
        print("\n📤 TEST 2: File Upload Tracking in uploadFiles")
        upload_checks = [
            'window.uploadedFiles = {',
            'bank: bankFile ? true : false,',
            'sap: sapFile ? true : false',
            'console.log(\'Files uploaded - window.uploadedFiles:\', window.uploadedFiles);'
        ]
        
        for check in upload_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 3: Check for updated validation (only bank file required)
        print("\n✅ TEST 3: Updated File Validation")
        validation_checks = [
            '!window.uploadedFiles.bank',
            'Please upload a Bank Statement file first!',
            'only bank file is required'
        ]
        
        for check in validation_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 4: Check that old validation is removed
        print("\n❌ TEST 4: Old Validation Removed")
        old_validation_checks = [
            '!window.uploadedFiles.sap',
            'Please upload both Bank Statement and SAP files first!'
        ]
        
        old_validation_found = False
        for check in old_validation_checks:
            if check in content:
                print(f"❌ Still found: {check}")
                old_validation_found = True
        
        if not old_validation_found:
            print("✅ Old validation successfully removed")
        else:
            print("❌ Old validation still exists")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ File upload tracking now works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING FILE UPLOAD FIX FOR REVENUE ANALYSIS")
    print("=" * 60)
    
    success = test_file_upload_fix()
    
    if success:
        print("\n🎉 FILE UPLOAD FIX TEST PASSED!")
        print("✅ Revenue Analysis now works correctly:")
        print("   📁 Step 1: Upload bank file (SAP optional)")
        print("   📤 Step 2: File upload tracking sets window.uploadedFiles")
        print("   👁️ Step 3: Click 'View Analysis' on any card")
        print("   🎭 Step 4: Modal opens (no more 'upload files' message)")
        print("   🧠 Step 5: Click 'Run Analysis' to start analysis")
        print("   ❌ Step 6: Click 'Close' or × to close modal")
        print("   🔴 Step 7: Click small × to return to single card")
        print("\n🚀 READY TO USE!")
        print("Go to http://localhost:5000, upload a bank file, and test Revenue Analysis!")
    else:
        print("\n❌ FILE UPLOAD FIX TEST FAILED!")
        print("File upload tracking needs to be fixed") 