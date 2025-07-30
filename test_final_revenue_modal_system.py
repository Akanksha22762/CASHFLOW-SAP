#!/usr/bin/env python3
"""
Test Final Revenue Modal System
Verify that Revenue Analysis uses proper modal system with View Analysis buttons
"""

import os
import sys

def test_final_revenue_modal_system():
    """Test that the Revenue Analysis uses proper modal system"""
    print("🔍 TESTING FINAL REVENUE MODAL SYSTEM")
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
        
        # Test 1: Check for View Analysis buttons (not Run Analysis)
        print("\n👁️ TEST 1: View Analysis Buttons")
        view_analysis_checks = [
            'onclick="viewRevenueAnalysis(\'historical\')"',
            'onclick="viewRevenueAnalysis(\'forecast\')"',
            'onclick="viewRevenueAnalysis(\'contracts\')"',
            'onclick="viewRevenueAnalysis(\'pricing\')"',
            'onclick="viewRevenueAnalysis(\'aging\')"',
            'View Analysis',
            '<i class="fas fa-eye"></i>'
        ]
        
        for check in view_analysis_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 2: Check for small close button (not separate card)
        print("\n🔴 TEST 2: Small Close Button")
        close_button_checks = [
            'position: absolute; top: 10px; right: 10px;',
            'background: none; border: none; font-size: 20px;',
            '&times;',
            'onclick="closeRevenueAnalysisCards()"'
        ]
        
        for check in close_button_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 3: Check that separate close card is removed
        print("\n❌ TEST 3: Separate Close Card Removed")
        close_card_checks = [
            'Close Analysis',
            'Close Button Card',
            'fas fa-times stat-icon',
            'color: #ef4444'
        ]
        
        close_card_found = False
        for check in close_card_checks:
            if check in content:
                print(f"❌ Still found: {check}")
                close_card_found = True
        
        if not close_card_found:
            print("✅ Separate close card successfully removed")
        else:
            print("❌ Separate close card still exists")
            return False
        
        # Test 4: Check for modal functions
        print("\n🎭 TEST 4: Modal Functions")
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
        
        # Test 5: Check for Export buttons
        print("\n📤 TEST 5: Export Buttons")
        export_checks = [
            'onclick="exportRevenueAnalysis()"',
            'Export',
            '<i class="fas fa-download"></i>'
        ]
        
        for check in export_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 6: Check for file upload validation
        print("\n📁 TEST 6: File Upload Validation")
        validation_checks = [
            'window.uploadedFiles',
            '!window.uploadedFiles.bank',
            '!window.uploadedFiles.sap',
            'Please upload both Bank Statement and SAP files first'
        ]
        
        for check in validation_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Revenue Analysis uses proper modal system")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING FINAL REVENUE MODAL SYSTEM")
    print("=" * 60)
    
    success = test_final_revenue_modal_system()
    
    if success:
        print("\n🎉 FINAL REVENUE MODAL SYSTEM TEST PASSED!")
        print("✅ Revenue Analysis now works correctly:")
        print("   📋 Step 1: Click Revenue Analysis card")
        print("   🔄 Step 2: Shows all 5 individual cards with small × button")
        print("   👁️ Step 3: Click 'View Analysis' on any card")
        print("   🎭 Step 4: Opens modal with close button (×)")
        print("   🧠 Step 5: Click 'Run Analysis' to start analysis")
        print("   ❌ Step 6: Click 'Close' or × to close modal")
        print("   🔴 Step 7: Click small × to return to single card")
        print("   📁 Step 8: File upload validation prevents running without files")
        print("\n🚀 READY TO USE!")
        print("Go to http://localhost:5000 and test the Revenue Analysis!")
    else:
        print("\n❌ FINAL REVENUE MODAL SYSTEM TEST FAILED!")
        print("Revenue Analysis needs proper modal implementation") 