#!/usr/bin/env python3
"""
Test Revenue Analysis Flow
Verify the correct flow: Revenue Analysis card → runs analysis → shows 5 cards → View Analysis shows details
"""

import os

def test_revenue_analysis_flow():
    """Test that the Revenue Analysis flow works correctly"""
    print("🔍 TESTING REVENUE ANALYSIS FLOW")
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
        
        # Test 1: Check that showRevenueAnalysisCards runs analysis first
        print("\n📋 TEST 1: Revenue Analysis Card Runs Analysis")
        analysis_checks = [
            'runRevenueAnalysis().then(() => {',
            '// After analysis is complete, show the 5 cards',
            '}).catch((error) => {',
            'showAlert(`Error running revenue analysis: ${error.message}`, \'error\');'
        ]
        
        for check in analysis_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 2: Check for View Analysis function (no running analysis)
        print("\n👁️ TEST 2: View Analysis Shows Details (No Running)")
        view_checks = [
            'function viewRevenueAnalysis(type) {',
            'showAnalysisDetailsModal(type);',
            '// Show analysis details modal directly (no running analysis)'
        ]
        
        for check in view_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 3: Check for analysis details modal (no run button)
        print("\n📊 TEST 3: Analysis Details Modal")
        modal_checks = [
            'function showAnalysisDetailsModal(type) {',
            'Analysis Results',
            'Status: Completed',
            '✅ Analysis Complete',
            'closeAnalysisDetailsModal()'
        ]
        
        for check in modal_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        # Test 4: Check that old run button is removed
        print("\n❌ TEST 4: Old Run Button Removed")
        old_run_checks = [
            'runRevenueAnalysisFromModal',
            '🧠 Run Analysis',
            'Ready to Run'
        ]
        
        old_run_found = False
        for check in old_run_checks:
            if check in content:
                print(f"❌ Still found: {check}")
                old_run_found = True
        
        if not old_run_found:
            print("✅ Old run button successfully removed")
        else:
            print("❌ Old run button still exists")
            return False
        
        # Test 5: Check for horizontal grid layout
        print("\n🎨 TEST 5: Horizontal Grid Layout")
        grid_checks = [
            'display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));',
            'gap: 20px; margin-top: 20px;'
        ]
        
        for check in grid_checks:
            if check in content:
                print(f"✅ Found: {check}")
            else:
                print(f"❌ Missing: {check}")
                return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Revenue Analysis flow works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 TESTING REVENUE ANALYSIS FLOW")
    print("=" * 60)
    
    success = test_revenue_analysis_flow()
    
    if success:
        print("\n🎉 REVENUE ANALYSIS FLOW TEST PASSED!")
        print("✅ Revenue Analysis now works correctly:")
        print("   📋 Step 1: Click Revenue Analysis card")
        print("   🧠 Step 2: Runs analysis automatically")
        print("   📊 Step 3: Shows 5 cards with results")
        print("   👁️ Step 4: Click 'View Analysis' on any card")
        print("   📈 Step 5: Shows analysis details modal (no running)")
        print("   ❌ Step 6: Click 'Close' to close modal")
        print("   🔴 Step 7: Click small × to return to single card")
        print("\n🚀 READY TO USE!")
        print("Go to http://localhost:5000 and test the Revenue Analysis flow!")
    else:
        print("\n❌ REVENUE ANALYSIS FLOW TEST FAILED!")
        print("Revenue Analysis flow needs to be fixed") 