#!/usr/bin/env python3
"""
Debug script to show template content
"""

import requests

def debug_template():
    print("🔍 Debugging template content...")
    
    try:
        response = requests.get('http://localhost:5000/advanced-revenue-analysis')
        
        if response.status_code == 200:
            content = response.text
            
            # Check for specific content
            checks = [
                ("Operating Expenses (OPEX)", "OPEX Card"),
                ("Historical Revenue Trends", "Historical Trends Card"),
                ("Sales Forecast", "Sales Forecast Card"),
                ("Customer Contracts", "Customer Contracts Card"),
                ("Pricing Models", "Pricing Models Card"),
                ("AR Aging", "AR Aging Card"),
                ("feature-card", "Feature Card CSS Class"),
                ("feature-grid", "Feature Grid CSS Class")
            ]
            
            print("\n📊 Content Analysis:")
            for search_term, description in checks:
                if search_term in content:
                    print(f"✅ {description}: FOUND")
                else:
                    print(f"❌ {description}: NOT FOUND")
            
            # Find the feature cards section
            if "feature-grid" in content:
                start = content.find("feature-grid")
                end = content.find("</div>", start) + 6
                feature_section = content[start:end]
                
                print(f"\n📋 Feature Grid Section (first 1000 chars):")
                print(feature_section[:1000])
            else:
                print("\n❌ No feature-grid found in template")
                
        else:
            print(f"❌ Failed to get page. Status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    debug_template() 