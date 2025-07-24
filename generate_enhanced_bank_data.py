import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

def generate_enhanced_bank_data():
    """Generate comprehensive bank data with detailed descriptions for advanced analysis"""
    
    # Steel plant specific data
    steel_products = [
        'Steel Plates', 'Steel Coils', 'Steel Sheets', 'Steel Bars', 'Steel Pipes',
        'Steel Wire', 'Steel Beams', 'Steel Angles', 'Steel Channels', 'Steel Rods',
        'Hot Rolled Coils', 'Cold Rolled Sheets', 'Galvanized Steel', 'Color Coated Steel'
    ]
    
    raw_materials = [
        'Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements',
        'Refractory Materials', 'Lubricants', 'Chemicals', 'Oxygen', 'Nitrogen'
    ]
    
    customer_segments = [
        'Construction Company', 'Automotive Manufacturer', 'Shipbuilding Yard', 
        'Infrastructure Project', 'Oil & Gas Company', 'Railway Department',
        'Defense Contractor', 'Real Estate Developer', 'Engineering Firm'
    ]
    
    vendor_categories = [
        'Raw Material Supplier', 'Equipment Supplier', 'Service Provider',
        'Logistics Provider', 'Technology Provider', 'Maintenance Contractor'
    ]
    
    project_types = [
        'Bridge Construction', 'Highway Project', 'Metro Rail', 'Airport Terminal',
        'Commercial Building', 'Residential Complex', 'Industrial Plant',
        'Power Plant', 'Oil Refinery', 'Ship Building'
    ]
    
    bank_transactions = []
    
    # Operating Activities (65% of transactions)
    for i in range(450):
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        if random.random() < 0.65:  # 65% operating
            if random.random() < 0.65:  # 65% inflows (customer payments)
                transaction_type = 'Credit'
                
                # Enhanced customer payment descriptions
                customer = random.choice(customer_segments)
                product = random.choice(steel_products)
                project = random.choice(project_types) if random.random() < 0.4 else None
                
                # Payment patterns
                payment_patterns = [
                    f"Customer Payment - {customer} - {product} - Order #{random.randint(1000, 9999)} - Net 30",
                    f"Advance Payment - {customer} - {product} - 30% Advance - Project {project}",
                    f"Milestone Payment - {customer} - {product} - Phase {random.randint(1, 5)} - {project}",
                    f"Final Payment - {customer} - {product} - Contract #{random.randint(100, 999)}",
                    f"Retention Payment - {customer} - {product} - 10% Retention - {project}",
                    f"Q{random.randint(1, 4)} Payment - {customer} - {product} - Quarterly Settlement",
                    f"VIP Customer Payment - {customer} - {product} - Priority Order - Net 45",
                    f"New Customer Payment - {customer} - {product} - First Order - COD",
                    f"Bulk Order Payment - {customer} - {product} - {random.randint(100, 1000)} Tonnes",
                    f"Export Payment - {customer} - {product} - International Order - LC Payment"
                ]
                
                description = random.choice(payment_patterns)
                amount = random.uniform(50000, 500000)
                category = 'Operating - Revenue'
                
            else:  # 35% outflows (expenses)
                transaction_type = 'Debit'
                
                # Enhanced expense descriptions
                if random.random() < 0.4:  # Raw materials
                    material = random.choice(raw_materials)
                    supplier = f"{random.choice(vendor_categories)} {random.randint(1, 50)}"
                    quantity = random.randint(100, 2000)
                    
                    expense_patterns = [
                        f"Payment to {supplier} - {material} - {quantity} Tonnes - Net 30",
                        f"Raw Material Payment - {supplier} - {material} - Quality Grade A",
                        f"Procurement Payment - {supplier} - {material} - Bulk Order",
                        f"Import Payment - {supplier} - {material} - Customs Duty Included",
                        f"Supplier Payment - {supplier} - {material} - Contract Rate"
                    ]
                    description = random.choice(expense_patterns)
                    category = 'Operating - Raw Materials'
                    
                elif random.random() < 0.3:  # Utilities and services
                    service_patterns = [
                        f"Utility Payment - Electricity Bill - {random.randint(1000, 5000)} MWh - Monthly",
                        f"Water Supply Payment - Municipal Corporation - {random.randint(100, 500)} KL",
                        f"Gas Payment - Industrial Gas Supply - {random.randint(50, 200)} Cubic Meters",
                        f"Internet Payment - High Speed Connection - Monthly Subscription",
                        f"Telephone Payment - Landline & Mobile - Monthly Charges",
                        f"Maintenance Payment - Equipment Service - Preventive Maintenance",
                        f"Security Payment - Security Services - Monthly Contract",
                        f"Cleaning Payment - Housekeeping Services - Monthly"
                    ]
                    description = random.choice(service_patterns)
                    category = 'Operating - Services'
                    
                else:  # Other expenses
                    other_patterns = [
                        f"Salary Payment - Employee Payroll - {random.randint(50, 200)} Employees",
                        f"Bonus Payment - Performance Bonus - Q{random.randint(1, 4)}",
                        f"Insurance Payment - Plant Insurance - Annual Premium",
                        f"Legal Payment - Legal Services - Contract Review",
                        f"Accounting Payment - Audit Services - Annual Audit",
                        f"Marketing Payment - Advertisement - Trade Show",
                        f"Training Payment - Employee Training - Technical Skills",
                        f"Transport Payment - Logistics Services - Freight Charges"
                    ]
                    description = random.choice(other_patterns)
                    category = 'Operating - Expenses'
                
                amount = random.uniform(10000, 200000)
        
        # Investing Activities (20% of transactions)
        elif random.random() < 0.8:  # 20% of remaining
            if random.random() < 0.25:  # 25% inflows (asset sales)
                transaction_type = 'Credit'
                asset_patterns = [
                    f"Asset Sale Proceeds - Old Machinery - Blast Furnace Equipment - Scrap Value",
                    f"Equipment Sale - Surplus Rolling Mill - {random.randint(10, 50)} Years Old",
                    f"Property Sale - Industrial Land - {random.randint(1, 10)} Acres",
                    f"Scrap Metal Sale - Excess Steel Scrap - {random.randint(100, 1000)} Tonnes",
                    f"Investment Liquidation - Mutual Fund Units - Capital Gains",
                    f"Asset Disposal - Obsolete Equipment - Salvage Value"
                ]
                description = random.choice(asset_patterns)
                amount = random.uniform(50000, 300000)
                category = 'Investing - Asset Sales'
                
            else:  # 75% outflows (capital expenditure)
                transaction_type = 'Debit'
                capex_patterns = [
                    f"CapEx Payment - New Blast Furnace - Phase {random.randint(1, 3)} - Installation",
                    f"Equipment Purchase - Rolling Mill Upgrade - Advanced Technology",
                    f"Plant Expansion - New Production Line - Capacity Increase",
                    f"Technology Investment - Automation System - Industry 4.0",
                    f"Infrastructure Development - Warehouse Construction - {random.randint(1000, 5000)} sq ft",
                    f"Machinery Purchase - Quality Testing Equipment - ISO Standards",
                    f"Software Investment - ERP System - Digital Transformation",
                    f"Renovation Payment - Plant Modernization - Energy Efficiency"
                ]
                description = random.choice(capex_patterns)
                amount = random.uniform(100000, 1000000)
                category = 'Investing - Capital Expenditure'
        
        # Financing Activities (15% of transactions)
        else:
            if random.random() < 0.4:  # 40% inflows (borrowing)
                transaction_type = 'Credit'
                financing_patterns = [
                    f"Bank Loan Disbursement - Working Capital - {random.randint(5, 20)} Crores - 12% Interest",
                    f"Equipment Financing - New Machinery - {random.randint(2, 10)} Crores - 10% Interest",
                    f"Term Loan - Plant Expansion - {random.randint(10, 50)} Crores - 8% Interest",
                    f"Line of Credit - Short Term - {random.randint(1, 5)} Crores - 15% Interest",
                    f"Bridge Loan - Project Funding - {random.randint(5, 25)} Crores - 12% Interest",
                    f"Export Credit - International Order - {random.randint(1, 10)} Crores - 6% Interest"
                ]
                description = random.choice(financing_patterns)
                amount = random.uniform(500000, 2000000)
                category = 'Financing - Borrowing'
                
            else:  # 60% outflows (repayments)
                transaction_type = 'Debit'
                repayment_patterns = [
                    f"Loan EMI Payment - Principal + Interest - EMI #{random.randint(1, 60)}",
                    f"Interest Payment - Working Capital Loan - Monthly Interest",
                    f"Principal Repayment - Term Loan - Quarterly Payment",
                    f"Loan Closure Payment - Final Settlement - Early Closure",
                    f"Bank Charges - Processing Fee - Loan Maintenance",
                    f"Penalty Payment - Late Payment Charges - Overdue Interest"
                ]
                description = random.choice(repayment_patterns)
                amount = random.uniform(100000, 500000)
                category = 'Financing - Repayment'
        
        # Add seasonal and business cycle indicators
        if random.random() < 0.3:  # 30% chance to add seasonal indicators
            seasonal_indicators = [
                " - Q4 Year End",
                " - Monsoon Season",
                " - Festival Season",
                " - Summer Peak",
                " - Winter Slowdown",
                " - Financial Year End",
                " - Budget Period",
                " - Tax Season"
            ]
            description += random.choice(seasonal_indicators)
        
        bank_transactions.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Transaction_ID': f"BANK_{i+1:06d}",
            'Type': transaction_type,
            'Description': description,
            'Amount': round(amount, 2),
            'Category': category,
            'Account_Number': f"ACC_{random.randint(1000, 9999)}",
            'Reference_Number': f"REF_{random.randint(100000, 999999)}",
            'Balance': round(random.uniform(1000000, 5000000), 2),
            'Bank_Charges': round(random.uniform(0, 500), 2) if transaction_type == 'Debit' else 0,
            'Payment_Terms': extract_payment_terms(description),
            'Customer_Vendor': extract_customer_vendor(description),
            'Product_Type': extract_product_type(description),
            'Project_Reference': extract_project_reference(description),
            'Quantity': extract_quantity(description),
            'Seasonal_Indicator': extract_seasonal_indicator(description)
        })
    
    return pd.DataFrame(bank_transactions)

def extract_payment_terms(description):
    """Extract payment terms from description"""
    if 'Net 30' in description:
        return 'Net 30'
    elif 'Net 45' in description:
        return 'Net 45'
    elif 'Net 60' in description:
        return 'Net 60'
    elif 'COD' in description:
        return 'Cash on Delivery'
    elif 'Advance' in description:
        return 'Advance Payment'
    elif 'Milestone' in description:
        return 'Milestone Payment'
    elif 'Final' in description:
        return 'Final Payment'
    elif 'Retention' in description:
        return 'Retention Payment'
    else:
        return 'Standard Terms'

def extract_customer_vendor(description):
    """Extract customer/vendor name from description"""
    if 'Customer Payment' in description or 'Advance Payment' in description:
        # Extract customer name
        if 'Construction Company' in description:
            return 'Construction Company'
        elif 'Automotive Manufacturer' in description:
            return 'Automotive Manufacturer'
        elif 'Shipbuilding Yard' in description:
            return 'Shipbuilding Yard'
        elif 'Infrastructure Project' in description:
            return 'Infrastructure Project'
        elif 'Oil & Gas Company' in description:
            return 'Oil & Gas Company'
        elif 'Railway Department' in description:
            return 'Railway Department'
        elif 'Defense Contractor' in description:
            return 'Defense Contractor'
        elif 'Real Estate Developer' in description:
            return 'Real Estate Developer'
        elif 'Engineering Firm' in description:
            return 'Engineering Firm'
        elif 'VIP Customer' in description:
            return 'VIP Customer'
        elif 'New Customer' in description:
            return 'New Customer'
        else:
            return 'Unknown Customer'
    elif 'Payment to' in description or 'Supplier Payment' in description:
        # Extract vendor name
        if 'Raw Material Supplier' in description:
            return 'Raw Material Supplier'
        elif 'Equipment Supplier' in description:
            return 'Equipment Supplier'
        elif 'Service Provider' in description:
            return 'Service Provider'
        elif 'Logistics Provider' in description:
            return 'Logistics Provider'
        elif 'Technology Provider' in description:
            return 'Technology Provider'
        elif 'Maintenance Contractor' in description:
            return 'Maintenance Contractor'
        else:
            return 'Unknown Vendor'
    else:
        return 'Other'

def extract_product_type(description):
    """Extract product type from description"""
    products = [
        'Steel Plates', 'Steel Coils', 'Steel Sheets', 'Steel Bars', 'Steel Pipes',
        'Steel Wire', 'Steel Beams', 'Steel Angles', 'Steel Channels', 'Steel Rods',
        'Hot Rolled Coils', 'Cold Rolled Sheets', 'Galvanized Steel', 'Color Coated Steel',
        'Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements'
    ]
    
    for product in products:
        if product in description:
            return product
    
    return 'Other'

def extract_project_reference(description):
    """Extract project reference from description"""
    projects = [
        'Bridge Construction', 'Highway Project', 'Metro Rail', 'Airport Terminal',
        'Commercial Building', 'Residential Complex', 'Industrial Plant',
        'Power Plant', 'Oil Refinery', 'Ship Building'
    ]
    
    for project in projects:
        if project in description:
            return project
    
    return 'No Project'

def extract_quantity(description):
    """Extract quantity from description"""
    import re
    
    # Look for patterns like "100 Tonnes", "1000 sq ft", etc.
    quantity_patterns = [
        r'(\d+)\s+Tonnes',
        r'(\d+)\s+sq ft',
        r'(\d+)\s+Cubic Meters',
        r'(\d+)\s+MWh',
        r'(\d+)\s+KL',
        r'(\d+)\s+Employees',
        r'(\d+)\s+Acres'
    ]
    
    for pattern in quantity_patterns:
        match = re.search(pattern, description)
        if match:
            return int(match.group(1))
    
    return None

def extract_seasonal_indicator(description):
    """Extract seasonal indicator from description"""
    seasonal_indicators = [
        'Q1', 'Q2', 'Q3', 'Q4',
        'Monsoon Season', 'Festival Season', 'Summer Peak', 'Winter Slowdown',
        'Financial Year End', 'Budget Period', 'Tax Season', 'Year End'
    ]
    
    for indicator in seasonal_indicators:
        if indicator in description:
            return indicator
    
    return 'No Seasonal Indicator'

def main():
    """Generate enhanced bank data with detailed descriptions"""
    
    print("🏭 GENERATING ENHANCED BANK DATA WITH DETAILED DESCRIPTIONS")
    print("=" * 70)
    
    # Generate enhanced bank data
    bank_df = generate_enhanced_bank_data()
    
    # Save to steel_plant_datasets directory
    if not os.path.exists('steel_plant_datasets'):
        os.makedirs('steel_plant_datasets')
    
    filepath = os.path.join('steel_plant_datasets', 'enhanced_steel_plant_bank_data.xlsx')
    bank_df.to_excel(filepath, index=False)
    
    print(f"✅ Enhanced Bank Data: {len(bank_df)} records")
    print(f"   File: {filepath}")
    print(f"   Size: {round(os.path.getsize(filepath) / (1024*1024), 2)} MB")
    
    # Show sample descriptions
    print("\n📋 SAMPLE ENHANCED DESCRIPTIONS:")
    print("-" * 50)
    
    sample_descriptions = [
        "Customer Payment - Construction Company - Steel Plates - Order #1234 - Net 30",
        "Advance Payment - Automotive Manufacturer - Steel Coils - 30% Advance - Project Bridge Construction",
        "Milestone Payment - Shipbuilding Yard - Steel Beams - Phase 2 - Metro Rail",
        "Payment to Raw Material Supplier 15 - Iron Ore - 1000 Tonnes - Net 30",
        "CapEx Payment - New Blast Furnace - Phase 1 - Installation",
        "Bank Loan Disbursement - Working Capital - 10 Crores - 12% Interest",
        "Loan EMI Payment - Principal + Interest - EMI #15",
        "Asset Sale Proceeds - Old Machinery - Blast Furnace Equipment - Scrap Value - Q4 Year End"
    ]
    
    for i, desc in enumerate(sample_descriptions, 1):
        print(f"{i}. {desc}")
    
    # Show extracted information
    print("\n🔍 EXTRACTED INFORMATION FROM DESCRIPTIONS:")
    print("-" * 50)
    
    # Sample analysis
    sample_row = bank_df.iloc[0]
    print(f"Payment Terms: {sample_row['Payment_Terms']}")
    print(f"Customer/Vendor: {sample_row['Customer_Vendor']}")
    print(f"Product Type: {sample_row['Product_Type']}")
    print(f"Project Reference: {sample_row['Project_Reference']}")
    print(f"Quantity: {sample_row['Quantity']}")
    print(f"Seasonal Indicator: {sample_row['Seasonal_Indicator']}")
    
    # Create summary
    summary = {
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_transactions': len(bank_df),
        'operating_activities': len(bank_df[bank_df['Category'].str.contains('Operating')]),
        'investing_activities': len(bank_df[bank_df['Category'].str.contains('Investing')]),
        'financing_activities': len(bank_df[bank_df['Category'].str.contains('Financing')]),
        'unique_customers': bank_df['Customer_Vendor'].nunique(),
        'unique_products': bank_df['Product_Type'].nunique(),
        'unique_projects': bank_df['Project_Reference'].nunique(),
        'payment_terms_distribution': bank_df['Payment_Terms'].value_counts().to_dict(),
        'seasonal_indicators': bank_df['Seasonal_Indicator'].value_counts().to_dict()
    }
    
    summary_file = os.path.join('steel_plant_datasets', 'enhanced_bank_data_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 SUMMARY SAVED: {summary_file}")
    print(f"🎯 ENHANCED BANK DATA READY FOR ADVANCED ANALYSIS!")
    print(f"   All 13 cash flow parameters can now be extracted from descriptions!")

if __name__ == "__main__":
    main() 