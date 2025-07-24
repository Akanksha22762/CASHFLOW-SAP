import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

def generate_steel_plant_sap_data():
    """Generate Steel Plant SAP Data (AP/AR) with steel-specific transactions"""
    print("Generating Steel Plant SAP Data...")
    
    # Steel-specific transaction types
    steel_products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars', 'Steel Wire', 'Steel Beams', 'Steel Angles']
    steel_services = ['Maintenance Service', 'Equipment Repair', 'Quality Testing', 'Transportation', 'Warehousing', 'Technical Support']
    steel_materials = ['Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements', 'Refractory Materials']
    steel_utilities = ['Electricity', 'Water', 'Gas', 'Compressed Air', 'Steam', 'Oxygen']
    
    data = []
    for i in range(450):
        # Determine transaction type
        if random.random() < 0.6:  # 60% AP transactions
            transaction_type = 'Accounts Payable'
            descriptions = [
                f'Purchase - {random.choice(steel_materials)}',
                f'{random.choice(steel_services)} Charges',
                f'{random.choice(steel_utilities)} Bill Payment',
                f'Equipment Spare Parts',
                f'Raw Material Supply',
                f'Maintenance Contract',
                f'Transportation Charges',
                f'Quality Control Services'
            ]
        else:  # 40% AR transactions
            transaction_type = 'Accounts Receivable'
            descriptions = [
                f'Sale - {random.choice(steel_products)}',
                f'Export Sale - {random.choice(steel_products)}',
                f'Domestic Sale - {random.choice(steel_products)}',
                f'Commission Income',
                f'Scrap Sale',
                f'Service Income',
                f'Royalty Income',
                f'Interest Income'
            ]
        
        amount = random.uniform(50000, 5000000)
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Description': random.choice(descriptions),
            'Type': transaction_type,
            'Amount': round(amount, 2),
            'Status': random.choice(['Pending', 'Paid', 'Partially Paid', 'Overdue', 'Received']),
            'GL_Code': f'GL-{random.randint(100000, 999999)}',
            'Cost_Center': f'CC-{random.randint(100, 999)}',
            'Profit_Center': f'PC-{random.randint(1, 10)}',
            'Document_Number': f'DOC{random.randint(10000, 99999)}',
            'Vendor_Customer': f'{"Vendor" if transaction_type == "Accounts Payable" else "Customer"}_{random.randint(1, 100)}',
            'Payment_Terms': random.choice([30, 45, 60, 90]),
            'Currency': 'INR',
            'Tax_Code': f'TC-{random.randint(1, 5)}',
            'Segment': f'SEG-{random.randint(100, 999)}'
        })
    
    df = pd.DataFrame(data)
    df.to_excel('steel_plant_sap_data.xlsx', index=False)
    print(f"✅ Generated Steel Plant SAP Data: {len(df)} records")
    return df

def generate_steel_plant_bank_data():
    """Generate Steel Plant Bank Statement Data with steel-specific transactions"""
    print("Generating Steel Plant Bank Statement Data...")
    
    # Steel-specific transaction descriptions
    steel_inflows = [
        'Customer Payment - Steel Plates',
        'Customer Payment - Steel Coils', 
        'Customer Payment - Steel Pipes',
        'Export Sale Receipt',
        'Domestic Sale Receipt',
        'Commission Income',
        'Scrap Sale Receipt',
        'Interest Credit',
        'Loan Received',
        'Investment Income',
        'Government Grant',
        'Export Incentive'
    ]
    
    steel_outflows = [
        'Payment to Iron Ore Supplier',
        'Payment to Coal Supplier',
        'Payment to Equipment Vendor',
        'Electricity Bill Payment',
        'Water Bill Payment',
        'Gas Bill Payment',
        'Maintenance Service Payment',
        'Transportation Charges',
        'Quality Testing Payment',
        'Employee Salary Payment',
        'Tax Payment',
        'Loan Repayment',
        'Equipment Purchase',
        'Raw Material Purchase'
    ]
    
    data = []
    balance = 50000000  # Starting balance
    
    for i in range(500):
        if random.random() < 0.55:  # 55% outflows
            description = random.choice(steel_outflows)
            transaction_type = 'Debit'
            amount = random.uniform(10000, 1000000)
            balance -= amount
        else:  # 45% inflows
            description = random.choice(steel_inflows)
            transaction_type = 'Credit'
            amount = random.uniform(50000, 2000000)
            balance += amount
        
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Description': description,
            'Type': transaction_type,
            'Amount': round(amount, 2),
            'Balance': round(balance, 2),
            'Transaction_ID': f'TXN-{i+1:04d}',
            'Reference_Number': f'REF-{random.randint(10000, 99999)}',
            'Category': 'Operating' if 'Payment' in description or 'Sale' in description else 'Financial',
            'Bank_Account': f'ACC-{random.randint(1000, 9999)}',
            'Clearing_Status': random.choice(['Cleared', 'Pending', 'Rejected']),
            'Narration': f'{description} - {date.strftime("%Y-%m-%d")}'
        })
    
    df = pd.DataFrame(data)
    df.to_excel('steel_plant_bank_data.xlsx', index=False)
    print(f"✅ Generated Steel Plant Bank Data: {len(df)} records")
    return df

def generate_steel_production_data():
    """Generate Steel Production Data with manufacturing metrics"""
    print("Generating Steel Production Data...")
    
    steel_products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars', 'Steel Wire']
    production_lines = ['Blast Furnace Line', 'BOS Line', 'Continuous Casting', 'Hot Rolling Mill', 'Cold Rolling Mill', 'Pipe Manufacturing']
    quality_grades = ['Grade A', 'Grade B', 'Grade C', 'Premium Grade', 'Standard Grade']
    
    data = []
    for i in range(480):
        product = random.choice(steel_products)
        planned_capacity = random.uniform(1000, 10000)
        actual_output = planned_capacity * random.uniform(0.75, 1.05)
        
        data.append({
            'Production_ID': f'PROD-{i+1:04d}',
            'Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Production_Line': random.choice(production_lines),
            'Product_Type': product,
            'Planned_Capacity_Tonnes': round(planned_capacity, 2),
            'Actual_Output_Tonnes': round(actual_output, 2),
            'Efficiency_Rate': round(actual_output / planned_capacity, 3),
            'Quality_Grade': random.choice(quality_grades),
            'Energy_Consumption_MWh': round(actual_output * random.uniform(0.5, 1.5), 2),
            'Raw_Material_Consumption_Tonnes': round(actual_output * random.uniform(1.1, 1.3), 2),
            'Labor_Hours': round(actual_output * random.uniform(0.1, 0.3), 2),
            'Cost_Per_Tonne': round(random.uniform(30000, 80000), 2),
            'Waste_Percentage': round(random.uniform(0.01, 0.08), 3),
            'Downtime_Hours': round(random.uniform(0, 12), 2),
            'Maintenance_Hours': round(random.uniform(0, 8), 2),
            'Safety_Incidents': random.randint(0, 2),
            'Environmental_Compliance_Score': round(random.uniform(0.85, 1.0), 3),
            'Inventory_Level_Tonnes': round(random.uniform(500, 5000), 2),
            'Customer_Orders_Tonnes': round(random.uniform(800, 12000), 2),
            'Supplier_Delivery_Performance': round(random.uniform(0.8, 0.98), 3)
        })
    
    df = pd.DataFrame(data)
    df.to_excel('steel_production_data.xlsx', index=False)
    print(f"✅ Generated Steel Production Data: {len(df)} records")
    return df

def generate_steel_inventory_data():
    """Generate Steel Inventory Management Data"""
    print("Generating Steel Inventory Data...")
    
    steel_products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars', 'Steel Wire']
    raw_materials = ['Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements', 'Refractory Materials']
    warehouses = ['Raw Material Warehouse', 'Finished Goods Warehouse', 'Spare Parts Warehouse', 'Scrap Yard']
    
    data = []
    for i in range(460):
        if random.random() < 0.4:  # 40% raw materials
            item_type = 'Raw Material'
            item_name = random.choice(raw_materials)
            unit_cost = random.uniform(5000, 50000)
            reorder_point = random.randint(100, 1000)
        else:  # 60% finished goods
            item_type = 'Finished Goods'
            item_name = random.choice(steel_products)
            unit_cost = random.uniform(30000, 100000)
            reorder_point = random.randint(50, 500)
        
        current_stock = random.randint(100, 10000)
        max_stock = current_stock * random.uniform(1.5, 3.0)
        
        data.append({
            'Inventory_ID': f'INV-{i+1:04d}',
            'Item_Name': item_name,
            'Item_Type': item_type,
            'SKU': f'SKU-{i+1:04d}',
            'Warehouse': random.choice(warehouses),
            'Current_Stock_Tonnes': current_stock,
            'Reorder_Point_Tonnes': reorder_point,
            'Max_Stock_Tonnes': round(max_stock),
            'Unit_Cost_Per_Tonne': round(unit_cost, 2),
            'Total_Value': round(current_stock * unit_cost, 2),
            'Supplier': f'Supplier_{random.randint(1, 50)}',
            'Lead_Time_Days': random.randint(7, 45),
            'Turnover_Rate': round(random.uniform(2, 15), 2),
            'Last_Updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'Expiry_Date': (datetime.now() + timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
            'Storage_Cost_Per_Tonne': round(random.uniform(100, 1000), 2),
            'Quality_Grade': random.choice(['A', 'B', 'C']),
            'Location_Zone': random.choice(['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']),
            'Safety_Stock_Tonnes': round(reorder_point * 0.5),
            'Forecasted_Demand_Tonnes': round(random.uniform(500, 5000), 2)
        })
    
    df = pd.DataFrame(data)
    df.to_excel('steel_inventory_data.xlsx', index=False)
    print(f"✅ Generated Steel Inventory Data: {len(df)} records")
    return df

def generate_steel_customer_data():
    """Generate Steel Customer Data with steel industry specifics"""
    print("Generating Steel Customer Data...")
    
    steel_industries = ['Construction', 'Automotive', 'Infrastructure', 'Manufacturing', 'Energy', 'Aerospace', 'Shipbuilding', 'Railways']
    customer_tiers = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
    regions = ['North India', 'South India', 'East India', 'West India', 'Central India', 'International']
    steel_products = ['Steel Plates', 'Steel Coils', 'Steel Pipes', 'Steel Sheets', 'Steel Bars', 'Steel Wire']
    
    data = []
    for i in range(450):
        annual_revenue = random.uniform(1000000, 100000000)
        credit_limit = annual_revenue * random.uniform(0.1, 0.4)
        
        data.append({
            'Customer_ID': f'CUST-{i+1:04d}',
            'Customer_Name': f'Steel_Customer_{i+1}',
            'Industry': random.choice(steel_industries),
            'Customer_Tier': random.choice(customer_tiers),
            'Region': random.choice(regions),
            'Annual_Revenue': round(annual_revenue, 2),
            'Credit_Limit': round(credit_limit, 2),
            'Payment_Terms_Days': random.choice([30, 45, 60, 90]),
            'Preferred_Products': random.choice(steel_products),
            'Average_Order_Value': round(annual_revenue / random.randint(10, 100), 2),
            'Order_Frequency': random.randint(1, 50),
            'Customer_Since': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d'),
            'Last_Order_Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Total_Orders': random.randint(1, 200),
            'Payment_History_Score': round(random.uniform(0.5, 1.0), 3),
            'Days_Sales_Outstanding': random.randint(15, 90),
            'Customer_Satisfaction_Score': round(random.uniform(3.0, 5.0), 1),
            'Lifetime_Value': round(annual_revenue * random.uniform(1, 5), 2),
            'Churn_Risk': random.choice(['Low', 'Medium', 'High']),
            'Contact_Person': f'Contact_{i+1}',
            'Phone': f'+91-{random.randint(7000000000, 9999999999)}',
            'Email': f'customer{i+1}@steelcompany.com',
            'Address': f'Steel Address_{i+1}, {random.choice(regions)}',
            'Account_Manager': f'AM_{random.randint(1, 15)}',
            'Certification_Required': random.choice([True, False]),
            'Quality_Standards': random.choice(['ISO 9001', 'API', 'ASTM', 'BS', 'DIN'])
        })
    
    df = pd.DataFrame(data)
    df.to_excel('steel_customer_data.xlsx', index=False)
    print(f"✅ Generated Steel Customer Data: {len(df)} records")
    return df

def generate_steel_supplier_data():
    """Generate Steel Supplier Data"""
    print("Generating Steel Supplier Data...")
    
    supplier_types = ['Raw Material Supplier', 'Equipment Supplier', 'Service Provider', 'Transportation', 'Utilities']
    materials = ['Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements', 'Refractory Materials', 'Equipment Spares']
    regions = ['North India', 'South India', 'East India', 'West India', 'Central India', 'International']
    
    data = []
    for i in range(400):
        supplier_type = random.choice(supplier_types)
        annual_spend = random.uniform(100000, 10000000)
        
        data.append({
            'Supplier_ID': f'SUPP-{i+1:04d}',
            'Supplier_Name': f'Steel_Supplier_{i+1}',
            'Supplier_Type': supplier_type,
            'Primary_Material': random.choice(materials),
            'Region': random.choice(regions),
            'Annual_Spend': round(annual_spend, 2),
            'Payment_Terms_Days': random.choice([30, 45, 60, 90]),
            'Credit_Limit': round(annual_spend * random.uniform(0.2, 0.8), 2),
            'Lead_Time_Days': random.randint(1, 60),
            'Quality_Rating': round(random.uniform(3.0, 5.0), 1),
            'Delivery_Performance': round(random.uniform(0.7, 0.98), 3),
            'Contract_Start_Date': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d'),
            'Contract_End_Date': (datetime.now() + timedelta(days=random.randint(365, 1095))).strftime('%Y-%m-%d'),
            'Supplier_Since': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d'),
            'Last_Order_Date': (datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d'),
            'Total_Orders': random.randint(10, 500),
            'On_Time_Delivery_Rate': round(random.uniform(0.8, 0.98), 3),
            'Price_Competitiveness': round(random.uniform(0.7, 1.0), 3),
            'Contact_Person': f'Supplier_Contact_{i+1}',
            'Phone': f'+91-{random.randint(7000000000, 9999999999)}',
            'Email': f'supplier{i+1}@company.com',
            'Address': f'Supplier Address_{i+1}, {random.choice(regions)}',
            'Certification': random.choice(['ISO 9001', 'ISO 14001', 'OHSAS 18001', 'None']),
            'Risk_Level': random.choice(['Low', 'Medium', 'High'])
        })
    
    df = pd.DataFrame(data)
    df.to_excel('steel_supplier_data.xlsx', index=False)
    print(f"✅ Generated Steel Supplier Data: {len(df)} records")
    return df

def generate_steel_operational_data():
    """Generate Steel Plant Operational Data"""
    print("Generating Steel Operational Data...")
    
    departments = ['Blast Furnace', 'BOS Plant', 'Continuous Casting', 'Hot Rolling', 'Cold Rolling', 'Pipe Plant', 'Quality Control', 'Maintenance']
    shifts = ['Morning', 'Afternoon', 'Night']
    equipment = ['Blast Furnace #1', 'BOS Converter #1', 'Caster #1', 'Hot Mill #1', 'Cold Mill #1', 'Pipe Mill #1']
    
    data = []
    for i in range(480):
        department = random.choice(departments)
        planned_hours = random.randint(20, 24)
        actual_hours = planned_hours * random.uniform(0.8, 1.0)
        
        data.append({
            'Operational_ID': f'OP-{i+1:04d}',
            'Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Department': department,
            'Shift': random.choice(shifts),
            'Equipment': random.choice(equipment),
            'Planned_Operating_Hours': planned_hours,
            'Actual_Operating_Hours': round(actual_hours, 2),
            'Availability_Rate': round(actual_hours / planned_hours, 3),
            'Production_Output_Tonnes': round(random.uniform(100, 5000), 2),
            'Energy_Consumption_MWh': round(random.uniform(100, 5000), 2),
            'Raw_Material_Consumption_Tonnes': round(random.uniform(150, 7500), 2),
            'Labor_Hours': round(random.uniform(50, 500), 2),
            'Maintenance_Hours': round(random.uniform(0, 8), 2),
            'Downtime_Hours': round(random.uniform(0, 6), 2),
            'Quality_Defects_Tonnes': round(random.uniform(0, 100), 2),
            'Safety_Incidents': random.randint(0, 3),
            'Environmental_Compliance_Score': round(random.uniform(0.85, 1.0), 3),
            'Cost_Per_Tonne': round(random.uniform(25000, 75000), 2),
            'Efficiency_Rate': round(random.uniform(0.7, 0.95), 3),
            'Temperature_Celsius': round(random.uniform(800, 1600), 1),
            'Pressure_Bar': round(random.uniform(1, 50), 2),
            'Flow_Rate_Tonnes_Per_Hour': round(random.uniform(10, 200), 2)
        })
    
    df = pd.DataFrame(data)
    df.to_excel('steel_operational_data.xlsx', index=False)
    print(f"✅ Generated Steel Operational Data: {len(df)} records")
    return df

def main():
    """Generate all steel plant datasets"""
    print("🏭 GENERATING STEEL PLANT AI CASH FLOW MODEL DATASETS")
    print("=" * 60)
    
    # Generate all steel-specific datasets
    datasets = {}
    
    datasets['sap_data'] = generate_steel_plant_sap_data()
    datasets['bank_data'] = generate_steel_plant_bank_data()
    datasets['production'] = generate_steel_production_data()
    datasets['inventory'] = generate_steel_inventory_data()
    datasets['customers'] = generate_steel_customer_data()
    datasets['suppliers'] = generate_steel_supplier_data()
    datasets['operational'] = generate_steel_operational_data()
    
    # Create summary report
    summary = {
        'total_datasets': len(datasets),
        'total_records': sum(len(df) for df in datasets.values()),
        'datasets': {name: len(df) for name, df in datasets.items()},
        'generated_date': datetime.now().isoformat(),
        'steel_plant_specific': True
    }
    
    with open('steel_plant_datasets_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n📊 STEEL PLANT DATASET GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total Datasets Generated: {summary['total_datasets']}")
    print(f"Total Records Generated: {summary['total_records']:,}")
    print("\nIndividual Dataset Records:")
    for name, count in summary['datasets'].items():
        print(f"  • {name.replace('_', ' ').title()}: {count:,} records")
    
    print(f"\n💾 Summary saved to: steel_plant_datasets_summary.json")
    print("\n✅ All steel plant datasets generated successfully!")
    print("\n🏭 These datasets are specifically designed for steel manufacturing cash flow analysis!")

if __name__ == "__main__":
    main() 