import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

def generate_complete_steel_plant_sap_data():
    """Generate comprehensive SAP AP/AR data with all cash flow categories"""
    
    # Steel plant specific data
    steel_products = [
        'Steel Plates', 'Steel Coils', 'Steel Sheets', 'Steel Bars', 'Steel Pipes',
        'Steel Wire', 'Steel Beams', 'Steel Angles', 'Steel Channels', 'Steel Rods'
    ]
    
    raw_materials = [
        'Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements',
        'Refractory Materials', 'Lubricants', 'Chemicals', 'Oxygen', 'Nitrogen'
    ]
    
    equipment_services = [
        'Blast Furnace Maintenance', 'Rolling Mill Equipment', 'Crane Services',
        'Conveyor Systems', 'Cooling Systems', 'Compressor Maintenance',
        'Electrical Systems', 'Safety Equipment', 'Quality Testing Equipment'
    ]
    
    # Operating Activities (60% of transactions)
    operating_transactions = []
    for i in range(400):
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        if random.random() < 0.6:  # 60% operating
            if random.random() < 0.7:  # 70% revenue
                transaction_type = 'AR'
                description = f"Sale - {random.choice(steel_products)}"
                amount = random.uniform(50000, 500000)
                category = 'Operating - Revenue'
            else:  # 30% expenses
                transaction_type = 'AP'
                if random.random() < 0.6:
                    description = f"Purchase - {random.choice(raw_materials)}"
                    category = 'Operating - Raw Materials'
                else:
                    description = f"Payment - {random.choice(equipment_services)}"
                    category = 'Operating - Services'
                amount = random.uniform(10000, 200000)
        
        # Investing Activities (25% of transactions)
        elif random.random() < 0.85:  # 25% of remaining
            transaction_type = 'AP'
            if random.random() < 0.3:  # 30% inflows (asset sales)
                description = f"Sale - {random.choice(['Old Machinery', 'Scrap Equipment', 'Surplus Materials'])}"
                amount = random.uniform(50000, 300000)
                category = 'Investing - Asset Sales'
            else:  # 70% outflows (capital expenditure)
                description = f"Purchase - {random.choice(['New Blast Furnace', 'Rolling Mill Equipment', 'Crane System', 'Conveyor Belt', 'Cooling Tower'])}"
                amount = random.uniform(100000, 1000000)
                category = 'Investing - Capital Expenditure'
        
        # Financing Activities (15% of transactions)
        else:
            if random.random() < 0.4:  # 40% inflows (borrowing)
                transaction_type = 'AR'
                description = f"{random.choice(['Bank Loan', 'Working Capital Loan', 'Equipment Loan'])}"
                amount = random.uniform(500000, 2000000)
                category = 'Financing - Borrowing'
            else:  # 60% outflows (repayments)
                transaction_type = 'AP'
                description = f"{random.choice(['Loan Repayment', 'Interest Payment', 'Principal Payment'])}"
                amount = random.uniform(100000, 500000)
                category = 'Financing - Repayment'
        
        operating_transactions.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Transaction_ID': f"TXN_{i+1:06d}",
            'Type': transaction_type,
            'Description': description,
            'Amount': round(amount, 2),
            'Category': category,
            'Vendor_Customer': f"{'Customer' if transaction_type == 'AR' else 'Vendor'}_{random.randint(1, 50):03d}",
            'Payment_Terms': random.choice(['Net 30', 'Net 45', 'Net 60', 'Immediate']),
            'Status': random.choice(['Completed', 'Pending', 'Overdue']),
            'Department': random.choice(['Production', 'Procurement', 'Sales', 'Finance', 'Operations'])
        })
    
    return pd.DataFrame(operating_transactions)

def generate_complete_steel_plant_bank_data():
    """Generate comprehensive bank statement data with all cash flow categories"""
    
    # Steel plant specific transactions
    steel_products = [
        'Steel Plates', 'Steel Coils', 'Steel Sheets', 'Steel Bars', 'Steel Pipes',
        'Steel Wire', 'Steel Beams', 'Steel Angles', 'Steel Channels', 'Steel Rods'
    ]
    
    raw_materials = [
        'Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements',
        'Refractory Materials', 'Lubricants', 'Chemicals', 'Oxygen', 'Nitrogen'
    ]
    
    # Operating Activities (65% of transactions)
    bank_transactions = []
    for i in range(450):
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        if random.random() < 0.65:  # 65% operating
            if random.random() < 0.65:  # 65% inflows
                transaction_type = 'Credit'
                description = f"Customer Payment - {random.choice(steel_products)}"
                amount = random.uniform(50000, 500000)
                category = 'Operating - Revenue'
            else:  # 35% outflows
                transaction_type = 'Debit'
                if random.random() < 0.7:
                    description = f"Payment to {random.choice(raw_materials)} Supplier"
                    category = 'Operating - Raw Materials'
                else:
                    description = f"Payment - {random.choice(['Utility Bills', 'Employee Salaries', 'Maintenance Services', 'Insurance Premium'])}"
                    category = 'Operating - Expenses'
                amount = random.uniform(10000, 200000)
        
        # Investing Activities (20% of transactions)
        elif random.random() < 0.8:  # 20% of remaining
            if random.random() < 0.25:  # 25% inflows
                transaction_type = 'Credit'
                description = f"Sale Proceeds - {random.choice(['Old Equipment', 'Scrap Metal', 'Surplus Materials'])}"
                amount = random.uniform(50000, 300000)
                category = 'Investing - Asset Sales'
            else:  # 75% outflows
                transaction_type = 'Debit'
                description = f"Payment - {random.choice(['New Machinery', 'Plant Equipment', 'Technology Upgrade', 'Infrastructure Development'])}"
                amount = random.uniform(100000, 1000000)
                category = 'Investing - Capital Expenditure'
        
        # Financing Activities (15% of transactions)
        else:
            if random.random() < 0.4:  # 40% inflows
                transaction_type = 'Credit'
                description = f"{random.choice(['Bank Loan Disbursement', 'Working Capital Advance', 'Equipment Financing'])}"
                amount = random.uniform(500000, 2000000)
                category = 'Financing - Borrowing'
            else:  # 60% outflows
                transaction_type = 'Debit'
                description = f"{random.choice(['Loan Repayment', 'Interest Payment', 'Principal Payment', 'Bank Charges'])}"
                amount = random.uniform(100000, 500000)
                category = 'Financing - Repayment'
        
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
            'Bank_Charges': round(random.uniform(0, 500), 2) if transaction_type == 'Debit' else 0
        })
    
    return pd.DataFrame(bank_transactions)

def generate_complete_production_data():
    """Generate comprehensive production data with detailed metrics"""
    
    production_lines = [
        'Blast Furnace Line 1', 'Blast Furnace Line 2', 'Basic Oxygen Furnace',
        'Electric Arc Furnace', 'Continuous Casting Line 1', 'Continuous Casting Line 2',
        'Hot Rolling Mill', 'Cold Rolling Mill', 'Galvanizing Line', 'Coating Line'
    ]
    
    products = [
        'Steel Plates', 'Steel Coils', 'Steel Sheets', 'Steel Bars', 'Steel Pipes',
        'Steel Wire', 'Steel Beams', 'Steel Angles', 'Steel Channels', 'Steel Rods'
    ]
    
    production_data = []
    for i in range(600):
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        line = random.choice(production_lines)
        product = random.choice(products)
        
        # Production metrics
        planned_output = random.uniform(100, 1000)
        actual_output = planned_output * random.uniform(0.85, 1.15)  # 85-115% efficiency
        energy_consumption = actual_output * random.uniform(0.8, 1.2)  # MWh per tonne
        labor_hours = actual_output * random.uniform(0.5, 1.5)  # hours per tonne
        
        production_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Production_Line': line,
            'Product': product,
            'Planned_Output_Tonnes': round(planned_output, 2),
            'Actual_Output_Tonnes': round(actual_output, 2),
            'Efficiency_Percentage': round((actual_output / planned_output) * 100, 2),
            'Energy_Consumption_MWh': round(energy_consumption, 2),
            'Labor_Hours': round(labor_hours, 2),
            'Quality_Score': round(random.uniform(85, 99), 2),
            'Downtime_Hours': round(random.uniform(0, 8), 2),
            'Maintenance_Status': random.choice(['Normal', 'Scheduled', 'Emergency', 'Preventive']),
            'Raw_Material_Consumption_Tonnes': round(actual_output * random.uniform(1.1, 1.3), 2),
            'Waste_Generation_Tonnes': round(actual_output * random.uniform(0.05, 0.15), 2),
            'Cost_Per_Tonne': round(random.uniform(30000, 60000), 2),
            'Revenue_Per_Tonne': round(random.uniform(45000, 80000), 2)
        })
    
    return pd.DataFrame(production_data)

def generate_complete_inventory_data():
    """Generate comprehensive inventory data for all materials"""
    
    raw_materials = [
        'Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements',
        'Refractory Materials', 'Lubricants', 'Chemicals', 'Oxygen', 'Nitrogen'
    ]
    
    finished_goods = [
        'Steel Plates', 'Steel Coils', 'Steel Sheets', 'Steel Bars', 'Steel Pipes',
        'Steel Wire', 'Steel Beams', 'Steel Angles', 'Steel Channels', 'Steel Rods'
    ]
    
    spare_parts = [
        'Blast Furnace Parts', 'Rolling Mill Parts', 'Crane Parts', 'Conveyor Parts',
        'Electrical Components', 'Safety Equipment', 'Quality Testing Equipment'
    ]
    
    inventory_data = []
    for i in range(650):
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        # Mix of different inventory types
        if random.random() < 0.4:  # 40% raw materials
            item_type = 'Raw Material'
            item_name = random.choice(raw_materials)
            unit = 'Tonnes'
            opening_stock = random.uniform(1000, 10000)
            unit_cost = random.uniform(5000, 25000)
        elif random.random() < 0.8:  # 40% finished goods
            item_type = 'Finished Good'
            item_name = random.choice(finished_goods)
            unit = 'Tonnes'
            opening_stock = random.uniform(500, 5000)
            unit_cost = random.uniform(30000, 80000)
        else:  # 20% spare parts
            item_type = 'Spare Part'
            item_name = random.choice(spare_parts)
            unit = 'Units'
            opening_stock = random.uniform(10, 100)
            unit_cost = random.uniform(1000, 50000)
        
        # Inventory movements
        received = random.uniform(0, opening_stock * 0.3)
        issued = random.uniform(0, opening_stock * 0.4)
        closing_stock = opening_stock + received - issued
        
        inventory_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Item_Type': item_type,
            'Item_Name': item_name,
            'Unit': unit,
            'Opening_Stock': round(opening_stock, 2),
            'Received': round(received, 2),
            'Issued': round(issued, 2),
            'Closing_Stock': round(closing_stock, 2),
            'Unit_Cost': round(unit_cost, 2),
            'Total_Value': round(closing_stock * unit_cost, 2),
            'Reorder_Point': round(opening_stock * 0.2, 2),
            'Maximum_Stock': round(opening_stock * 1.5, 2),
            'Stock_Turnover_Ratio': round(random.uniform(2, 8), 2),
            'Days_of_Inventory': round(random.uniform(30, 180), 2),
            'Storage_Location': random.choice(['Warehouse A', 'Warehouse B', 'Production Floor', 'External Storage']),
            'Supplier': f"Supplier_{random.randint(1, 30):03d}",
            'Last_Updated': date.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return pd.DataFrame(inventory_data)

def generate_complete_customer_data():
    """Generate comprehensive customer data with detailed profiles"""
    
    industries = [
        'Construction', 'Automotive', 'Shipbuilding', 'Aerospace', 'Oil & Gas',
        'Infrastructure', 'Manufacturing', 'Energy', 'Railway', 'Defense'
    ]
    
    regions = [
        'North India', 'South India', 'East India', 'West India', 'Central India',
        'International - Asia', 'International - Europe', 'International - Americas'
    ]
    
    customer_data = []
    for i in range(700):
        industry = random.choice(industries)
        region = random.choice(regions)
        
        # Customer profile based on industry
        if industry in ['Construction', 'Infrastructure']:
            preferred_products = ['Steel Beams', 'Steel Bars', 'Steel Plates']
            avg_order_value = random.uniform(200000, 800000)
        elif industry == 'Automotive':
            preferred_products = ['Steel Sheets', 'Steel Coils', 'Steel Wire']
            avg_order_value = random.uniform(150000, 500000)
        elif industry in ['Shipbuilding', 'Aerospace']:
            preferred_products = ['Steel Plates', 'Steel Sheets', 'Specialty Steel']
            avg_order_value = random.uniform(500000, 1500000)
        else:
            preferred_products = random.sample(['Steel Plates', 'Steel Coils', 'Steel Sheets', 'Steel Bars'], 2)
            avg_order_value = random.uniform(100000, 600000)
        
        customer_data.append({
            'Customer_ID': f"CUST_{i+1:04d}",
            'Customer_Name': f"{industry} Company {i+1}",
            'Industry': industry,
            'Region': region,
            'Preferred_Products': ', '.join(preferred_products),
            'Credit_Limit': round(avg_order_value * random.uniform(2, 5), 2),
            'Payment_Terms': random.choice(['Net 30', 'Net 45', 'Net 60', 'Net 90']),
            'Avg_Order_Value': round(avg_order_value, 2),
            'Total_Orders': random.randint(5, 100),
            'Total_Revenue': round(avg_order_value * random.randint(10, 200), 2),
            'Last_Order_Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Customer_Since': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d'),
            'Credit_Rating': random.choice(['A+', 'A', 'B+', 'B', 'C+', 'C']),
            'Contact_Person': f"Contact_{i+1}",
            'Phone': f"+91-{random.randint(7000000000, 9999999999)}",
            'Email': f"customer{i+1}@{industry.lower().replace(' ', '')}.com",
            'Address': f"Address {i+1}, {region}",
            'Tax_ID': f"TAX{i+1:06d}",
            'Bank_Account': f"ACC{i+1:012d}",
            'Payment_Method': random.choice(['Bank Transfer', 'Cheque', 'Letter of Credit', 'Cash']),
            'Discount_Percentage': round(random.uniform(0, 15), 2),
            'Customer_Status': random.choice(['Active', 'Inactive', 'Prospect', 'VIP']),
            'Sales_Representative': f"Sales_Rep_{random.randint(1, 20):02d}",
            'Notes': f"Customer in {industry} industry, {region} region"
        })
    
    return pd.DataFrame(customer_data)

def generate_complete_supplier_data():
    """Generate comprehensive supplier data with detailed profiles"""
    
    supplier_categories = [
        'Raw Material Supplier', 'Equipment Supplier', 'Service Provider',
        'Logistics Provider', 'Technology Provider', 'Maintenance Contractor'
    ]
    
    raw_materials = [
        'Iron Ore', 'Coal', 'Limestone', 'Scrap Metal', 'Alloy Elements',
        'Refractory Materials', 'Lubricants', 'Chemicals', 'Oxygen', 'Nitrogen'
    ]
    
    equipment_types = [
        'Blast Furnace Equipment', 'Rolling Mill Equipment', 'Crane Systems',
        'Conveyor Systems', 'Cooling Systems', 'Compressor Systems',
        'Electrical Systems', 'Safety Equipment', 'Quality Testing Equipment'
    ]
    
    supplier_data = []
    for i in range(600):
        category = random.choice(supplier_categories)
        
        if category == 'Raw Material Supplier':
            supplied_items = random.choice(raw_materials)
            avg_order_value = random.uniform(50000, 300000)
            payment_terms = random.choice(['Net 30', 'Net 45', 'Net 60'])
        elif category == 'Equipment Supplier':
            supplied_items = random.choice(equipment_types)
            avg_order_value = random.uniform(200000, 1000000)
            payment_terms = random.choice(['Net 60', 'Net 90', 'Letter of Credit'])
        else:
            supplied_items = random.choice(['Maintenance Services', 'Logistics Services', 'Technology Services'])
            avg_order_value = random.uniform(25000, 150000)
            payment_terms = random.choice(['Net 30', 'Net 45'])
        
        supplier_data.append({
            'Supplier_ID': f"SUPP_{i+1:04d}",
            'Supplier_Name': f"{category.replace(' ', '')} {i+1}",
            'Category': category,
            'Supplied_Items': supplied_items,
            'Contact_Person': f"Supplier_Contact_{i+1}",
            'Phone': f"+91-{random.randint(7000000000, 9999999999)}",
            'Email': f"supplier{i+1}@{category.lower().replace(' ', '')}.com",
            'Address': f"Supplier Address {i+1}",
            'Tax_ID': f"SUP_TAX{i+1:06d}",
            'Bank_Account': f"SUP_ACC{i+1:012d}",
            'Payment_Terms': payment_terms,
            'Credit_Limit': round(avg_order_value * random.uniform(3, 8), 2),
            'Avg_Order_Value': round(avg_order_value, 2),
            'Total_Orders': random.randint(10, 150),
            'Total_Spend': round(avg_order_value * random.randint(20, 300), 2),
            'Last_Order_Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'Supplier_Since': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d'),
            'Performance_Rating': round(random.uniform(3.0, 5.0), 1),
            'Delivery_Time_Days': random.randint(1, 30),
            'Quality_Rating': round(random.uniform(3.0, 5.0), 1),
            'Price_Rating': round(random.uniform(3.0, 5.0), 1),
            'Payment_Method': random.choice(['Bank Transfer', 'Cheque', 'Letter of Credit']),
            'Discount_Percentage': round(random.uniform(0, 20), 2),
            'Supplier_Status': random.choice(['Active', 'Inactive', 'Approved', 'Under Review']),
            'Contract_End_Date': (datetime.now() + timedelta(days=random.randint(30, 1095))).strftime('%Y-%m-%d'),
            'Insurance_Coverage': random.choice(['Yes', 'No']),
            'Certification': random.choice(['ISO 9001', 'ISO 14001', 'OHSAS 18001', 'None']),
            'Procurement_Manager': f"Proc_Manager_{random.randint(1, 10):02d}",
            'Notes': f"{category} providing {supplied_items}"
        })
    
    return pd.DataFrame(supplier_data)

def generate_complete_operational_data():
    """Generate comprehensive operational data with detailed metrics"""
    
    departments = [
        'Blast Furnace', 'Basic Oxygen Furnace', 'Electric Arc Furnace',
        'Continuous Casting', 'Hot Rolling', 'Cold Rolling', 'Galvanizing',
        'Quality Control', 'Maintenance', 'Utilities', 'Logistics'
    ]
    
    operational_data = []
    for i in range(650):
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        department = random.choice(departments)
        
        # Department-specific metrics
        if 'Furnace' in department:
            efficiency = random.uniform(85, 98)
            energy_consumption = random.uniform(500, 2000)
            temperature = random.uniform(1200, 1800)
            key_metric = 'Temperature_Celsius'
        elif 'Rolling' in department:
            efficiency = random.uniform(90, 99)
            energy_consumption = random.uniform(200, 800)
            speed = random.uniform(50, 200)
            key_metric = 'Speed_Meters_Per_Minute'
        elif 'Quality' in department:
            efficiency = random.uniform(95, 100)
            energy_consumption = random.uniform(50, 200)
            defect_rate = random.uniform(0.1, 2.0)
            key_metric = 'Defect_Rate_Percentage'
        else:
            efficiency = random.uniform(80, 95)
            energy_consumption = random.uniform(100, 500)
            uptime = random.uniform(85, 99)
            key_metric = 'Uptime_Percentage'
        
        operational_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Department': department,
            'Shift': random.choice(['Morning', 'Afternoon', 'Night']),
            'Efficiency_Percentage': round(efficiency, 2),
            'Energy_Consumption_MWh': round(energy_consumption, 2),
            'Labor_Hours': round(random.uniform(8, 24), 2),
            'Production_Output_Tonnes': round(random.uniform(50, 500), 2),
            'Quality_Score': round(random.uniform(85, 99), 2),
            'Downtime_Hours': round(random.uniform(0, 6), 2),
            'Maintenance_Hours': round(random.uniform(0, 4), 2),
            'Safety_Incidents': random.randint(0, 2),
            'Key_Metric_Name': key_metric,
            'Key_Metric_Value': round(random.uniform(50, 2000), 2),
            'Cost_Per_Tonne': round(random.uniform(20000, 60000), 2),
            'Revenue_Per_Tonne': round(random.uniform(40000, 90000), 2),
            'Raw_Material_Usage_Tonnes': round(random.uniform(100, 800), 2),
            'Waste_Generation_Tonnes': round(random.uniform(5, 50), 2),
            'Water_Consumption_Liters': round(random.uniform(1000, 10000), 2),
            'Air_Emissions_Kg': round(random.uniform(100, 1000), 2),
            'Noise_Level_DB': round(random.uniform(70, 95), 2),
            'Temperature_Celsius': round(random.uniform(20, 40), 2),
            'Humidity_Percentage': round(random.uniform(40, 80), 2),
            'Supervisor': f"Supervisor_{random.randint(1, 20):02d}",
            'Team_Size': random.randint(5, 25),
            'Overtime_Hours': round(random.uniform(0, 8), 2),
            'Training_Hours': round(random.uniform(0, 4), 2),
            'Equipment_Status': random.choice(['Normal', 'Maintenance', 'Repair', 'Upgrade']),
            'Notes': f"{department} operations on {date.strftime('%Y-%m-%d')}"
        })
    
    return pd.DataFrame(operational_data)

def generate_complete_financial_data():
    """Generate comprehensive financial data with all cash flow categories"""
    
    financial_data = []
    for i in range(700):
        date = datetime.now() - timedelta(days=random.randint(1, 365))
        
        # Operating Activities
        operating_revenue = random.uniform(5000000, 20000000)
        operating_expenses = operating_revenue * random.uniform(0.6, 0.85)
        net_operating_cash = operating_revenue - operating_expenses
        
        # Investing Activities
        investing_inflows = random.uniform(0, 2000000)  # Asset sales
        investing_outflows = random.uniform(1000000, 8000000)  # Capital expenditure
        net_investing_cash = investing_inflows - investing_outflows
        
        # Financing Activities
        financing_inflows = random.uniform(0, 5000000)  # Borrowing
        financing_outflows = random.uniform(500000, 3000000)  # Repayments
        net_financing_cash = financing_inflows - financing_outflows
        
        # Net cash flow
        net_cash_flow = net_operating_cash + net_investing_cash + net_financing_cash
        
        financial_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Period': f"Period_{i+1}",
            
            # Operating Activities
            'Operating_Revenue': round(operating_revenue, 2),
            'Operating_Expenses': round(operating_expenses, 2),
            'Net_Operating_Cash_Flow': round(net_operating_cash, 2),
            
            # Investing Activities
            'Investing_Inflows': round(investing_inflows, 2),
            'Investing_Outflows': round(investing_outflows, 2),
            'Net_Investing_Cash_Flow': round(net_investing_cash, 2),
            
            # Financing Activities
            'Financing_Inflows': round(financing_inflows, 2),
            'Financing_Outflows': round(financing_outflows, 2),
            'Net_Financing_Cash_Flow': round(net_financing_cash, 2),
            
            # Summary
            'Net_Cash_Flow': round(net_cash_flow, 2),
            'Opening_Cash_Balance': round(random.uniform(10000000, 50000000), 2),
            'Closing_Cash_Balance': round(random.uniform(10000000, 50000000), 2),
            
            # Additional Metrics
            'EBITDA': round(operating_revenue * random.uniform(0.15, 0.35), 2),
            'Depreciation': round(operating_expenses * random.uniform(0.05, 0.15), 2),
            'Interest_Expense': round(random.uniform(100000, 500000), 2),
            'Tax_Expense': round(random.uniform(200000, 1000000), 2),
            'Net_Income': round(random.uniform(500000, 3000000), 2),
            
            # Ratios
            'Operating_Margin': round((net_operating_cash / operating_revenue) * 100, 2),
            'Cash_Flow_Margin': round((net_cash_flow / operating_revenue) * 100, 2),
            'Debt_Service_Coverage': round(random.uniform(1.5, 4.0), 2),
            'Current_Ratio': round(random.uniform(1.2, 3.0), 2),
            'Quick_Ratio': round(random.uniform(0.8, 2.5), 2),
            
            # Working Capital
            'Accounts_Receivable': round(operating_revenue * random.uniform(0.1, 0.3), 2),
            'Accounts_Payable': round(operating_expenses * random.uniform(0.15, 0.35), 2),
            'Inventory': round(random.uniform(5000000, 20000000), 2),
            'Working_Capital': round(random.uniform(2000000, 10000000), 2),
            
            'Notes': f"Financial data for {date.strftime('%Y-%m-%d')}"
        })
    
    return pd.DataFrame(financial_data)

def main():
    """Generate all comprehensive steel plant datasets"""
    
    print("🏭 GENERATING COMPLETE STEEL PLANT DATASETS")
    print("=" * 60)
    
    # Create steel_plant_datasets directory if it doesn't exist
    if not os.path.exists('steel_plant_datasets'):
        os.makedirs('steel_plant_datasets')
    
    # Generate all datasets
    datasets = {
        'steel_plant_sap_data.xlsx': generate_complete_steel_plant_sap_data(),
        'steel_plant_bank_data.xlsx': generate_complete_steel_plant_bank_data(),
        'steel_production_data.xlsx': generate_complete_production_data(),
        'steel_inventory_data.xlsx': generate_complete_inventory_data(),
        'steel_customer_data.xlsx': generate_complete_customer_data(),
        'steel_supplier_data.xlsx': generate_complete_supplier_data(),
        'steel_operational_data.xlsx': generate_complete_operational_data(),
        'steel_financial_data.xlsx': generate_complete_financial_data()
    }
    
    # Save all datasets
    for filename, df in datasets.items():
        filepath = os.path.join('steel_plant_datasets', filename)
        df.to_excel(filepath, index=False)
        print(f"✅ {filename}: {len(df)} records")
        print(f"   Columns: {list(df.columns)}")
    
    # Create comprehensive summary
    summary = {
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_datasets': len(datasets),
        'datasets': {}
    }
    
    for filename, df in datasets.items():
        summary['datasets'][filename] = {
            'records': len(df),
            'columns': list(df.columns),
            'file_size_mb': round(os.path.getsize(os.path.join('steel_plant_datasets', filename)) / (1024*1024), 2)
        }
    
    # Save summary
    summary_file = os.path.join('steel_plant_datasets', 'complete_datasets_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 SUMMARY SAVED: {summary_file}")
    print(f"📁 ALL DATASETS SAVED IN: steel_plant_datasets/")
    print(f"🎯 TOTAL RECORDS: {sum(len(df) for df in datasets.values())}")
    
    # Create detailed catalog
    catalog_content = """# COMPLETE STEEL PLANT DATASETS CATALOG

## Overview
This catalog describes all datasets generated for the steel plant AI cash flow model, including comprehensive data for Operating, Investing, and Financing activities.

## Datasets

### 1. steel_plant_sap_data.xlsx
**Purpose**: SAP AP/AR transactions with all cash flow categories
**Records**: 400+ transactions
**Key Features**:
- Operating Activities (60%): Revenue and expenses
- Investing Activities (25%): Asset sales and capital expenditure
- Financing Activities (15%): Borrowing and repayments
- Steel-specific descriptions and categories

### 2. steel_plant_bank_data.xlsx
**Purpose**: Bank statement data with all cash flow categories
**Records**: 450+ transactions
**Key Features**:
- Operating Activities (65%): Customer payments and supplier payments
- Investing Activities (20%): Equipment sales and purchases
- Financing Activities (15%): Loan disbursements and repayments
- Detailed transaction categorization

### 3. steel_production_data.xlsx
**Purpose**: Production metrics and efficiency data
**Records**: 600+ production records
**Key Features**:
- Multiple production lines (Blast Furnace, Rolling Mills, etc.)
- Efficiency metrics and quality scores
- Energy consumption and labor hours
- Cost and revenue per tonne

### 4. steel_inventory_data.xlsx
**Purpose**: Inventory management for all materials
**Records**: 650+ inventory records
**Key Features**:
- Raw materials, finished goods, and spare parts
- Stock levels and movements
- Reorder points and turnover ratios
- Storage locations and suppliers

### 5. steel_customer_data.xlsx
**Purpose**: Customer profiles and relationship data
**Records**: 700+ customer records
**Key Features**:
- Industry-specific customer profiles
- Credit limits and payment terms
- Order history and revenue data
- Regional and product preferences

### 6. steel_supplier_data.xlsx
**Purpose**: Supplier profiles and performance data
**Records**: 600+ supplier records
**Key Features**:
- Supplier categories and supplied items
- Performance ratings and delivery times
- Contract terms and payment methods
- Quality and price ratings

### 7. steel_operational_data.xlsx
**Purpose**: Department-level operational metrics
**Records**: 650+ operational records
**Key Features**:
- Department-specific efficiency metrics
- Energy consumption and environmental data
- Safety incidents and maintenance hours
- Cost and revenue per tonne by department

### 8. steel_financial_data.xlsx
**Purpose**: Comprehensive financial statements
**Records**: 700+ financial records
**Key Features**:
- Complete cash flow statements (Operating, Investing, Financing)
- Financial ratios and working capital
- EBITDA, depreciation, and tax data
- Balance sheet components

## Cash Flow Categories Covered

### Operating Activities
- Revenue from steel sales
- Raw material purchases
- Operating expenses
- Customer and supplier payments

### Investing Activities
- Capital expenditure (new equipment)
- Asset sales (old machinery, scrap)
- Plant expansion and upgrades
- Technology investments

### Financing Activities
- Bank loans and borrowings
- Loan repayments and interest
- Working capital financing
- Equipment financing

## AI Model Integration
All datasets are designed to work together for comprehensive cash flow analysis:
- SAP/Bank data for transaction-level analysis
- Production data for operational efficiency impact
- Inventory data for working capital optimization
- Customer/Supplier data for relationship insights
- Operational data for cost optimization
- Financial data for comprehensive reporting

## Usage Instructions
1. Upload SAP and Bank data through the Flask interface
2. AI model automatically integrates with all steel plant datasets
3. Enhanced analysis considers all cash flow categories
4. Comprehensive predictions and insights generated
"""
    
    catalog_file = os.path.join('steel_plant_datasets', 'COMPLETE_DATASETS_CATALOG.md')
    with open(catalog_file, 'w') as f:
        f.write(catalog_content)
    
    print(f"📚 CATALOG CREATED: {catalog_file}")
    print("\n🎉 COMPLETE DATASET GENERATION FINISHED!")
    print("All datasets include Operating, Investing, and Financing activities!")

if __name__ == "__main__":
    main() 