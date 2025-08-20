import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import os
from collections import Counter
import time
from sklearn.preprocessing import LabelEncoder

class UniversalVendorExtractor:
    """Universal Vendor Extractor - Completely Rebuilt for Accuracy"""
    
    def __init__(self):
        self.xgb_model = None
        self.vectorizer = None
        self.label_encoder = None
        
    def extract_vendors_intelligently(self, descriptions):
        """Main vendor extraction method - Completely rebuilt"""
        print("🚀 UNIVERSAL VENDOR EXTRACTION - REBUILT FOR ACCURACY")
        print("=" * 60)
        
        if not descriptions or len(descriptions) == 0:
            print("❌ No descriptions provided")
            return []
        
        print(f"📊 Processing {len(descriptions)} transaction descriptions...")
        
        # Step 1: Extract vendors using Ollama with proper prompts
        print("\n🧠 Step 1: Ollama Vendor Extraction...")
        ollama_vendors = self._extract_vendors_with_ollama(descriptions)
        
        # Step 2: Create training data and train XGBoost
        print("\n🤖 Step 2: XGBoost AI Training...")
        xgb_vendors = self._classify_vendors_with_ai(descriptions)
        
        # Step 3: Consolidate results intelligently
        print("\n🧠 Step 3: Intelligent Vendor Consolidation...")
        final_vendors = self._consolidate_vendors_intelligently(ollama_vendors, xgb_vendors, descriptions)
        
        return final_vendors
    
    def _extract_vendors_with_ollama(self, descriptions):
        """Extract vendors using Ollama with proper prompts"""
        vendors = []
        
        try:
            from ollama_simple_integration import simple_ollama, check_ollama_availability
            
            if not check_ollama_availability():
                print("   ⚠️  Ollama not available, using fallback extraction")
                return self._extract_vendors_fallback(descriptions)
            
            print("   🧠 Using Ollama for vendor extraction...")
            
            # Process in smaller batches for better accuracy
            batch_size = 20
            processed_count = 0
            
            for i in range(0, len(descriptions), batch_size):
                batch = descriptions[i:i+batch_size]
                batch_vendors = []
                
                # Create proper prompt for vendor extraction
                batch_prompt = f"""
                You are an expert at identifying REAL VENDOR COMPANIES from transaction descriptions.
                
                CRITICAL: A vendor is ONLY a REAL BUSINESS ENTITY that provides goods/services.
                
                🔍 VENDOR IDENTIFICATION RULES:
                
                ✅ REAL VENDORS (extract these - actual company names):
                - "Equipment Purchase - ABC Equipment Suppliers Ltd" → ABC Equipment Suppliers Ltd
                - "Retention Payment - XYZ Construction Company" → XYZ Construction Company
                - "Raw Material Payment - DEF Steel Corporation" → DEF Steel Corporation
                - "Maintenance Payment - GHI Service Providers" → GHI Service Providers
                - "Supplier Payment - Logistics Provider 28" → Logistics Provider 28
                - "Payment to Equipment Supplier - JKL Machinery Co" → JKL Machinery Co
                
                ❌ NOT VENDORS (ignore these - not company names):
                - "Plant Expansion - New Production Line" → NO_VENDOR (project description)
                - "Infrastructure Development - Warehouse Construction" → NO_VENDOR (project description)
                - "Machinery Purchase - Quality Testing Equipment" → NO_VENDOR (equipment type)
                - "VIP Customer Payment - Construction Company" → NO_VENDOR (customer, not vendor)
                - "Real Estate Developer" → NO_VENDOR (business type, not company name)
                - "Oil & Gas Company" → NO_VENDOR (business type, not company name)
                - "Automotive Manufacturer" → NO_VENDOR (business type, not company name)
                - "Defense Contractor" → NO_VENDOR (business type, not company name)
                - "Railway Department" → NO_VENDOR (government department, not vendor)
                - "Shipbuilding Yard" → NO_VENDOR (facility type, not company name)
                
                ANALYZE THESE DESCRIPTIONS (one per line):
                {chr(10).join([f"{idx+1}. {str(desc)[:150]}" for idx, desc in enumerate(batch) if not pd.isna(desc) and str(desc).strip() != ''])}
                
                OUTPUT FORMAT: For each description, output ONLY the vendor company name if found, or 'NO_VENDOR' if none.
                If vendor found, output the actual company name (e.g., "ABC Construction Co", "XYZ Equipment Ltd").
                If no vendor found, output exactly 'NO_VENDOR'.
                
                Vendor names:"""
                
                try:
                    response = simple_ollama(batch_prompt, "llama2:7b", max_tokens=200)
                    if response:
                        lines = response.strip().split('\n')
                        for idx, (desc, line) in enumerate(zip(batch, lines)):
                            if pd.isna(desc) or str(desc).strip() == '':
                                continue
                            
                            # Clean the response line
                            line_clean = line.strip()
                            
                            # Handle numbered responses
                            if '.' in line_clean:
                                line_clean = line_clean.split('. ', 1)[-1] if '. ' in line_clean else line_clean
                            
                            # Check if it's a real vendor name (not NO_VENDOR)
                            if line_clean.upper() != 'NO_VENDOR' and len(line_clean.strip()) > 2:
                                # Validate it's a real company name
                                vendor_name = self._validate_vendor_name(line_clean.strip())
                                if vendor_name:
                                    vendors.append(vendor_name)
                                    batch_vendors.append(vendor_name)
                                    print(f"   ✅ Vendor found: {vendor_name}")
                                else:
                                    print(f"   ⚠️  Invalid vendor name: {line_clean}")
                            else:
                                print(f"   ⚠️  No vendor: {line_clean}")
                            
                            processed_count += 1
                    
                    print(f"   📊 Processed batch {i//batch_size + 1}: {len(batch_vendors)} vendors found")
                    
                except Exception as e:
                    print(f"   ⚠️  Batch processing failed: {e}")
                    # Fallback to individual processing for this batch
                    for desc in batch:
                        if pd.isna(desc) or str(desc).strip() == '':
                            continue
                        vendor = self._extract_vendor_fallback(desc)
                        if vendor and vendor != "NO_VENDOR":
                            vendors.append(vendor)
                            processed_count += 1
            
            print(f"   🧠 Ollama extracted {len(vendors)} vendor candidates")
            return vendors
            
        except Exception as e:
            print(f"   ⚠️  Ollama integration failed: {e}")
            print("   🔄 FALLBACK: Using rule-based pattern matching for vendor extraction")
            return self._extract_vendors_fallback(descriptions)
    
    def _validate_vendor_name(self, vendor_name):
        """Validate if a name is actually a real vendor company name"""
        if not vendor_name or len(vendor_name.strip()) < 3:
            return None
        
        vendor_clean = vendor_name.strip()
        vendor_lower = vendor_clean.lower()
        
        # Remove common non-company words
        invalid_words = {
            'payment', 'purchase', 'equipment', 'machinery', 'infrastructure', 'development',
            'expansion', 'modernization', 'quality', 'testing', 'warehouse', 'construction',
            'production', 'line', 'capacity', 'increase', 'energy', 'efficiency', 'renovation',
            'plant', 'new', 'advanced', 'technology', 'system', 'digital', 'transformation',
            'project', 'description', 'activity', 'process', 'material', 'raw', 'steel',
            'rolling', 'blast', 'furnace', 'upgrade', 'installation', 'maintenance',
            'service', 'provider', 'supplier', 'vendor', 'company', 'corp', 'ltd', 'inc',
            'real', 'estate', 'developer', 'oil', 'gas', 'automotive', 'manufacturer',
            'defense', 'contractor', 'railway', 'department', 'shipbuilding', 'yard',
            'logistics', 'accounting', 'banking', 'finance', 'investment', 'performance',
            'industrial', 'sale', 'purchase', 'advance', 'retention', 'final', 'milestone',
            'bulk', 'capex', 'bonus', 'bridge', 'loan', 'cleaning', 'gas', 'internet',
            'liquidation', 'legal', 'line', 'credit', 'emi', 'closure', 'marketing',
            'procurement', 'property', 'salary', 'scrap', 'metal', 'security', 'software',
            'telephone', 'training', 'transport', 'utility', 'water', 'supply'
        }
        
        # Check if it's just a generic word
        if vendor_lower in invalid_words:
            return None
        
        # Check if it's a project description
        if any(word in vendor_lower for word in ['project', 'description', 'activity']):
            return None
        
        # Check if it's a material type
        if any(word in vendor_lower for word in ['material', 'equipment', 'machinery', 'steel']):
            return None
        
        # Check if it's a process name
        if any(word in vendor_lower for word in ['production', 'modernization', 'expansion', 'development']):
            return None
        
        # Check if it's a business activity
        if any(word in vendor_lower for word in ['payment', 'purchase', 'sale', 'investment', 'loan']):
            return None
        
        # Check if it's a generic business term
        if any(word in vendor_lower for word in ['real', 'estate', 'oil', 'gas', 'automotive', 'defense']):
            return None
        
        # Check if it's a department or authority
        if any(word in vendor_lower for word in ['department', 'ministry', 'authority', 'railway']):
            return None
        
        # Check if it's a single generic word (likely not a company)
        if len(vendor_clean.split()) == 1 and vendor_lower in ['warehouse', 'production', 'project', 'real', 'oil', 'gas', 'automotive', 'defense', 'railway', 'shipbuilding', 'logistics', 'accounting', 'banking', 'finance', 'investment', 'performance', 'industrial', 'sale', 'purchase', 'advance', 'retention', 'final', 'milestone', 'bulk', 'capex', 'bonus', 'bridge', 'loan', 'cleaning', 'gas', 'internet', 'liquidation', 'legal', 'line', 'credit', 'emi', 'closure', 'marketing', 'procurement', 'property', 'salary', 'scrap', 'metal', 'security', 'software', 'telephone', 'training', 'transport', 'utility', 'water', 'supply']:
            return None
        
        # If it passes all checks, it might be a real vendor
        return vendor_clean
    
    def _extract_vendor_fallback(self, description):
        """Fallback vendor extraction using intelligent pattern matching"""
        desc = str(description).lower()
        
        # Look for actual company names in the description
        company_patterns = [
            # Look for "Company Name - Service/Product" pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:company|corp|corporation|ltd|limited|inc|incorporated)',
            # Look for "Service Provider" pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:provider|supplier|vendor|contractor)',
            # Look for "Engineering Firm" pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:firm|agency|organization)',
            # Look for "Department" pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:department|ministry|authority)',
            # Look for "Manufacturer" pattern
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:manufacturer|producer|maker)'
        ]
        
        import re
        for pattern in company_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                vendor = match.group(1).strip()
                if self._validate_vendor_name(vendor):
                    return vendor
        
        # Look for specific vendor patterns that are likely real companies
        specific_vendor_patterns = [
            r'logistics\s+provider\s+(\d+)',  # Logistics Provider 28
            r'service\s+provider\s+(\d+)',    # Service Provider 47
            r'equipment\s+supplier',           # Equipment Supplier
            r'raw\s+material\s+supplier\s+(\d+)',  # Raw Material Supplier 30
            r'coal\s+supplier',                # Coal Supplier
            r'limestone\s+supplier',           # Limestone Supplier
            r'alloy\s+supplier',               # Alloy Supplier
            r'steel\s+supplier',               # Steel Supplier
            r'logistics\s+provider',           # Logistics Provider (without number)
            r'service\s+provider',             # Service Provider (without number)
        ]
        
        for pattern in specific_vendor_patterns:
            match = re.search(pattern, desc, re.IGNORECASE)
            if match:
                if 'logistics provider' in desc:
                    number = match.group(1) if match.groups() else ''
                    return f"Logistics Provider {number}".strip()
                elif 'service provider' in desc:
                    number = match.group(1) if match.groups() else ''
                    return f"Service Provider {number}".strip()
                elif 'equipment supplier' in desc:
                    return "Equipment Supplier"
                elif 'raw material supplier' in desc:
                    number = match.group(1) if match.groups() else ''
                    return f"Raw Material Supplier {number}".strip()
                elif 'coal supplier' in desc:
                    return "Coal Supplier"
                elif 'limestone supplier' in desc:
                    return "Limestone Supplier"
                elif 'alloy supplier' in desc:
                    return "Alloy Supplier"
                elif 'steel supplier' in desc:
                    return "Steel Supplier"
        
        # Look for capitalized words that might be company names
        words = description.split()
        for i, word in enumerate(words):
            if (word[0].isupper() and len(word) > 2 and 
                not self._is_generic_word(word.lower()) and
                not any(term in word.lower() for term in ['payment', 'purchase', 'invoice', 'project', 'description'])):
                
                # Check if next word is also capitalized (likely company name)
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    potential_vendor = f"{word} {words[i + 1]}"
                    if self._validate_vendor_name(potential_vendor):
                        return potential_vendor
                
                # Single word vendor - only if it's a real company name
                if self._validate_vendor_name(word):
                    return word
        
        return "NO_VENDOR"
    
    def _is_generic_word(self, word):
        """Check if word is generic (not a real vendor)"""
        generic_words = {
            'payment', 'invoice', 'contract', 'order', 'delivery', 'service',
            'product', 'item', 'goods', 'materials', 'equipment', 'supplies',
            'company', 'corporation', 'limited', 'inc', 'ltd', 'corp',
            'department', 'division', 'section', 'unit', 'group', 'team',
            'firm', 'agency', 'organization', 'association', 'foundation',
            'plant', 'expansion', 'infrastructure', 'development', 'machinery',
            'purchase', 'advance', 'retention', 'final', 'export', 'import',
            'vip', 'customer', 'milestone', 'bulk', 'capex', 'bonus',
            'bridge', 'loan', 'cleaning', 'gas', 'internet', 'investment',
            'liquidation', 'legal', 'line', 'credit', 'emi', 'closure',
            'logistics', 'maintenance', 'marketing', 'new', 'penalty',
            'procurement', 'property', 'renovation', 'salary', 'scrap',
            'metal', 'security', 'software', 'technology', 'telephone',
            'training', 'transport', 'utility', 'water', 'supply',
            'project', 'description', 'activity', 'process', 'modernization',
            'quality', 'testing', 'warehouse', 'construction', 'production',
            'line', 'capacity', 'increase', 'energy', 'efficiency', 'renovation',
            'plant', 'new', 'advanced', 'technology', 'system', 'digital',
            'transformation', 'material', 'raw', 'steel', 'rolling', 'blast',
            'furnace', 'upgrade', 'installation', 'service', 'provider',
            'supplier', 'vendor', 'manufacturer', 'producer', 'maker',
            'real', 'estate', 'developer', 'oil', 'gas', 'automotive',
            'defense', 'contractor', 'railway', 'shipbuilding', 'yard',
            'accounting', 'banking', 'finance', 'investment', 'performance',
            'industrial', 'sale', 'purchase', 'advance', 'retention', 'final',
            'milestone', 'bulk', 'capex', 'bonus', 'bridge', 'loan',
            'cleaning', 'gas', 'internet', 'liquidation', 'legal', 'line',
            'credit', 'emi', 'closure', 'marketing', 'procurement', 'property',
            'salary', 'scrap', 'metal', 'security', 'software', 'telephone',
            'training', 'transport', 'utility', 'water', 'supply'
        }
        return word.lower() in generic_words
    
    def _classify_vendors_with_ai(self, descriptions):
        """AI-powered vendor classification using XGBoost"""
        if len(descriptions) < 10:
            print("   ⚠️  Insufficient data for AI model training")
            return []
        
        try:
            # Create training data using Ollama
            training_data = self._create_intelligent_training_data(descriptions)
            
            if len(training_data) < 20:
                print("   ⚠️  Insufficient training data for AI model")
                return []
            
            # Train XGBoost model
            self._train_xgboost_model(training_data)
            
            # Make predictions
            return self._predict_vendors_ai(descriptions)
            
        except Exception as e:
            print(f"   ⚠️  AI classification failed: {e}")
            return []
    
    def _create_intelligent_training_data(self, descriptions):
        """Create intelligent training data using Ollama"""
        training_data = []
        
        try:
            from ollama_simple_integration import simple_ollama, check_ollama_availability
            
            if not check_ollama_availability():
                print("   ⚠️  Ollama not available, using fallback training data")
                return self._create_fallback_training_data(descriptions)
            
            print("   🧠 Creating training data using Ollama...")
            
            # Sample descriptions for training
            sample_descriptions = descriptions[:30]  # Reduced for better quality
            
            # Create batch prompt for training data
            batch_prompt = f"""
            You are an expert at identifying REAL VENDOR COMPANIES from transaction descriptions.
            
            CRITICAL: A vendor is ONLY a REAL BUSINESS ENTITY that provides goods/services.
            
            🔍 VENDOR IDENTIFICATION RULES:
            
            ✅ REAL VENDORS (extract these - actual company names):
            - "Equipment Purchase - ABC Equipment Suppliers Ltd" → ABC Equipment Suppliers Ltd
            - "Retention Payment - XYZ Construction Company" → XYZ Construction Company
            - "Raw Material Payment - DEF Steel Corporation" → DEF Steel Corporation
            - "Maintenance Payment - GHI Service Providers" → GHI Service Providers
            - "Supplier Payment - Logistics Provider 28" → Logistics Provider 28
            - "Payment to Equipment Supplier - JKL Machinery Co" → JKL Machinery Co
            
            ❌ NOT VENDORS (ignore these - not company names):
            - "Plant Expansion - New Production Line" → NO_VENDOR (project description)
            - "Infrastructure Development - Warehouse Construction" → NO_VENDOR (project description)
            - "Machinery Purchase - Quality Testing Equipment" → NO_VENDOR (equipment type)
            - "VIP Customer Payment - Construction Company" → NO_VENDOR (customer, not vendor)
            - "Real Estate Developer" → NO_VENDOR (business type, not company name)
            - "Oil & Gas Company" → NO_VENDOR (business type, not company name)
            - "Automotive Manufacturer" → NO_VENDOR (business type, not company name)
            - "Defense Contractor" → NO_VENDOR (business type, not company name)
            - "Railway Department" → NO_VENDOR (government department, not vendor)
            - "Shipbuilding Yard" → NO_VENDOR (facility type, not company name)
            
            ANALYZE THESE DESCRIPTIONS (one per line):
            {chr(10).join([f"{idx+1}. {str(desc)[:150]}" for idx, desc in enumerate(sample_descriptions) if not pd.isna(desc) and str(desc).strip() != ''])}
            
            OUTPUT FORMAT: For each description, output ONLY the vendor company name if found, or 'NO_VENDOR' if none.
            If vendor found, output the actual company name (e.g., "ABC Construction Co", "XYZ Equipment Ltd").
            If no vendor found, output exactly 'NO_VENDOR'.
            
            Vendor names:"""
            
            try:
                response = simple_ollama(batch_prompt, "llama2:7b", max_tokens=200)
                if response:
                    lines = response.strip().split('\n')
                    for idx, (desc, line) in enumerate(zip(sample_descriptions, lines)):
                        if pd.isna(desc) or str(desc).strip() == '':
                            continue
                        
                        # Clean the response line
                        line_clean = line.strip()
                        
                        # Handle numbered responses
                        if '.' in line_clean:
                            line_clean = line_clean.split('. ', 1)[-1] if '. ' in line_clean else line_clean
                        
                        # Check if it's a real vendor name
                        if line_clean.upper() != 'NO_VENDOR' and len(line_clean.strip()) > 2:
                            vendor_name = self._validate_vendor_name(line_clean.strip())
                            if vendor_name:
                                training_data.append({
                                    'description': str(desc),
                                    'vendor': vendor_name,
                                    'confidence': 'high'
                                })
                                print(f"   ✅ Training example {idx+1}: VENDOR → {vendor_name}")
                            else:
                                print(f"   ⚠️  Training example {idx+1}: Invalid vendor name")
                        else:
                            print(f"   ⚠️  Training example {idx+1}: NO_VENDOR (skipped)")
                
                print(f"   🎯 Created {len(training_data)} training examples using Ollama")
                return training_data
                
            except Exception as e:
                print(f"   ⚠️  Ollama training data creation failed: {e}")
                return self._create_fallback_training_data(descriptions)
            
        except Exception as e:
            print(f"   ⚠️  Training data creation failed: {e}")
            return self._create_fallback_training_data(descriptions)
    
    def _create_fallback_training_data(self, descriptions):
        """Create fallback training data using intelligent pattern matching"""
        training_data = []
        
        print("   🔄 Creating fallback training data using intelligent pattern matching...")
        
        # Sample descriptions for training
        sample_descriptions = descriptions[:50]
        
        for desc in sample_descriptions:
            if pd.isna(desc) or str(desc).strip() == '':
                continue
            
            # Extract vendor using intelligent fallback
            vendor = self._extract_vendor_fallback(desc)
            if vendor and vendor != "NO_VENDOR":
                training_data.append({
                    'description': str(desc),
                    'vendor': vendor,
                    'confidence': 'medium'
                })
        
        print(f"   🎯 Created {len(training_data)} fallback training examples")
        return training_data
    
    def _train_xgboost_model(self, training_data):
        """Train XGBoost model for vendor classification"""
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        descriptions = [item['description'] for item in training_data]
        vendors = [item['vendor'] for item in training_data]
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        vendor_labels = self.label_encoder.fit_transform(vendors)
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        X_features = self.vectorizer.fit_transform(descriptions)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_features, vendor_labels)
        
        print(f"   ✅ XGBoost model trained successfully")
    
    def _predict_vendors_ai(self, descriptions):
        """Predict vendors using trained AI model"""
        if not self.xgb_model or not self.vectorizer:
            return []
        
        # Transform descriptions
        features = self.vectorizer.transform(descriptions)
        
        # Get predictions
        predictions = self.xgb_model.predict(features)
        confidence_scores = self.xgb_model.predict_proba(features).max(axis=1)
        
        # Decode predictions
        predicted_vendors = self.label_encoder.inverse_transform(predictions)
        
        # Filter high-confidence predictions
        high_confidence_vendors = []
        for vendor, conf in zip(predicted_vendors, confidence_scores):
            if conf > 0.6:  # High confidence threshold
                high_confidence_vendors.append(vendor)
        
        print(f"   🤖 XGBoost predicted {len(high_confidence_vendors)} high-confidence vendors")
        return high_confidence_vendors
    
    def _consolidate_vendors_intelligently(self, ollama_vendors, xgb_vendors, descriptions):
        """Intelligently consolidate vendor results"""
        print("\n🧠 Step 3: Intelligent Vendor Consolidation...")
        
        # Combine all vendors
        all_vendors = ollama_vendors + xgb_vendors
        
        if not all_vendors:
            print("   ⚠️  No vendors found by any method")
            return []
        
        # Count vendor occurrences
        vendor_counts = Counter(all_vendors)
        
        # Filter out generic/invalid vendors with stricter validation
        final_vendors = []
        filtered_vendors = []
        for vendor, count in vendor_counts.items():
            # Apply stricter validation for final list
            if self._validate_vendor_name(vendor) and count >= 1:
                # Additional check: ensure it's not a generic business term
                vendor_lower = vendor.lower()
                if not any(generic in vendor_lower for generic in [
                    'production', 'project', 'automotive', 'real', 'estate', 'oil', 'gas',
                    'defense', 'railway', 'shipbuilding', 'logistics', 'accounting', 'banking',
                    'finance', 'investment', 'performance', 'industrial', 'sale', 'purchase',
                    'advance', 'retention', 'final', 'milestone', 'bulk', 'capex', 'bonus',
                    'bridge', 'loan', 'cleaning', 'gas', 'internet', 'liquidation', 'legal',
                    'line', 'credit', 'emi', 'closure', 'marketing', 'procurement', 'property',
                    'salary', 'scrap', 'metal', 'security', 'software', 'telephone', 'training',
                    'transport', 'utility', 'water', 'supply', 'warehouse', 'construction',
                    'modernization', 'quality', 'testing', 'rolling', 'blast', 'furnace',
                    'upgrade', 'installation', 'maintenance', 'service', 'provider', 'supplier',
                    'vendor', 'manufacturer', 'producer', 'maker'
                ]):
                    final_vendors.append(vendor)
                else:
                    filtered_vendors.append(vendor)
                    print(f"   ⚠️  Filtered out generic vendor: {vendor}")
            else:
                filtered_vendors.append(vendor)
                print(f"   ⚠️  Filtered out invalid vendor: {vendor}")
        
        # Show filtering summary
        if filtered_vendors:
            print(f"\n🚫 FILTERED OUT VENDORS ({len(filtered_vendors)}):")
            for vendor in filtered_vendors:
                count = vendor_counts[vendor]
                print(f"   • {vendor} ({count} transactions) - Generic/Invalid")
        
        # Sort by frequency
        final_vendors.sort(key=lambda x: vendor_counts[x], reverse=True)
        
        # Display results
        print(f"\n📊 FINAL VENDOR EXTRACTION RESULTS:")
        print(f"   🎯 Total Transactions: {len(descriptions)}")
        print(f"   🏢 Unique Vendors: {len(final_vendors)}")
        print(f"   🤖 XGBoost AI: {len(xgb_vendors)}")
        print(f"   🧠 Ollama AI: {len(ollama_vendors)}")
        
        if final_vendors:
            print(f"\n🏢 REAL VENDORS IDENTIFIED:")
            for vendor in final_vendors:
                count = vendor_counts[vendor]
                print(f"   • {vendor} ({count} transactions)")
        
        return final_vendors
    
    def _extract_vendors_fallback(self, descriptions):
        """Fallback vendor extraction using intelligent pattern matching"""
        vendors = []
        
        print("   🔄 Using intelligent fallback vendor extraction...")
        
        for desc in descriptions:
            if pd.isna(desc) or str(desc).strip() == '':
                continue
            
            vendor = self._extract_vendor_fallback(desc)
            if vendor and vendor != "NO_VENDOR":
                vendors.append(vendor)
        
        print(f"   🔄 Fallback extracted {len(vendors)} vendor candidates")
        return vendors

def analyze_real_vendors_fast(df):
    """Fast vendor analysis - Completely rebuilt for accuracy"""
    if df is None or df.empty:
        print("❌ No data provided for vendor analysis")
        return []
    
    print("🚀 FAST VENDOR ANALYSIS - REBUILT FOR ACCURACY")
    print("=" * 60)
    
    # Find description column
    description_col = None
    for col in df.columns:
        if 'description' in col.lower() or 'desc' in col.lower() or 'narration' in col.lower():
            description_col = col
            break
    
    if description_col is None:
        print("❌ No description column found")
        return []
    
    print(f"✅ Using uploaded DataFrame with {len(df)} transactions")
    print(f"✅ Using description column: '{description_col}'")
    
    # Extract descriptions
    descriptions = df[description_col].dropna().tolist()
    
    if not descriptions:
        print("❌ No descriptions found in the data")
        return []
    
    # Create vendor extractor and process
    extractor = UniversalVendorExtractor()
    start_time = time.time()
    
    vendors = extractor.extract_vendors_intelligently(descriptions)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Performance summary
    print(f"\n⚡ PERFORMANCE SUMMARY:")
    print(f"   🚀 Total Processing Time: {processing_time:.2f} seconds")
    print(f"   📊 Transactions Processed: {len(descriptions)}")
    print(f"   🎯 Vendors Identified: {len(vendors)}")
    print(f"   ⚡ Speed: {len(descriptions)/processing_time:.1f} transactions/second")
    
    if vendors:
        print(f"\n✅ Fast vendor analysis complete!")
        print(f"🎯 Found {len(vendors)} real vendors")
    else:
        print(f"\n⚠️  No vendors found")
    
    return vendors

def main():
    """Main function - for testing only"""
    print("🚀 UNIVERSAL VENDOR EXTRACTION SYSTEM - REBUILT FOR ACCURACY")
    print("=" * 60)
    print("⚠️  This file should be run from the main application")
    print("📁 Use the analyze_real_vendors_fast() function instead")
    print("=" * 60)

if __name__ == "__main__":
    main() 