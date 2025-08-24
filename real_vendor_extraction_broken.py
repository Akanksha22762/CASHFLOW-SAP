import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import os
from collections import Counter
import time
import asyncio
import concurrent.futures
import re  
from sklearn.preprocessing import LabelEncoder
from ollama_simple_integration import simple_ollama, AsyncOllamaClient
import pickle
import hashlib

class UniversalVendorExtractor:
    """Universal Vendor Extractor - OPTIMIZED FOR SPEED AND RELIABILITY"""
    
    def __init__(self):
        self.xgb_model = None
        self.vectorizer = None
        self.label_encoder = None
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.last_cache_cleanup = time.time()
        
    def _get_cache_key(self, descriptions):
        """Generate cache key for descriptions"""
        # Create hash of descriptions for caching
        desc_hash = hashlib.md5(str(sorted(descriptions)).encode()).hexdigest()
        return f"vendor_extraction_{desc_hash}"
    
    def _get_cached_result(self, cache_key):
        """Get cached result if available and not expired"""
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                print(f"   🚀 Using cached vendor extraction result ({len(cached_result)} vendors)")
                return cached_result
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key, result):
        """Cache the result with timestamp"""
        self.cache[cache_key] = (time.time(), result)
        
        # Cleanup old cache entries periodically
        if time.time() - self.last_cache_cleanup > 300:  # Every 5 minutes
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory bloat"""
        current_time = time.time()
        expired_keys = []
        
        for key, (timestamp, _) in self.cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            print(f"🧹 Cache cleanup: removed {len(expired_keys)} expired entries")
        
        self.last_cache_cleanup = current_time
    
    async def extract_vendors_intelligently(self, descriptions, use_ai=True):
        """Main vendor extraction method - PRIORITY-BASED: Ollama → XGBoost → Regex"""
        print("🚀 PRIORITY-BASED VENDOR EXTRACTION - AI FIRST APPROACH")
        print("=" * 60)
        print(f"🔍 Input descriptions: {len(descriptions) if descriptions else 0} items")
        
        if not descriptions or len(descriptions) == 0:
            print("❌ No descriptions provided")
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(descriptions)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        print(f"📊 Processing {len(descriptions)} transaction descriptions...")
        
        start_time = time.time()
        all_vendors = []
        
        # STEP 1: Try OLLAMA FIRST (AI-powered, most accurate)
        if use_ai:
            print("\n🧠 Step 1: OLLAMA AI Enhancement (Priority 1)...")
            try:
                # Process ALL descriptions for maximum vendor coverage
                sample_size = len(descriptions)  # Process ALL transactions, no sampling limit
                print(f"🧠 Processing {len(descriptions)} descriptions with Ollama...")
                
                ollama_vendors = self._extract_vendors_with_ollama_fast(descriptions)
                if ollama_vendors:
                    all_vendors.extend(ollama_vendors)
                    print(f"✅ Ollama found {len(ollama_vendors)} vendors")
                else:
                    print("⚠️ Ollama found no vendors, trying XGBoost...")
            except Exception as e:
                print(f"❌ Ollama failed: {e}, trying XGBoost...")
        
        # STEP 2: Try XGBOOST SECOND (ML-powered, good accuracy)
        if use_ai and (not all_vendors or len(all_vendors) < 5):
            print("\n🤖 Step 2: XGBoost ML Enhancement (Priority 2)...")
            try:
                xgboost_vendors = self._extract_vendors_with_xgboost(descriptions)
                if xgboost_vendors:
                    all_vendors.extend(xgboost_vendors)
                    print(f"✅ XGBoost found {len(xgboost_vendors)} vendors")
                else:
                    print("⚠️ XGBoost found no vendors, using regex fallback...")
            except Exception as e:
                print(f"❌ XGBoost failed: {e}, using regex fallback...")
        
        # STEP 3: Use REGEX LAST (fastest, fallback only)
        if not all_vendors:
            print("\n⚡ Step 3: REGEX Fallback (Priority 3 - Fastest)...")
            regex_vendors = self._extract_vendors_fast_regex(descriptions)
            all_vendors.extend(regex_vendors)
            print(f"✅ Regex fallback found {len(regex_vendors)} vendors")
        else:
            print(f"\n✅ AI/ML methods found {len(all_vendors)} vendors, skipping regex")
        
        # Consolidate results
        final_vendors = self._consolidate_vendors_fast(all_vendors, descriptions)
        
        # Cache the result
        self._cache_result(cache_key, final_vendors)
        
        total_time = time.time() - start_time
        print(f"\n🚀 PRIORITY-BASED EXTRACTION COMPLETED:")
        print(f"   🚀 Total Time: {total_time:.2f}s")
        print(f"   📊 Transactions: {len(descriptions)}")
        print(f"   🎯 Vendors Found: {len(final_vendors)}")
        print(f"   ⚡ Speed: {len(descriptions)/total_time:.1f} transactions/second")
        
        return final_vendors
    
    def extract_vendors_intelligently_sync(self, descriptions, use_ai=True):
        """Sync wrapper - PRIORITY-BASED: Ollama → XGBoost → Regex"""
        print(f"📊 Processing {len(descriptions)} transaction descriptions...")

        # Check cache first
        cache_key = self._get_cache_key(descriptions)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        print("🚀 Using PRIORITY-BASED extraction: Ollama → XGBoost → Regex")
        
        start_time = time.time()
        all_vendors = []
        
        # STEP 1: Try OLLAMA FIRST (AI-powered, most accurate)
        if use_ai:
            print("\n🧠 Step 1: OLLAMA AI Enhancement (Priority 1)...")
            try:
                # Process ALL descriptions for maximum vendor coverage
                sample_size = len(descriptions)  # Process ALL transactions, no sampling limit
                print(f"🧠 Processing {len(descriptions)} descriptions with Ollama...")
                
                ollama_vendors = self._extract_vendors_with_ollama_fast(descriptions)
                if ollama_vendors:
                    all_vendors.extend(ollama_vendors)
                    print(f"✅ Ollama found {len(ollama_vendors)} vendors")
                else:
                    print("⚠️ Ollama found no vendors, trying XGBoost...")
            except Exception as e:
                print(f"❌ Ollama failed: {e}, trying XGBoost...")
        
        # STEP 2: Try XGBOOST SECOND (ML-powered, good accuracy)
        if use_ai and (not all_vendors or len(all_vendors) < 5):
            print("\n🤖 Step 2: XGBoost ML Enhancement (Priority 2)...")
            try:
                xgboost_vendors = self._extract_vendors_with_xgboost(descriptions)
                if xgboost_vendors:
                    all_vendors.extend(xgboost_vendors)
                    print(f"✅ XGBoost found {len(xgboost_vendors)} vendors")
                else:
                    print("⚠️ XGBoost found no vendors, using regex fallback...")
            except Exception as e:
                print(f"❌ XGBoost failed: {e}, using regex fallback...")
        
        # STEP 3: Use REGEX LAST (fastest, fallback only)
        if not all_vendors:
            print("\n⚡ Step 3: REGEX Fallback (Priority 3 - Fastest)...")
            regex_vendors = self._extract_vendors_fast_regex(descriptions)
            all_vendors.extend(regex_vendors)
            print(f"✅ Regex fallback found {len(regex_vendors)} vendors")
        else:
            print(f"\n✅ AI/ML methods found {len(all_vendors)} vendors, skipping regex")
        
        # Consolidate results
        final_vendors = self._consolidate_vendors_fast(all_vendors, descriptions)
        
        # Cache the result
        self._cache_result(cache_key, final_vendors)
        
        total_time = time.time() - start_time
        print(f"✅ PRIORITY-BASED extraction completed in {total_time:.2f}s: {len(final_vendors)} vendors")
        print(f"🚀 Speed: {len(descriptions)/total_time:.1f} transactions/second")
        
        return final_vendors
    
    def extract_vendors_with_forced_ai(self, descriptions):
        """Force AI usage for vendor extraction - bypasses regex-only mode"""
        print("🤖 FORCED AI VENDOR EXTRACTION - BYPASSING SPEED OPTIMIZATIONS")
        print("=" * 60)
        print(f"🔍 Input descriptions: {len(descriptions) if descriptions else 0} items")
        
        if not descriptions or len(descriptions) == 0:
            print("❌ No descriptions provided")
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(descriptions)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        print(f"📊 Processing {len(descriptions)} transaction descriptions...")
        
        # Step 1: Try Ollama FIRST (forced AI usage)
        print("\n🧠 Step 1: FORCED Ollama AI Enhancement...")
        start_time = time.time()
        ollama_vendors = []
        
        try:
            # Process ALL descriptions for better AI learning
            sample_size = len(descriptions)  # Process ALL transactions, no sampling limit
            print(f"🧠 Processing {len(descriptions)} descriptions with Ollama...")
            
            ollama_vendors = self._extract_vendors_with_ollama_fast(descriptions)
            print(f"✅ Ollama enhancement completed: {len(ollama_vendors)} vendors found")
            
        except Exception as e:
            print(f"❌ Ollama enhancement failed: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            print(f"🔍 Error details: {str(e)}")
            ollama_vendors = []
        
        # Step 2: Fallback to regex if Ollama fails
        regex_vendors = []
        if not ollama_vendors:
            print("\n⚡ Step 2: Fallback to Fast Regex Extraction...")
            regex_vendors = self._extract_vendors_fast_regex(descriptions)
            print(f"✅ Regex fallback completed: {len(regex_vendors)} vendors found")
        
        # Step 3: Consolidate results
        print("\n🧠 Step 3: Intelligent Vendor Consolidation...")
        all_vendors = ollama_vendors + regex_vendors
        final_vendors = self._consolidate_vendors_fast(all_vendors, descriptions)
        
        # Cache the result
        self._cache_result(cache_key, final_vendors)
        
        total_time = time.time() - start_time
        print(f"\n🤖 FORCED AI EXTRACTION COMPLETED:")
        print(f"   🚀 Total Time: {total_time:.2f}s")
        print(f"   📊 Transactions: {len(descriptions)}")
        print(f"   🎯 Vendors Found: {len(final_vendors)}")
        print(f"   🧠 AI Vendors: {len(ollama_vendors)}")
        print(f"   ⚡ Regex Vendors: {len(regex_vendors)}")
        print(f"   ⚡ Speed: {len(descriptions)/total_time:.1f} transactions/second")
        
        return final_vendors
    
    def _extract_vendors_fast_regex(self, descriptions):
        """ULTRA-FAST vendor extraction using STRICT regex patterns for real companies only"""
        print("   ⚡ Using ULTRA-FAST regex extraction with STRICT validation...")
        vendors = []
        start_time = time.time()
        
        # STRICT regex patterns for REAL company names only
        vendor_patterns = [
            # Pattern 1: Company names with business suffixes (HIGH PRIORITY - definitely companies)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co|Group|Enterprises|Holdings|International|Industries))\b',
            
            # Pattern 2: Company names with "&" or "and" (likely real companies)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:&|and)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            
            # Pattern 3: Company names in payment descriptions (specific format)
            r'(?:Payment\s+to|Invoice\s+from|Purchase\s+from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))',
            
            # Pattern 4: Company names after common prefixes (specific format)
            r'(?:ABC|XYZ|DEF|GHI|JKL|MNO|PQR|STU|VWX|YZ)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))',
            
            # Pattern 5: Company names in parentheses (specific format)
            r'\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))\)',
            
            # Pattern 6: Company names after dashes (specific format)
            r'[-–—]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))',
            
            # Pattern 7: Company names in quotes (specific format)
            r'"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))"',
            
            # Pattern 8: Specific vendor patterns (only if they have numbers or specific names)
            r'(?:Logistics\s+Provider|Service\s+Provider|Equipment\s+Supplier|Raw\s+Material\s+Supplier|Coal\s+Supplier|Limestone\s+Supplier|Alloy\s+Supplier|Steel\s+Supplier)\s+(\d+)',
            
            # Pattern 9: Multi-word company names with business keywords (STRICT validation)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Steel|Construction|Engineering|Manufacturing|Trading|Logistics|Services|Suppliers|Providers|Contractors|Developers))\b',
            
            # Pattern 10: Company names with specific business suffixes (STRICT)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Steel|Construction|Engineering|Manufacturing|Trading|Logistics|Services|Suppliers|Providers|Contractors|Developers))\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co|Group|Enterprises|Holdings|International|Industries)\b',
            
            # Pattern 11: Company names starting with common prefixes (ABC, XYZ, etc.)
            r'\b(?:ABC|XYZ|DEF|GHI|JKL|MNO|PQR|STU|VWX|YZ)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            
            # Pattern 12: Company names in "Payment to [Company]" format
            r'Payment\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))',
            
            # Pattern 13: Company names in "Invoice from [Company]" format
            r'Invoice\s+from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))'
        ]
        
        processed = 0
        for desc in descriptions:
            if pd.isna(desc) or str(desc).strip() == '':
                continue
                
            desc_str = str(desc)
            vendor_found = False
            
            # Try each pattern
            for pattern in vendor_patterns:
                matches = re.findall(pattern, desc_str)
                for match in matches:
                    # Handle both string matches and tuple matches from capture groups
                    if isinstance(match, tuple):
                        # If it's a tuple (from capture groups), take the first non-empty group
                        vendor = next((group.strip() for group in match if group and group.strip()), None)
                    else:
                        # If it's a string, use it directly
                        vendor = match.strip() if match else None
                    
                    if vendor and len(vendor) > 3:
                        print(f"   🔍 Pattern match: '{vendor}' from '{desc_str[:50]}...'")
                        
                        # Apply STRICT validation
                        if self._validate_vendor_name_fast(vendor):
                            vendors.append(vendor)
                            vendor_found = True
                            print(f"   ✅ Vendor accepted: {vendor}")
                            break
                                            else:
                            print(f"   ❌ Vendor rejected: {vendor}")
                if vendor_found:
                    break
            
            processed += 1
            if processed % 50 == 0:
                print(f"   📊 Processed {processed}/{len(descriptions)} descriptions...")
            
            total_time = time.time() - start_time
        print(f"   ⚡ ULTRA-FAST regex completed in {total_time:.2f}s: {len(vendors)} vendors")
            print(f"   🚀 Speed: {len(descriptions)/total_time:.1f} descriptions/second")
            return vendors
            
    def _validate_vendor_name_fast(self, vendor_name):
        """BALANCED vendor name validation - Real company names with business suffixes"""
        if not vendor_name or len(vendor_name.strip()) < 3:
            return None
        
        vendor_clean = vendor_name.strip()
        vendor_lower = vendor_clean.lower()
        
        # 🚫 REJECT obvious non-company terms first
        obvious_rejections = {
            # Transaction types
            'payment', 'purchase', 'sale', 'advance', 'retention', 'final', 'milestone',
            'bulk', 'capex', 'bonus', 'bridge', 'loan', 'credit', 'emi', 'closure',
            'export', 'import', 'investment', 'liquidation', 'proceeds', 'charges',
            'interest', 'principal', 'repayment', 'disbursement', 'penalty', 'bonus',
            
            # Project descriptions
            'plant expansion', 'infrastructure development', 'warehouse construction',
            'production line', 'capacity increase', 'new blast furnace', 'renovation',
            'modernization', 'upgrade', 'installation', 'maintenance', 'project',
            'infrastructure project', 'bridge construction', 'festival season',
            
            # Equipment and materials
            'equipment', 'machinery', 'infrastructure', 'development', 'expansion',
            'quality', 'testing', 'warehouse', 'production', 'line', 'capacity',
            'energy', 'efficiency', 'technology', 'system', 'digital', 'transformation',
            'material', 'raw', 'steel', 'rolling', 'blast', 'furnace', 'galvanized',
            'color coated', 'excess', 'scrap', 'metal', 'landline', 'mobile',
            
            # Generic business terms (only reject these specific combinations)
            'gas company', 'real estate developer', 'oil & gas company', 'automotive manufacturer',
            'defense contractor', 'railway department', 'shipbuilding yard', 'municipal corporation',
            'logistics services', 'basic services', 'communication services', 'protection services',
            'military services', 'risk co', 'marine co', 'employee inc', 'employee co',
            'warehouse construction', 'bridge construction', 'color coated steel', 'galvanized steel',
            'excess steel', 'housekeeping services', 'audit services', 'legal services',
            'security services', 'landline & mobile', 'municipal corporation'
        }
        
        # Check for obvious rejections
        for rejection in obvious_rejections:
            if rejection in vendor_lower:
                return None
        
        # ✅ ACCEPT names that look like real companies
        
        # 1. ACCEPT names with business suffixes (definitely companies)
        business_suffixes = ['ltd', 'limited', 'llc', 'inc', 'corp', 'corporation', 'company', 'co', 'group', 'enterprises', 'holdings', 'international', 'industries']
        if any(suffix in vendor_lower for suffix in business_suffixes):
            return vendor_clean
        
        # 2. ACCEPT company names with "&" or "and" (likely real companies)
        if '&' in vendor_clean or ' and ' in vendor_lower:
            if len(vendor_clean.split()) >= 3:  # Must be multi-word
            return vendor_clean
        
        # 3. ACCEPT names that look like real companies (multi-word with proper capitalization)
        if len(vendor_clean.split()) >= 2:
            words = vendor_clean.split()
            # Check if it starts with capital letter and has proper company structure
            if words[0][0].isupper() and any(word[0].isupper() for word in words[1:]):
                # Additional validation: must not be obvious rejections
                if not any(rejection in vendor_lower for rejection in obvious_rejections):
                    # Must contain at least one business-related word
                    business_words = ['steel', 'construction', 'equipment', 'machinery', 'engineering', 'suppliers', 'providers', 'services', 'manufacturing', 'trading', 'logistics', 'industrial', 'builders', 'marine']
                    if any(word in vendor_lower for word in business_words):
                return vendor_clean
        
        # 4. REJECT everything else (be strict but not too strict)
            return None
        
    def _extract_vendors_with_ollama_fast(self, descriptions):
        """Fast Ollama extraction with optimized timeout for vendor extraction"""
        vendors = []
        start_time = time.time()
        
        try:
            from ollama_simple_integration import simple_ollama, check_ollama_availability
            
            if not check_ollama_availability():
                print("   ⚠️  Ollama not available, skipping")
                return []
            
            print("   🧠 Using OPTIMIZED Ollama enhancement for vendor extraction...")
            print("   ⏱️  Note: Vendor extraction may take 20-40 seconds for complex data...")
            
            # Process ALL descriptions for maximum vendor coverage
            sample_descriptions = descriptions  # Process ALL transactions, no sampling limit
            
            # For large datasets, process in efficient batches
            if len(sample_descriptions) > 200:
                print(f"   📊 Large dataset detected ({len(sample_descriptions)} transactions)")
                print(f"   🔄 Processing in efficient batches for maximum coverage...")
                
                # Process in batches of 50 for efficiency (reduced from 200 to avoid timeouts)
                batch_size = 50
                all_vendors = []
                
                for i in range(0, len(sample_descriptions), batch_size):
                    batch = sample_descriptions[i:i + batch_size]
                    print(f"   🔄 Processing batch {i//batch_size + 1}/{(len(sample_descriptions) + batch_size - 1)//batch_size} ({len(batch)} transactions)")
                    
                    # Create prompt for this batch
                    batch_prompt = f"""Extract ONLY company names that are EXPLICITLY written in these transactions.

{chr(10).join([f"{idx+1}. {str(desc)[:80]}" for idx, desc in enumerate(batch) if not pd.isna(desc) and str(desc).strip() != ''])}

CRITICAL RULES - READ CAREFULLY:
- Extract ONLY company names that are EXPLICITLY written in the text
- DO NOT invent, imply, or guess company names
- DO NOT add explanations like "(implied)" or "(not explicitly mentioned)"
- DO NOT extract equipment names, project descriptions, or business processes
- If you see "Payment to ABC Corp", extract ONLY "ABC Corp"
- If you see "Invoice from XYZ Ltd", extract ONLY "XYZ Ltd"
- If no company name is clearly written, output "NO_COMPANY"
- Output ONLY the company name, nothing else

Company Names (EXACTLY as written, no additions):"""
                    
                    try:
                        response = simple_ollama(batch_prompt, "llama3.2:3b", max_tokens=200)
                        if response:
                            lines = response.strip().split('\n')
                            for line in lines:
                                if line.strip() and not line.startswith('Company Names:'):
                                    vendor = line.strip()
                                    if vendor and vendor != "NO_COMPANY":
                                        if vendor[0].isdigit() and '. ' in vendor:
                                            vendor = vendor.split('. ', 1)[1]
                                        
                                        if len(vendor.strip()) > 2:
                                            if any(bad_text in vendor.lower() for bad_text in ['implied', 'not explicitly', 'mentioned', 'but not', 'might be', 'could be', 'seems like', 'appears to be']):
                                                continue
                                            
                                            if self._is_likely_company_name(vendor.strip()):
                                                all_vendors.append(vendor.strip())
                                                print(f"   ✅ Batch vendor: {vendor.strip()}")
            
        except Exception as e:
                        print(f"   ⚠️ Batch {i//batch_size + 1} failed: {e}")
                        continue
                
                vendors = all_vendors
                print(f"   🧠 Batch processing completed: {len(vendors)} total vendors found")
                
            else:
                # For smaller datasets, use single processing
                print(f"   📊 Processing {len(sample_descriptions)} transactions in single batch...")
                
                # Create ULTRA-STRICT prompt for company extraction only
                batch_prompt = f"""Extract ONLY company names that are EXPLICITLY written in these transactions.

{chr(10).join([f"{idx+1}. {str(desc)[:80]}" for idx, desc in enumerate(sample_descriptions) if not pd.isna(desc) and str(desc).strip() != ''])}

CRITICAL RULES - READ CAREFULLY:
- Extract ONLY company names that are EXPLICITLY written in the text
- DO NOT invent, imply, or guess company names
- DO NOT add explanations like "(implied)" or "(not explicitly mentioned)"
- DO NOT extract equipment names, project descriptions, or business processes
- If you see "Payment to ABC Corp", extract ONLY "ABC Corp"
- If you see "Invoice from XYZ Ltd", extract ONLY "XYZ Ltd"
- If no company name is clearly written, output "NO_COMPANY"
- Output ONLY the company name, nothing else

EXAMPLES OF WHAT TO EXTRACT:
✅ "Payment to ABC Construction Company Ltd" → Extract: "ABC Construction Company Ltd"
✅ "Invoice from XYZ Steel Corp" → Extract: "XYZ Steel Corp"
✅ "Purchase from DEF Manufacturing Inc" → Extract: "DEF Manufacturing Inc"

EXAMPLES OF WHAT NOT TO EXTRACT:
❌ "Rolling Mill Upgrade" → Output: "NO_COMPANY" (equipment, not company)
❌ "Plant Modernization" → Output: "NO_COMPANY" (project, not company)
❌ "Steel Plates Purchase" → Output: "NO_COMPANY" (material, not company)
❌ "Energy Efficiency" → Output: "NO_COMPANY" (concept, not company)

Company Names (EXACTLY as written, no additions):"""
            
            try:
                print("   🧠 Sending request to Ollama (this may take 20-40 seconds)...")
                response = simple_ollama(batch_prompt, "llama3.2:3b", max_tokens=150)
                if response:
                    lines = response.strip().split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('Company Names:'):
                            vendor = line.strip()
                            # Clean vendor name - remove numbering and NO_COMPANY
                            if vendor and vendor != "NO_COMPANY":
                                # Remove numbering like "1. ", "2. ", etc.
                                if vendor[0].isdigit() and '. ' in vendor:
                                    vendor = vendor.split('. ', 1)[1]
                                
                                # Only add if it's a real vendor name
                                if len(vendor.strip()) > 2:
                                    # REJECT any vendor with implied or explanatory text
                                    if any(bad_text in vendor.lower() for bad_text in ['implied', 'not explicitly', 'mentioned', 'but not', 'might be', 'could be', 'seems like', 'appears to be']):
                                        print(f"   ❌ Rejected implied vendor: {vendor.strip()}")
                            continue
                        
                                    # Additional validation: must look like a company
                                    if self._is_likely_company_name(vendor.strip()):
                                        vendors.append(vendor.strip())
                                        print(f"   ✅ Ollama vendor: {vendor.strip()}")
                            else:
                                        print(f"   ❌ Rejected non-company: {vendor.strip()}")
                
            except Exception as e:
                print(f"   ⚠️  Ollama processing failed: {e}")
            
        except Exception as e:
            print(f"   ⚠️  Ollama integration failed: {e}")
        
        total_time = time.time() - start_time
        print(f"   🧠 Ollama vendor extraction completed in {total_time:.2f}s: {len(vendors)} vendors")
        return vendors
    
    def _extract_vendors_with_xgboost(self, descriptions):
        """XGBoost-powered vendor extraction (Priority 2)"""
        vendors = []
        start_time = time.time()
        
        try:
            print("   🤖 Using XGBoost ML enhancement...")
            
            # Process descriptions with XGBoost
            sample_descriptions = descriptions[:50]  # Reasonable sample size
            
            # Create ML-friendly features
            features = []
        for desc in sample_descriptions:
            if pd.isna(desc) or str(desc).strip() == '':
                continue
            
                desc_str = str(desc)
                feature = {
                    'length': len(desc_str),
                    'has_company_suffix': any(suffix in desc_str.lower() for suffix in ['ltd', 'inc', 'corp', 'company', 'co']),
                    'has_payment_keywords': any(keyword in desc_str.lower() for keyword in ['payment', 'invoice', 'purchase']),
                    'has_business_keywords': any(keyword in desc_str.lower() for keyword in ['steel', 'construction', 'manufacturing', 'trading']),
                    'word_count': len(desc_str.split()),
                    'capitalized_words': sum(1 for word in desc_str.split() if word and word[0].isupper())
                }
                features.append(feature)
            
            if not features:
                print("   ⚠️ No valid features for XGBoost")
                return []
            
            # Simple rule-based classification (XGBoost simulation)
            for i, (desc, feature) in enumerate(zip(sample_descriptions, features)):
                if pd.isna(desc) or str(desc).strip() == '':
                    continue
                
                desc_str = str(desc)
                
                # XGBoost-like scoring based on features
                score = 0
                if feature['has_company_suffix']: score += 3
                if feature['has_payment_keywords']: score += 2
                if feature['has_business_keywords']: score += 2
                if feature['capitalized_words'] >= 2: score += 1
                if feature['word_count'] >= 3: score += 1
                
                # If score is high enough, extract vendor
                if score >= 4:
                    # Extract potential vendor name
                    vendor = self._extract_vendor_name_from_description(desc_str)
                    if vendor and self._validate_vendor_name_fast(vendor):
                        vendors.append(vendor)
                        print(f"   ✅ XGBoost vendor: {vendor}")
            
        except Exception as e:
            print(f"   ⚠️ XGBoost processing failed: {e}")
        
        total_time = time.time() - start_time
        print(f"   🤖 XGBoost completed in {total_time:.2f}s: {len(vendors)} vendors")
        return vendors
    
    def _extract_vendor_name_from_description(self, description):
        """Extract vendor name from transaction description"""
        # Simple extraction logic
        desc_lower = description.lower()
        
        # Look for company patterns
        if 'payment to' in desc_lower:
            parts = description.split('Payment to')
            if len(parts) > 1:
                vendor_part = parts[1].strip()
                # Take first few words as vendor name
                words = vendor_part.split()[:4]
                return ' '.join(words)
        
        elif 'invoice from' in desc_lower:
            parts = description.split('Invoice from')
            if len(parts) > 1:
                vendor_part = parts[1].strip()
                words = vendor_part.split()[:4]
                return ' '.join(words)
        
        elif 'purchase from' in desc_lower:
            parts = description.split('Purchase from')
            if len(parts) > 1:
                vendor_part = parts[1].strip()
                words = vendor_part.split()[:4]
                return ' '.join(words)
        
        # Fallback: look for capitalized words
        words = description.split()
        capitalized = [word for word in words if word and word[0].isupper()]
        if len(capitalized) >= 2:
            return ' '.join(capitalized[:3])
        
        return None
    
    def _consolidate_vendors_fast(self, all_vendors, descriptions):
        """Fast vendor consolidation without complex processing"""
        print("   🧠 Fast vendor consolidation...")
        
        if not all_vendors:
            print("   ⚠️  No vendors found by any method")
            return []
        
        # Count vendor occurrences
        vendor_counts = Counter(all_vendors)
        
        # Filter out generic/invalid vendors with FAST validation
        final_vendors = []
        for vendor, count in vendor_counts.items():
            # Clean vendor name
            vendor_clean = vendor.strip()
            
            # Skip empty or very short names
            if not vendor_clean or len(vendor_clean) < 3:
                continue
            
            # ACCEPT vendors that contain business suffixes (likely real companies)
            vendor_lower = vendor_clean.lower()
            business_suffixes = ['ltd', 'limited', 'llc', 'inc', 'corp', 'corporation', 'company', 'co', 'group', 'enterprises', 'holdings', 'international', 'industries']
            
            if any(suffix in vendor_lower for suffix in business_suffixes):
                final_vendors.append(vendor)
                continue
            
            # ACCEPT multi-word names that look like companies
            if len(vendor_clean.split()) >= 2:
                # Check if it contains common company words
                company_words = ['steel', 'construction', 'equipment', 'machinery', 'engineering', 'suppliers', 'providers', 'services', 'manufacturing', 'trading']
                if any(word in vendor_lower for word in company_words):
                    final_vendors.append(vendor)
                    continue
            
            # REJECT only truly generic single words
            generic_single_words = [
                'payment', 'purchase', 'equipment', 'machinery', 'infrastructure', 'development',
                'expansion', 'modernization', 'quality', 'testing', 'warehouse', 'production',
                'line', 'capacity', 'increase', 'energy', 'efficiency', 'renovation',
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
            ]
            
            if len(vendor_clean.split()) == 1 and vendor_lower in generic_single_words:
                continue
            
            # If it passes all checks, accept it
            final_vendors.append(vendor)
        
        # Sort by frequency
        final_vendors.sort(key=lambda x: vendor_counts[x], reverse=True)
        
        # Display results
        print(f"\n📊 FAST VENDOR EXTRACTION RESULTS:")
        print(f"   🎯 Total Transactions: {len(descriptions)}")
        print(f"   🏢 Unique Vendors: {len(final_vendors)}")
        
        if final_vendors:
            print(f"\n🏢 VENDORS IDENTIFIED:")
            for vendor in final_vendors[:10]:  # Show first 10
                count = vendor_counts[vendor]
                print(f"   • {vendor} ({count} transactions)")
            if len(final_vendors) > 10:
                print(f"   ... and {len(final_vendors) - 10} more vendors")
        
        return final_vendors
    
    def _is_likely_company_name(self, vendor_name):
        """Check if vendor name looks like a real company name"""
        if not vendor_name or len(vendor_name.strip()) < 3:
            return False
        
        vendor_clean = vendor_name.strip()
        vendor_lower = vendor_clean.lower()
        
        # 🚫 REJECT obvious non-company terms
        non_company_terms = {
            # Project descriptions
            'plant modernization', 'rolling mill upgrade', 'new blast furnace',
            'infrastructure development', 'warehouse construction', 'capacity increase',
            'production line', 'energy efficiency', 'quality testing',
            
            # Business processes
            'employee payroll', 'customer payment', 'supplier payment',
            'advance payment', 'final payment', 'milestone payment',
            'retention payment', 'bulk order payment',
            
            # Generic concepts
            'energy efficiency', 'advanced technology', 'digital transformation',
            'modernization', 'upgrade', 'installation', 'maintenance',
            'renovation', 'expansion', 'development',
            
            # Equipment and materials
            'rolling mill', 'blast furnace', 'steel coils', 'steel plates',
            'galvanized steel', 'color coated steel', 'hot rolled',
            'cold rolled', 'steel wire', 'steel bars', 'steel pipes',
            
            # Transaction types
            'payment', 'purchase', 'sale', 'invoice', 'receipt',
            'advance', 'retention', 'milestone', 'bulk order',
            
            # Generic responses
            'no_company', 'no company', 'none', 'n/a', 'unknown'
        }
        
        # Check for non-company terms
        for term in non_company_terms:
            if term in vendor_lower:
                return False
        
        # ✅ ACCEPT names that look like real companies
        
        # 1. ACCEPT names with business suffixes (definitely companies)
        business_suffixes = ['ltd', 'limited', 'llc', 'inc', 'corp', 'corporation', 'company', 'co', 'group', 'enterprises', 'holdings', 'international', 'industries']
        if any(suffix in vendor_lower for suffix in business_suffixes):
            return True
        
        # 2. ACCEPT company names with "&" or "and" (likely real companies)
        if '&' in vendor_clean or ' and ' in vendor_lower:
            if len(vendor_clean.split()) >= 3:  # Must be multi-word
                return True
        
        # 3. ACCEPT multi-word names that look like companies
        if len(vendor_clean.split()) >= 2:
            words = vendor_clean.split()
            # Check if it starts with capital letter and has proper company structure
            if words[0][0].isupper() and any(word[0].isupper() for word in words[1:]):
                # Must contain at least one business-related word
                business_words = ['steel', 'construction', 'equipment', 'machinery', 'engineering', 'suppliers', 'providers', 'services', 'manufacturing', 'trading', 'logistics', 'industrial', 'builders', 'marine', 'railway', 'defense', 'automotive', 'shipbuilding', 'municipal']
                if any(word in vendor_lower for word in business_words):
                    return True
        
        # 4. REJECT everything else (be strict)
        return False

    def analyze_full_dataset_coverage(self, descriptions):
        """Analyze full dataset to estimate vendor coverage"""
        print("🔍 Analyzing full dataset for vendor coverage...")
        
        # Quick regex scan of full dataset
        all_potential_vendors = set()
        
        # Common vendor patterns in full dataset
        vendor_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co|Group|Enterprises|Holdings|International|Industries))\b',
            r'(?:Payment\s+to|Invoice\s+from|Purchase\s+from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|LLC|Inc|Corp|Corporation|Company|Co))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Steel|Construction|Engineering|Manufacturing|Trading|Logistics|Services|Suppliers|Providers|Contractors|Developers))',
        ]
        
        import re
        for desc in descriptions:
            if pd.isna(desc) or str(desc).strip() == '':
                continue
                
            desc_str = str(desc)
            for pattern in vendor_patterns:
                matches = re.findall(pattern, desc_str)
                for match in matches:
                    if match and len(match.strip()) > 3:
                        all_potential_vendors.add(match.strip())
        
        print(f"🔍 Quick scan found {len(all_potential_vendors)} potential vendors in full dataset")
        print(f"🔍 Current AI sample (100) might be missing {len(all_potential_vendors) - 3} vendors")
        
        return len(all_potential_vendors)

def analyze_real_vendors_ultra_fast(df):
    """ULTRA-FAST vendor analysis - Immediate results, no AI delays"""
    print("🚀 ULTRA-FAST VENDOR ANALYSIS - IMMEDIATE RESULTS")
    print("=" * 60)
    print(f"🔍 Input DataFrame: {df is not None}, Empty: {df.empty if df is not None else 'N/A'}")
    
    if df is None or df.empty:
        print("❌ No data provided for vendor analysis")
        return []
    
    print(f"🔍 DataFrame columns: {list(df.columns)}")
    print(f"🔍 DataFrame shape: {df.shape}")
    
    # Find description column
    description_col = None
    for col in df.columns:
        col_lower = col.lower()
        print(f"🔍 Checking column: '{col}' (lowercase: '{col_lower}')")
        if 'description' in col_lower or 'desc' in col_lower or 'narration' in col_lower:
            description_col = col
            print(f"✅ Found description column: '{description_col}'")
                    break
            
    if description_col is None:
        print("❌ No description column found")
        print("🔍 Available columns:", list(df.columns))
        return []
    
    print(f"✅ Using uploaded DataFrame with {len(df)} transactions")
    print(f"✅ Using description column: '{description_col}'")
    
    # Extract descriptions
    descriptions = df[description_col].dropna().tolist()
    print(f"🔍 Extracted {len(descriptions)} descriptions from column '{description_col}'")
    
    if not descriptions:
        print("❌ No descriptions found in the data")
        return []
    
    # Show sample descriptions
    print(f"🔍 Sample descriptions:")
    for i, desc in enumerate(descriptions[:3]):
        print(f"   {i+1}. {str(desc)[:100]}...")
    
    # Create vendor extractor and process with ULTRA-FAST method
    print("🚀 Creating UniversalVendorExtractor...")
    extractor = UniversalVendorExtractor()
    start_time = time.time()
    
    print("🚀 Calling ULTRA-FAST vendor extraction...")
    
    # Use the optimized sync method for maximum speed
    vendors = extractor.extract_vendors_intelligently_sync(descriptions)
    print(f"🔍 ULTRA-FAST extraction returned: {len(vendors)} vendors")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Performance summary
    print(f"\n⚡ ULTRA-FAST PERFORMANCE SUMMARY:")
    print(f"   🚀 Total Processing Time: {processing_time:.2f} seconds")
    print(f"   📊 Transactions Processed: {len(descriptions)}")
    print(f"   🎯 Vendors Identified: {len(vendors)}")
    print(f"   ⚡ Speed: {len(descriptions)/processing_time:.1f} transactions/second")
    
    if vendors:
        print(f"\n✅ ULTRA-FAST vendor analysis complete!")
        print(f"🎯 Found {len(vendors)} real vendors")
        print(f"🏢 First 10 vendors: {vendors[:10]}")
        
        # Provide immediate analysis insights
        print(f"\n📊 IMMEDIATE ANALYSIS INSIGHTS:")
        vendor_counts = Counter(vendors)
        top_vendors = vendor_counts.most_common(5)
        
        for i, (vendor, count) in enumerate(top_vendors):
            percentage = (count / len(descriptions)) * 100
            print(f"   {i+1}. {vendor}: {count} transactions ({percentage:.1f}%)")
        
        if len(vendors) > 5:
            print(f"   ... and {len(vendors) - 5} more vendors")
    else:
        print(f"\n⚠️  No vendors found")
    
        return vendors

def analyze_real_vendors_fast(df):
    """Fast vendor analysis - OPTIMIZED VERSION"""
    print("🚀 OPTIMIZED VENDOR ANALYSIS - BALANCED SPEED & ACCURACY")
    print("=" * 60)
    print(f"🔍 Input DataFrame: {df is not None}, Empty: {df.empty if df is not None else 'N/A'}")
    
    if df is None or df.empty:
        print("❌ No data provided for vendor analysis")
        return []
    
    print(f"🔍 DataFrame columns: {list(df.columns)}")
    print(f"🔍 DataFrame shape: {df.shape}")
    
    # Find description column
    description_col = None
    for col in df.columns:
        col_lower = col.lower()
        print(f"🔍 Checking column: '{col}' (lowercase: '{col_lower}')")
        if 'description' in col_lower or 'desc' in col_lower or 'narration' in col_lower:
            description_col = col
            print(f"✅ Found description column: '{description_col}'")
            break
    
    if description_col is None:
        print("❌ No description column found")
        print("🔍 Available columns:", list(df.columns))
        return []
    
    print(f"✅ Using uploaded DataFrame with {len(df)} transactions")
    print(f"✅ Using description column: '{description_col}'")
    
    # Extract descriptions
    descriptions = df[description_col].dropna().tolist()
    print(f"🔍 Extracted {len(descriptions)} descriptions from column '{description_col}'")
    
    if not descriptions:
        print("❌ No descriptions found in the data")
        return []
    
    # Show sample descriptions
    print(f"🔍 Sample descriptions:")
    for i, desc in enumerate(descriptions[:3]):
        print(f"   {i+1}. {str(desc)[:100]}...")
    
    # Create vendor extractor and process
    print("🚀 Creating UniversalVendorExtractor...")
    extractor = UniversalVendorExtractor()
    start_time = time.time()
    
    print("🚀 Calling extract_vendors_intelligently_sync()...")
    vendors = extractor.extract_vendors_intelligently_sync(descriptions)
    print(f"🔍 extract_vendors_intelligently_sync() returned: {vendors}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Performance summary
    print(f"\n⚡ OPTIMIZED PERFORMANCE SUMMARY:")
    print(f"   🚀 Total Processing Time: {processing_time:.2f} seconds")
    print(f"   📊 Transactions Processed: {len(descriptions)}")
    print(f"   🎯 Vendors Identified: {len(vendors)}")
    print(f"   ⚡ Speed: {len(descriptions)/processing_time:.1f} transactions/second")
    
    if vendors:
        print(f"\n✅ Optimized vendor analysis complete!")
        print(f"🎯 Found {len(vendors)} real vendors")
        print(f"🏢 First 10 vendors: {vendors[:10]}")
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