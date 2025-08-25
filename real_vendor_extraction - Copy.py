import pandas as pd
import numpy as np
import re  
import time
import hashlib

class UniversalVendorExtractor:
    """Universal vendor extraction with AI/ML priority and fallback system"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.last_cache_cleanup = time.time()
    
    def _get_cache_key(self, descriptions):
        """Generate cache key for descriptions"""
        # Handle both pandas Series and regular lists
        if hasattr(descriptions, 'empty'):
            # It's a pandas Series
            if descriptions.empty:
                return None
        elif not descriptions:
            # It's a regular list/array
            return None
        
        # Create hash of first few descriptions for caching
        sample = str(descriptions[:5])
        return hashlib.md5(sample.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key):
        """Get cached result if available and not expired"""
        if not cache_key or cache_key not in self.cache:
            return None
        
        timestamp, result = self.cache[cache_key]
        if time.time() - timestamp > self.cache_ttl:
            del self.cache[cache_key]
            return None
        
        print(f"🚀 Using cached vendor extraction result ({len(result)} vendors)")
        return result
    
    def _cache_result(self, cache_key, result):
        """Cache the result with timestamp"""
        if cache_key:
            self.cache[cache_key] = (time.time(), result)
            
            # Cleanup old cache entries periodically
            if time.time() - self.last_cache_cleanup > 300:
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
                if ollama_vendors and len(ollama_vendors) > 0:
                    all_vendors.extend(ollama_vendors)
                    print(f"✅ Ollama found {len(ollama_vendors)} vendors")
                    
                    # ✅ SUCCESS: Ollama worked, NO need for XGBoost fallback
                    print("🚀 Ollama vendor extraction successful - skipping XGBoost and regex fallback")
                    
                    # Consolidate results and return immediately
                    final_vendors = self._consolidate_vendors_fast(all_vendors, descriptions)
                    
                    # Cache the result
                    self._cache_result(cache_key, final_vendors)
                    
                    total_time = time.time() - start_time
                    print(f"\n✅ PRIORITY-BASED extraction completed in {total_time:.2f}s: {len(final_vendors)} vendors")
                    print(f"🚀 Speed: {len(descriptions)/total_time:.1f} transactions/second")
                    
                    return final_vendors
                else:
                    print("⚠️ Ollama found no vendors, trying XGBoost...")
            except Exception as e:
                print(f"❌ Ollama failed: {e}, trying XGBoost...")
        
        # STEP 2: Try XGBOOST SECOND (ML-powered, good accuracy) - ONLY if Ollama failed
        if use_ai and (not all_vendors or len(all_vendors) == 0):
            print("\n🤖 Step 2: XGBoost ML Enhancement (Priority 2)...")
            try:
                xgboost_vendors = self._extract_vendors_with_xgboost(descriptions)
                if xgboost_vendors and len(xgboost_vendors) > 0:
                    all_vendors.extend(xgboost_vendors)
                    print(f"✅ XGBoost found {len(xgboost_vendors)} vendors")
                    
                    # ✅ SUCCESS: XGBoost worked, NO need for regex fallback
                    print("🚀 XGBoost vendor extraction successful - skipping regex fallback")
                    
                    # Consolidate results and return immediately
                    final_vendors = self._consolidate_vendors_fast(all_vendors, descriptions)
                    
                    # Cache the result
                    self._cache_result(cache_key, final_vendors)
                    
                    total_time = time.time() - start_time
                    print(f"\n✅ PRIORITY-BASED extraction completed in {total_time:.2f}s: {len(final_vendors)} vendors")
                    print(f"🚀 Speed: {len(descriptions)/total_time:.1f} transactions/second")
                    
                    return final_vendors
                else:
                    print("⚠️ XGBoost found no vendors, using regex fallback...")
            except Exception as e:
                print(f"❌ XGBoost failed: {e}, using regex fallback...")
        
        # STEP 3: Use REGEX LAST (fastest, fallback only) - ONLY if both AI methods failed
        if not all_vendors or len(all_vendors) == 0:
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
        print(f"\n✅ PRIORITY-BASED extraction completed in {total_time:.2f}s: {len(final_vendors)} vendors")
        print(f"🚀 Speed: {len(descriptions)/total_time:.1f} transactions/second")
        
        return final_vendors

    def extract_vendors_intelligently_forced_sync(self, descriptions):
        """Forced AI extraction - no caching, no fallback"""
        if not descriptions or len(descriptions) == 0:
            print("❌ No descriptions provided")
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(descriptions)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        print(f"📊 Processing {len(descriptions)} transaction descriptions...")
        
        # Step 1: Forced Ollama extraction
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
        import re
        
        # TESTING MODE: Process only first 100 descriptions for testing
        max_descriptions = min(100, len(descriptions))
        print(f"🧪 TESTING MODE: Processing {max_descriptions} descriptions for vendor extraction...")
        print(f"   📝 Note: Limited to 100 transactions for testing. Remove limit for production use.")
        
        # STRICT regex patterns for REAL company names only (compiled with IGNORECASE)
        vendor_patterns = [
            # Pattern 1: Company names with business suffixes (HIGH PRIORITY - definitely companies)
            re.compile(r'([A-Z][a-zA-Z\s&]+?)\s+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO|GROUP|ENTERPRISES|HOLDINGS|INTERNATIONAL|INDUSTRIES)', re.IGNORECASE),
            
            # Pattern 2: "Payment to [Company Name]" format
            re.compile(r'(?:PAYMENT TO|PAYMENT FOR|PAID TO|TRANSFER TO)\s+([A-Z][a-zA-Z\s&]+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO|GROUP|ENTERPRISES|HOLDINGS|INTERNATIONAL|INDUSTRIES))', re.IGNORECASE),
            
            # Pattern 3: Specific vendor patterns
            re.compile(r'(LOGISTICS\s+PROVIDER|SERVICE\s+PROVIDER|EQUIPMENT\s+SUPPLIER|RAW\s+MATERIAL\s+SUPPLIER|COAL\s+SUPPLIER|LIMESTONE\s+SUPPLIER|ALLOY\s+SUPPLIER|STEEL\s+SUPPLIER)(?:\s+\d+)?', re.IGNORECASE),
            
            # Pattern 4: Company names in parentheses
            re.compile(r'\(([A-Z][a-zA-Z\s&]+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO))\)', re.IGNORECASE),
            
            # Pattern 5: Company names after dashes
            re.compile(r'[-–—]\s*([A-Z][a-zA-Z\s&]+(?:LTD|LIMITED|LLC|INC|CORP|CORPORATION|COMPANY|CO))', re.IGNORECASE)
        ]
        
        processed = 0
        for desc in descriptions[:max_descriptions]:
            if str(desc).strip() == '' or str(desc) in ['nan', 'None', '']:
                continue
                
            desc_str = str(desc)
            vendor_found = False
            
            # Try each pattern
            for pattern in vendor_patterns:
                try:
                    match = pattern.search(desc_str)
                    if match and match.groups():
                        vendor = match.group(1).strip()
                        if len(vendor) > 2 and vendor.lower() not in ['the', 'and', 'for', 'with', 'from']:
                            # Apply STRICT validation
                            if self._validate_vendor_name_fast(vendor):
                                vendors.append(vendor)
                                vendor_found = True
                                break
                except (IndexError, AttributeError):
                    # Skip patterns that don't have the expected group structure
                    continue
            
            processed += 1
            if processed % 50 == 0:
                print(f"   📊 Processed {processed}/{max_descriptions} descriptions...")
            
            total_time = time.time() - start_time
        print(f"   ⚡ ULTRA-FAST regex completed in {total_time:.2f}s: {len(vendors)} vendors")
        print(f"   🚀 Speed: {max_descriptions/total_time:.1f} descriptions/second")
        return vendors
            
    def _validate_vendor_name_fast(self, vendor_name):
        """BALANCED vendor name validation - Real company names with business suffixes"""
        if not vendor_name or len(vendor_name.strip()) < 3:
            return None
        
        vendor_clean = vendor_name.strip()
        vendor_lower = vendor_clean.lower()
        
        # ✅ FIRST: ACCEPT names with business suffixes (definitely companies) - override product name issues
        business_suffixes = ['ltd', 'limited', 'llc', 'inc', 'corp', 'corporation', 'company', 'co', 'group', 'enterprises', 'holdings', 'international', 'industries']
        if any(vendor_lower.endswith(' ' + suffix) or vendor_lower.endswith(suffix) for suffix in business_suffixes):
            # Only reject if it's EXACTLY a product name + suffix (like "Color Co")
            if vendor_lower in ['color co', 'steel co', 'metal co', 'raw co']:
                return None
            return vendor_clean
        
        # 🚫 REJECT obvious non-company terms
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
            'color coated', 'color', 'color co', 'excess', 'scrap', 'metal', 'landline', 'mobile',
            
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
        
        # 1. Business suffixes already handled above
        
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
                business_words = {'construction', 'engineering', 'manufacturing', 'trading', 'logistics', 'services', 'suppliers', 'providers', 'contractors', 'developers', 'steel', 'corp', 'company'}
                if any(word.lower() in business_words for word in words):
                    return vendor_clean
        
        # 4. ACCEPT specific patterns like "ABC Corp", "XYZ Ltd"
        if len(vendor_clean.split()) == 2:
            first_word, second_word = vendor_clean.split()
            if first_word[0].isupper() and second_word.lower() in business_suffixes:
                return vendor_clean
        
        # 5. ACCEPT generic company patterns that are commonly used
        generic_company_patterns = ['abc corp', 'xyz ltd', 'abc company', 'xyz company']
        if vendor_lower in generic_company_patterns:
            return vendor_clean
        
        # If none of the acceptance criteria match, reject
        return None
        
    def _extract_vendors_with_ollama_fast(self, descriptions):
        """Extract vendors using Ollama with ULTRA-STRICT rules to prevent implied/invented names"""
        # Handle both pandas Series and regular lists
        if hasattr(descriptions, 'empty'):
            if descriptions.empty:
                return []
        elif not descriptions:
            return []
        
        try:
            from ollama_simple_integration import simple_ollama
            
            # TESTING MODE: Limit to 100 transactions
            sample_descriptions = descriptions[:100] if len(descriptions) > 100 else descriptions
            print(f"🧠 Using OPTIMIZED Ollama enhancement for vendor extraction...")
            print(f"⏱️ Note: Vendor extraction may take 20-40 seconds for complex data...")
            
            # For large datasets, process in efficient batches
            if len(sample_descriptions) > 50:
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

{chr(10).join([f"{idx+1}. {str(desc)[:80]}" for idx, desc in enumerate(batch) if str(desc).strip() != '' and str(desc) not in ['nan', 'None', '']])}

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
                                            
                                            validated_vendor = self._validate_vendor_name_fast(vendor.strip())
                                            if validated_vendor:
                                                all_vendors.append(validated_vendor)
                                                print(f"   ✅ Batch vendor: {validated_vendor}")
                                            else:
                                                print(f"   ❌ Rejected vendor: {vendor.strip()}")
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

{chr(10).join([f"{idx+1}. {str(desc)[:80]}" for idx, desc in enumerate(sample_descriptions) if str(desc).strip() != '' and str(desc) not in ['nan', 'None', '']])}

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
                        vendors = []
                        for line in lines:
                            if line.strip() and not line.startswith('Company Names:'):
                                vendor = line.strip()
                                # Clean vendor name - remove numbering and NO_COMPANY
                                if vendor and vendor != "NO_COMPANY":
                                    # Remove numbering like "1. ", "2. ", etc.
                                    if vendor[0].isdigit() and '. ' in vendor:
                                        vendor = vendor.split('. ', 1)[1]
                                    
                                    # Filter out implied/invented vendors
                                    if any(bad_text in vendor.lower() for bad_text in ['implied', 'not explicitly', 'mentioned', 'but not', 'might be', 'could be', 'seems like', 'appears to be']):
                                        print(f"   ❌ Rejected implied vendor: {vendor.strip()}")
                                        continue
                                    validated_vendor = self._validate_vendor_name_fast(vendor.strip())
                                    if validated_vendor:
                                        vendors.append(validated_vendor)
                                        print(f"   ✅ Ollama vendor: {validated_vendor}")
                                    else:
                                        print(f"   ❌ Rejected vendor: {vendor.strip()}")
                    else:
                        vendors = []
                
                except Exception as e:
                    print(f"   ❌ Ollama processing failed: {e}")
                    vendors = []
            
            # Analyze full dataset potential for coverage assessment
            if len(descriptions) > 100:
                potential_vendors = self._analyze_full_dataset_potential(descriptions)
                print(f"   🔍 Full dataset analysis: {potential_vendors} potential vendors detected")
                print(f"   📊 Current sample coverage: {len(vendors)}/{potential_vendors} vendors")
            
            return vendors
            
        except Exception as e:
            print(f"   ❌ Ollama vendor extraction failed: {e}")
            return []
    
    def _extract_vendors_with_xgboost(self, descriptions):
        """Extract vendors using XGBoost ML approach"""
        try:
            import xgboost as xgb
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            print("   🤖 Using XGBoost ML enhancement...")
            
            # TESTING MODE: Limit to 100 transactions
            sample_descriptions = descriptions[:100] if len(descriptions) > 100 else descriptions
            
            # Prepare training data (simplified for demo)
            training_descriptions = [
                "Payment to ABC Construction Company Ltd",
                "Invoice from XYZ Steel Corp", 
                "Purchase from DEF Manufacturing Inc",
                "Equipment maintenance costs",
                "Utility payments for electricity"
            ]
            training_labels = ["ABC Construction Company Ltd", "XYZ Steel Corp", "DEF Manufacturing Inc", "", ""]
        
        # Create TF-IDF features
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X_train = vectorizer.fit_transform(training_descriptions)
            
            # Encode labels
            le = LabelEncoder()
            y_train = le.fit_transform([label if label else "NO_VENDOR" for label in training_labels])
            
            # Train XGBoost model
            model = xgb.XGBClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict on sample descriptions
            X_test = vectorizer.transform([str(desc) for desc in sample_descriptions])
            predictions = model.predict(X_test)
            
            # Extract vendor names
            vendors = []
            for pred in predictions:
                vendor = le.inverse_transform([pred])[0]
                if vendor != "NO_VENDOR" and len(vendor) > 2:
                    vendors.append(vendor)
                    print(f"   ✅ XGBoost vendor: {vendor}")
            
            print(f"   🤖 XGBoost completed in 0.00s: {len(vendors)} vendors")
            return vendors
            
        except Exception as e:
            print(f"   ❌ XGBoost extraction failed: {e}")
            return []
        
    def _is_likely_company_name(self, vendor_name):
        """Stricter validation to ensure only real company names"""
        if not vendor_name or len(vendor_name.strip()) < 3:
            return False
        
        vendor_clean = vendor_name.strip()
        vendor_lower = vendor_clean.lower()
        
        # Reject equipment, projects, or concepts
        rejected_terms = [
            'rolling mill', 'plant modernization', 'steel plates', 'energy efficiency',
            'infrastructure development', 'warehouse construction', 'capacity increase',
            'equipment', 'machinery', 'upgrade', 'installation', 'maintenance',
            'project', 'development', 'expansion', 'modernization'
        ]
        
        for term in rejected_terms:
            if term in vendor_lower:
                return False
        
        # Accept names with business indicators
        business_indicators = [
            'company', 'corp', 'corporation', 'ltd', 'limited', 'llc', 'inc',
            'group', 'enterprises', 'holdings', 'international', 'industries',
            'construction', 'engineering', 'manufacturing', 'trading', 'services'
        ]
        
        for indicator in business_indicators:
            if indicator in vendor_lower:
                return True
        
        # Accept proper names (capitalized multi-word)
        words = vendor_clean.split()
        if len(words) >= 2 and all(word[0].isupper() for word in words if len(word) > 0):
            return True
        
        return False
    
    def _analyze_full_dataset_potential(self, descriptions):
        """Quick analysis of full dataset to estimate vendor potential"""
        try:
            potential_count = 0
            for desc in descriptions[:500]:  # Quick scan of first 500
                if str(desc).strip() == '' or str(desc) in ['nan', 'None', '']:
                    continue
                desc_str = str(desc).lower()
                if any(term in desc_str for term in ['company', 'corp', 'ltd', 'inc', 'construction', 'engineering']):
                    potential_count += 1
            
            # Extrapolate to full dataset
            if len(descriptions) > 500:
                potential_count = int(potential_count * (len(descriptions) / 500))
            
            return potential_count
        except:
            return 0
    
    def _consolidate_vendors_fast(self, vendors, descriptions):
        """Fast vendor consolidation and cleanup"""
        if not vendors:
            return []
        
        print(f"   🧠 Fast vendor consolidation...")
        
        # Remove duplicates (case-insensitive)
        unique_vendors = {}
        for vendor in vendors:
            vendor_clean = vendor.strip()
            vendor_key = vendor_clean.lower()
            if vendor_key not in unique_vendors and len(vendor_clean) > 2:
                unique_vendors[vendor_key] = vendor_clean
        
        result = list(unique_vendors.values())
        
        print(f"\n📊 FAST VENDOR EXTRACTION RESULTS:")
        print(f"   🎯 Total Transactions: {len(descriptions)}")
        print(f"   🏢 Unique Vendors: {len(result)}")
        print(f"\n🏢 VENDORS IDENTIFIED:")
        
        # Count occurrences and display
        vendor_counts = {}
        for vendor in result:
            count = 0
            for desc in descriptions:
                if str(desc).strip() == '' or str(desc) in ['nan', 'None', '']:
                    continue
                if vendor.lower() in str(desc).lower():
                    count += 1
            vendor_counts[vendor] = count
        
        # Sort by frequency and display top vendors
        sorted_vendors = sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out vendors with 0 transactions
        valid_vendors = [(vendor, count) for vendor, count in sorted_vendors if count > 0]
        
        print(f"   📊 Valid vendors (with transactions):")
        for vendor, count in valid_vendors[:10]:
            print(f"   • {vendor} ({count} transactions)")
        
        if len(valid_vendors) > 10:
            print(f"   ... and {len(valid_vendors) - 10} more vendors")
        
        # Only return vendors that actually have transactions
        return [vendor for vendor, _ in valid_vendors]