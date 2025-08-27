import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class LLMInterface:
    """Simple LLM Interface for Phase 2 analysis with comprehensive logging"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.has_api_key = bool(self.api_key)
        
        # Setup logging
        self.setup_logging()
        
        if self.has_api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key)
            self.log_info("✅ LLMInterface initialized with OpenAI API")
        else:
            self.client = None
            self.log_info("⚠️ LLMInterface initialized in mock mode (no API key or OpenAI package)")
    
    def setup_logging(self):
        """Setup file logging for detailed tracking with Unicode support"""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('LLMInterface')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for detailed logs with UTF-8 encoding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/llm_analysis_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Simple formatter without emojis for file logging
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
        self.log_info(f"Logging started - File: {log_file}")
    
    def log_info(self, message: str):
        """Log info message with emoji-safe handling"""
        # Remove emojis for file logging to avoid encoding issues
        clean_message = self._clean_message_for_logging(message)
        self.logger.info(clean_message)
        print(f"[LLM] {message}")  # Keep emojis for console
    
    def log_error(self, message: str):
        """Log error message with emoji-safe handling"""
        clean_message = self._clean_message_for_logging(message)
        self.logger.error(clean_message)
        print(f"[LLM ERROR] {message}")
    
    def _clean_message_for_logging(self, message: str) -> str:
        """Remove emojis and special characters for safe file logging"""
        import re
        # Remove common emojis and special Unicode characters
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002700-\U000027BF"  # dingbats
            "\U0001f926-\U0001f937"  # additional emoticons
            "\U00010000-\U0010ffff"  # supplementary symbols
            "\u2640-\u2642"          # gender symbols
            "\u2600-\u2B55"          # miscellaneous symbols
            "\u200d"                 # zero width joiner
            "\u23cf"                 # eject symbol
            "\u23e9"                 # fast forward
            "\u231a"                 # watch
            "\ufe0f"                 # variation selector
            "\u3030"                 # wavy dash
            "]+", 
            flags=re.UNICODE
        )
        
        # Replace emojis with text equivalents
        clean_msg = message
        clean_msg = clean_msg.replace("🚀", "[START]")
        clean_msg = clean_msg.replace("📊", "[DATA]")
        clean_msg = clean_msg.replace("📋", "[INFO]")
        clean_msg = clean_msg.replace("📂", "[COLUMN]")
        clean_msg = clean_msg.replace("💾", "[SAVE]")
        clean_msg = clean_msg.replace("❌", "[ERROR]")
        clean_msg = clean_msg.replace("✅", "[SUCCESS]")
        clean_msg = clean_msg.replace("⚠️", "[WARNING]")
        clean_msg = clean_msg.replace("🔄", "[PROCESS]")
        clean_msg = clean_msg.replace("🤖", "[LLM]")
        clean_msg = clean_msg.replace("🌐", "[API]")
        clean_msg = clean_msg.replace("📄", "[FILE]")
        clean_msg = clean_msg.replace("📝", "[PROMPT]")
        clean_msg = clean_msg.replace("🔍", "[DETECT]")
        clean_msg = clean_msg.replace("🎭", "[MOCK]")
        clean_msg = clean_msg.replace("🪑", "[SEAT]")
        clean_msg = clean_msg.replace("✈️", "[FLIGHT]")
        clean_msg = clean_msg.replace("💰", "[PRICE]")
        clean_msg = clean_msg.replace("🛬", "[LAYOVER]")
        clean_msg = clean_msg.replace("🗂️", "[LOG]")
        
        # Remove any remaining emojis
        clean_msg = emoji_pattern.sub('', clean_msg)
        
        return clean_msg
    
    def save_data_to_file(self, data: Any, filename: str, description: str):
        """Save data to file and log it"""
        filepath = f"logs/{filename}"
        try:
            if isinstance(data, dict) or isinstance(data, list):
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                with open(filepath, 'w') as f:
                    f.write(str(data))
            
            self.log_info(f"💾 {description} saved to: {filepath}")
            return filepath
        except Exception as e:
            self.log_error(f"❌ Failed to save {description}: {str(e)}")
            return None
    
    def analyze_data_quality(self, sample_data, column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis function with comprehensive logging"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_info(f"🚀 Starting LLM analysis at {timestamp}")
        self.log_info(f"📊 Input data shape: {sample_data.shape}")
        self.log_info(f"📋 Columns to analyze: {list(column_info.keys())}")
        
        # Log detailed column information
        self.log_info("📂 Column Details:")
        for col, info in column_info.items():
            self.log_info(f"   • {col}: {info['dtype']} | Nulls: {info['null_count']} | Unique: {info['unique_count']}")
        
        # Save input data to files
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save sample data
        sample_file = self.save_data_to_file(
            sample_data.to_dict('records'), 
            f"input_sample_data_{timestamp_str}.json",
            "Input sample data"
        )
        
        # Save column info
        column_file = self.save_data_to_file(
            column_info,
            f"input_column_info_{timestamp_str}.json", 
            "Input column information"
        )
        
        # Determine analysis method
        if not self.has_api_key or not OPENAI_AVAILABLE:
            self.log_info("🔄 Using mock analysis (no API key or OpenAI package)")
            return self._get_mock_analysis(sample_data, column_info, timestamp_str)
        
        try:
            self.log_info("🤖 Using OpenAI API for real analysis")
            return self._get_openai_analysis(sample_data, column_info, timestamp_str)
        except Exception as e:
            self.log_error(f"❌ OpenAI API Error: {str(e)}")
            self.log_info("🔄 Falling back to mock analysis")
            return self._get_mock_analysis(sample_data, column_info, timestamp_str)
    
    def _get_openai_analysis(self, sample_data, column_info: Dict[str, Any], timestamp_str: str) -> Dict[str, Any]:
        """Real OpenAI analysis with comprehensive logging"""
        
        self.log_info("📋 Preparing data for OpenAI analysis...")
        
        # Prepare data summary
        data_summary = {
            'total_rows': len(sample_data),
            'total_columns': len(sample_data.columns),
            'column_info': column_info
        }
        
        # Get few sample rows
        sample_rows = sample_data.head(3).to_dict('records')
        self.log_info(f"📄 Sample rows prepared: {len(sample_rows)} rows")
        
        # Build prompt
        prompt = self._build_analysis_prompt(data_summary, sample_rows)
        self.log_info(f"📝 Prompt built (length: {len(prompt)} characters)")
        
        # Save prompt to file
        self.save_data_to_file(
            prompt,
            f"openai_prompt_{timestamp_str}.txt",
            "OpenAI prompt"
        )
        
        try:
            self.log_info("🌐 Sending request to OpenAI GPT-3.5-turbo...")
            
            # Log API request details
            request_data = {
                "model": "gpt-3.5-turbo",
                "temperature": 0.05,
                "max_tokens": 3500,
                "prompt_length": len(prompt),
                "timestamp": datetime.now().isoformat()
            }
            
            self.save_data_to_file(
                request_data,
                f"openai_request_{timestamp_str}.json",
                "OpenAI request details"
            )
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior data quality auditor and airline domain expert. You must perform systematic, comprehensive analysis using the provided checklist. Your analysis must be thorough, consistent, and catch ALL possible data quality issues. Always provide complete column renaming suggestions and detailed evidence for every issue found. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,
                max_tokens=3500
            )
            
            self.log_info("✅ OpenAI API response received")
            response_text = response.choices[0].message.content
            self.log_info(f"📄 Response length: {len(response_text)} characters")
            
            # Save raw response
            self.save_data_to_file(
                response_text,
                f"openai_response_raw_{timestamp_str}.txt",
                "OpenAI raw response"
            )
            
            # Log response details
            response_details = {
                "response_length": len(response_text),
                "model_used": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": dict(response.usage) if hasattr(response, 'usage') else None,
                "timestamp": datetime.now().isoformat()
            }
            
            self.save_data_to_file(
                response_details,
                f"openai_response_details_{timestamp_str}.json",
                "OpenAI response metadata"
            )
            
            # Try to parse JSON
            try:
                analysis_result = json.loads(response_text)
                self.log_info("✅ Successfully parsed JSON response")
                self.log_info(f"🔍 Issues found: {analysis_result.get('total_issues', 0)}")
                
                # Save parsed result
                self.save_data_to_file(
                    analysis_result,
                    f"openai_parsed_result_{timestamp_str}.json",
                    "Parsed OpenAI result"
                )
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                self.log_error(f"❌ JSON parsing failed: {str(e)}")
                self.log_info("🔄 Using text parsing fallback")
                
                # Save parsing error details
                error_details = {
                    "error": str(e),
                    "raw_response": response_text,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.save_data_to_file(
                    error_details,
                    f"json_parse_error_{timestamp_str}.json",
                    "JSON parsing error details"
                )
                
                return self._parse_text_response(response_text, timestamp_str)
                
        except Exception as e:
            self.log_error(f"❌ OpenAI API call failed: {str(e)}")
            
            # Save API error details
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            
            self.save_data_to_file(
                error_details,
                f"api_error_{timestamp_str}.json",
                "API error details"
            )
            
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def _build_analysis_prompt(self, data_summary: Dict, sample_rows: list) -> str:
        """Build enhanced structured analysis prompt for consistent comprehensive results"""
        
        prompt = f"""
You are an expert data quality auditor specializing in airline/flight data analysis. Your task is to perform a COMPREHENSIVE and SYSTEMATIC analysis to catch ALL possible data quality issues.

=== DATASET OVERVIEW ===
- Total Rows: {data_summary['total_rows']}
- Total Columns: {data_summary['total_columns']}

=== COLUMN DETAILS ===
{json.dumps(data_summary['column_info'], indent=2)}

=== SAMPLE DATA ===
{json.dumps(sample_rows, indent=2)}

=== SYSTEMATIC ANALYSIS CHECKLIST ===
You MUST examine EVERY category below. Do not skip any section. Report ALL issues you find.

1. DATA TYPE VALIDATION:
   ☐ Check for numeric fields stored as text
   ☐ Verify date/time fields have proper formats
   ☐ Identify mixed data types in same column
   ☐ Find categorical data that should be standardized

2. BUSINESS LOGIC VIOLATIONS:
   ☐ Impossible dates (departure after arrival, future bookings beyond reasonable limits)
   ☐ Negative values where they shouldn't exist (prices, ages, durations)
   ☐ Contradictory status combinations (cancelled but also confirmed)
   ☐ Missing required relationships (passenger without booking ID)

3. AIRLINE DOMAIN SPECIFIC CHECKS:
   ☐ Invalid airport codes (not 3-letter IATA format)
   ☐ Impossible flight durations (too short/long for distance)
   ☐ Unrealistic pricing (too high/low for route type)
   ☐ Invalid seat configurations or class codes
   ☐ Airline codes that don't match standard formats

4. DATA CONSISTENCY ISSUES:
   ☐ Same passenger with different names/details
   ☐ Flight numbers that don't match airline patterns
   ☐ Booking references with inconsistent formats
   ☐ Currency mismatches with routes/airlines

5. MISSING DATA PATTERNS:
   ☐ Critical fields with too many nulls
   ☐ Suspicious patterns in missing data
   ☐ Related fields that should be filled together
   ☐ Optional vs required field validation

6. STATISTICAL ANOMALIES:
   ☐ Extreme outliers in pricing, duration, dates
   ☐ Unusual distributions (all same values, impossible spikes)
   ☐ Values outside expected ranges for field type
   ☐ Duplicate rows or near-duplicate suspicious patterns

7. FORMAT AND ENCODING ISSUES:
   ☐ Special characters causing problems
   ☐ Inconsistent capitalization or spacing
   ☐ Date formats that vary within same column
   ☐ Phone numbers, emails with invalid formats

=== COLUMN RENAMING REQUIREMENTS ===
Provide business-friendly names for ALL columns. Focus on:
- Technical abbreviations (dep_time → Departure_Time)
- Cryptic codes (airlie_id → Airline_ID)
- Database naming (created_at → Record_Created_Date)
- Industry jargon (pax → Passenger_Count)

=== OUTPUT FORMAT ===
Return ONLY valid JSON in this exact structure:

{{
    "total_issues": <number>,
    "issues": [
        {{
            "category": "<data_type|business_logic|airline_domain|consistency|missing_data|statistical|format>",
            "severity": "<high|medium|low>",
            "description": "<specific detailed description of the issue>",
            "affected_columns": ["<exact column names>"],
            "fix_approach": "<specific actionable solution>",
            "evidence": "<specific examples from the data that show this issue>"
        }}
    ],
    "column_renaming": {{
        "<current_column_name>": "<clear_business_friendly_name>"
    }},
    "recommendations": [
        "<specific actionable recommendations for data quality improvement>"
    ],
    "analysis_confidence": "<high|medium|low>",
    "domain_notes": "<any airline-specific observations or domain expertise applied>"
}}

=== INSTRUCTIONS ===
- BE THOROUGH: Check every category systematically
- BE SPECIFIC: Provide exact column names and evidence
- BE ACTIONABLE: Give concrete solutions, not vague suggestions
- BE COMPREHENSIVE: Don't stop at first few issues, find ALL problems
- EXAMINE RELATIONSHIPS: Look for cross-column issues and patterns
- USE DOMAIN KNOWLEDGE: Apply airline industry expertise
- VALIDATE EVERYTHING: Question assumptions and verify business rules

Remember: Your goal is to catch EVERY possible data quality issue. Missing problems could lead to operational failures in airline systems.
"""
        return prompt
    
    def _parse_text_response(self, response_text: str, timestamp_str: str) -> Dict[str, Any]:
        """Parse non-JSON response from OpenAI with logging"""
        
        self.log_info("🔄 Parsing text response (non-JSON)")
        
        issues = []
        
        # Simple keyword detection with logging
        if "missing" in response_text.lower() or "null" in response_text.lower():
            issues.append({
                "category": "missing_data",
                "severity": "medium",
                "description": "Missing data issues detected from text analysis",
                "affected_columns": [],
                "fix_approach": "Handle missing values appropriately"
            })
            self.log_info("🔍 Detected missing data issue from text")
        
        if "seat" in response_text.lower() and ("window" in response_text.lower() or "aisle" in response_text.lower()):
            issues.append({
                "category": "seat_conflicts",
                "severity": "medium",
                "description": "Potential seat assignment conflicts detected",
                "affected_columns": ["window_seat", "aisle_seat"],
                "fix_approach": "Review seat assignment logic"
            })
            self.log_info("🔍 Detected seat conflict issue from text")
        
        if "price" in response_text.lower() or "fare" in response_text.lower():
            issues.append({
                "category": "pricing",
                "severity": "medium",
                "description": "Pricing inconsistencies detected",
                "affected_columns": ["fare"],
                "fix_approach": "Review pricing logic"
            })
            self.log_info("🔍 Detected pricing issue from text")
        
        if "cancel" in response_text.lower() or "status" in response_text.lower():
            issues.append({
                "category": "status_logic",
                "severity": "low",
                "description": "Status-related issues detected",
                "affected_columns": ["status"],
                "fix_approach": "Validate status consistency"
            })
            self.log_info("🔍 Detected status issue from text")
        
        result = {
            "total_issues": len(issues),
            "issues": issues,
            "column_renaming": {
                "example": "Column renaming suggestions would appear here from full analysis"
            },
            "recommendations": [
                "Review the full OpenAI response for detailed insights",
                f"Original response length: {len(response_text)} characters"
            ]
        }
        
        self.log_info(f"✅ Text parsing complete - found {len(issues)} issues")
        
        # Save text parsing result
        self.save_data_to_file(
            result,
            f"text_parsed_result_{timestamp_str}.json",
            "Text parsing result"
        )
        
        return result
    
    def _get_mock_analysis(self, sample_data, column_info: Dict[str, Any], timestamp_str: str) -> Dict[str, Any]:
        """Mock analysis for basic flight booking issues with comprehensive logging"""
        
        self.log_info("🎭 Starting mock analysis (no OpenAI API)")
        issues = []
        
        # Basic business logic checks for simulated data
        if len(sample_data) > 0:
            
            # 1. Seat assignment conflicts
            self.log_info("🪑 Checking seat assignment conflicts...")
            if 'window_seat' in sample_data.columns and 'aisle_seat' in sample_data.columns:
                try:
                    both_seats = sample_data[(sample_data['window_seat'] == True) & (sample_data['aisle_seat'] == True)]
                    if len(both_seats) > 0:
                        issues.append({
                            'category': 'seat_conflicts',
                            'severity': 'medium',
                            'description': f'Found {len(both_seats)} passengers with both window AND aisle seats assigned',
                            'affected_columns': ['window_seat', 'aisle_seat'],
                            'fix_approach': 'Passengers can only have window OR aisle, not both'
                        })
                        self.log_info(f"   ⚠️ Found {len(both_seats)} seat conflicts")
                    else:
                        self.log_info("   ✅ No seat conflicts found")
                except Exception as e:
                    self.log_error(f"   ❌ Error checking seat conflicts: {str(e)}")
            
            # 2. Cancelled flights with operational data
            self.log_info("✈️ Checking cancelled flight logic...")
            if 'status' in sample_data.columns:
                try:
                    cancelled = sample_data[sample_data['status'] == 'Cancelled']
                    self.log_info(f"   📊 Found {len(cancelled)} cancelled flights")
                    
                    if len(cancelled) > 0:
                        # Check if cancelled flights have gates
                        if 'gate' in sample_data.columns:
                            cancelled_with_gates = cancelled[cancelled['gate'].notna()]
                            if len(cancelled_with_gates) > 0:
                                issues.append({
                                    'category': 'status_logic',
                                    'severity': 'low',
                                    'description': f'Found {len(cancelled_with_gates)} cancelled flights with gate assignments',
                                    'affected_columns': ['status', 'gate'],
                                    'fix_approach': 'Remove operational details for cancelled flights'
                                })
                                self.log_info(f"   ⚠️ Found {len(cancelled_with_gates)} cancelled flights with gates")
                            else:
                                self.log_info("   ✅ No cancelled flights with gate assignments")
                except Exception as e:
                    self.log_error(f"   ❌ Error checking cancelled flights: {str(e)}")
            
            # 3. Basic pricing logic
            self.log_info("💰 Checking pricing logic...")
            if 'fare' in sample_data.columns:
                try:
                    fares = sample_data['fare'].dropna()
                    if len(fares) > 0:
                        very_high = fares[fares > 1000]
                        very_low = fares[fares < 50]
                        
                        self.log_info(f"   📊 Fare range: ${fares.min():.2f} - ${fares.max():.2f}")
                        
                        if len(very_high) > 0:
                            issues.append({
                                'category': 'pricing',
                                'severity': 'medium',
                                'description': f'Found {len(very_high)} tickets with unusually high fares (>${very_high.max():.0f})',
                                'affected_columns': ['fare'],
                                'fix_approach': 'Review pricing logic for extremely high fares'
                            })
                            self.log_info(f"   ⚠️ Found {len(very_high)} high-priced tickets")
                        
                        if len(very_low) > 0:
                            issues.append({
                                'category': 'pricing',
                                'severity': 'medium',
                                'description': f'Found {len(very_low)} tickets with unusually low fares (${very_low.min():.0f})',
                                'affected_columns': ['fare'],
                                'fix_approach': 'Verify if extremely low fares are intentional'
                            })
                            self.log_info(f"   ⚠️ Found {len(very_low)} low-priced tickets")
                        
                        if len(very_high) == 0 and len(very_low) == 0:
                            self.log_info("   ✅ All fares within reasonable range")
                    
                except Exception as e:
                    self.log_error(f"   ❌ Error checking pricing: {str(e)}")
            
            # 4. Layover logic
            self.log_info("🛬 Checking layover logic...")
            if 'number_of_stops' in sample_data.columns and 'layover_locations' in sample_data.columns:
                try:
                    no_stops = sample_data[sample_data['number_of_stops'] == 0]
                    no_stops_with_layovers = no_stops[no_stops['layover_locations'].notna() & (no_stops['layover_locations'] != '')]
                    
                    if len(no_stops_with_layovers) > 0:
                        issues.append({
                            'category': 'layover_logic',
                            'severity': 'low',
                            'description': f'Found {len(no_stops_with_layovers)} direct flights (0 stops) with layover locations',
                            'affected_columns': ['number_of_stops', 'layover_locations'],
                            'fix_approach': 'Direct flights should not have layover locations'
                        })
                        self.log_info(f"   ⚠️ Found {len(no_stops_with_layovers)} direct flights with layovers")
                    else:
                        self.log_info("   ✅ Layover logic appears consistent")
                
                except Exception as e:
                    self.log_error(f"   ❌ Error checking layover logic: {str(e)}")
        
        # 5. Missing critical data
        self.log_info("📋 Checking missing critical data...")
        critical_columns = ['passngr_nm', 'seat_no', 'booking_cd']
        for col in critical_columns:
            if col in column_info:
                col_info = column_info[col]
                if len(sample_data) > 0:
                    null_pct = (col_info['null_count'] / len(sample_data)) * 100
                    self.log_info(f"   📊 {col}: {null_pct:.1f}% missing")
                    
                    if null_pct > 10:  # More than 10% missing for critical fields
                        issues.append({
                            'category': 'missing_critical',
                            'severity': 'high',
                            'description': f'Critical field "{col}" has {null_pct:.1f}% missing values',
                            'affected_columns': [col],
                            'fix_approach': f'All bookings must have {col.replace("_", " ")}'
                        })
                        self.log_info(f"   ⚠️ High missing percentage for {col}")
                    else:
                        self.log_info(f"   ✅ {col} has acceptable missing rate")
        
        # Generate column renaming suggestions
        self.log_info("📝 Generating column renaming suggestions...")
        column_renaming = {}
        
        for col in sample_data.columns:
            # Suggest business-friendly names for common patterns
            if '_' in col or col.lower() in ['passngr_nm', 'flght#', 'dep_time', 'arrivl_time']:
                if 'passngr_nm' in col.lower():
                    column_renaming[col] = "Passenger_Name"
                elif 'flght' in col.lower():
                    column_renaming[col] = "Flight_Number"
                elif 'dep_time' in col.lower():
                    column_renaming[col] = "Departure_Time"
                elif 'arrivl_time' in col.lower():
                    column_renaming[col] = "Arrival_Time"
                elif 'booking_cd' in col.lower():
                    column_renaming[col] = "Booking_Code"
                elif 'seat_no' in col.lower():
                    column_renaming[col] = "Seat_Number"
                elif 'loyalty_pts' in col.lower():
                    column_renaming[col] = "Loyalty_Points"
                elif 'duration_hrs' in col.lower():
                    column_renaming[col] = "Flight_Duration_Hours"
                elif col.replace('_', ' ').replace('#', '_Number'):
                    # Generic improvement: replace underscores and # symbols
                    improved_name = col.replace('_', ' ').replace('#', '_Number').title().replace(' ', '_')
                    if improved_name != col:
                        column_renaming[col] = improved_name
        
        if column_renaming:
            self.log_info(f"   💡 Generated {len(column_renaming)} column renaming suggestions")
        else:
            self.log_info("   ✅ Column names appear business-friendly")
        
        # Generate recommendations
        recommendations = []
        if issues:
            recommendations.extend([
                'Implement data validation rules for consistency checks',
                'Add business logic validation before data entry',
                'Set up automated data quality monitoring'
            ])
        
        if column_renaming:
            recommendations.append('Consider renaming technical columns to business-friendly names')
            recommendations.append('Use descriptive column names for better user understanding')
        
        if not issues and not column_renaming:
            recommendations = ['Data quality and column naming appear well-structured']
        
        result = {
            'total_issues': len(issues),
            'issues': issues,
            'column_renaming': column_renaming,
            'recommendations': recommendations
        }
        
        self.log_info(f"✅ Mock analysis complete - found {len(issues)} total issues")
        
        # Save mock analysis result
        self.save_data_to_file(
            result,
            f"mock_analysis_result_{timestamp_str}.json",
            "Mock analysis result"
        )
        
        # Save detailed analysis log
        analysis_log = {
            "method": "mock_analysis",
            "timestamp": datetime.now().isoformat(),
            "input_data_shape": sample_data.shape,
            "columns_analyzed": list(column_info.keys()),
            "issues_found": len(issues),
            "issue_categories": [issue['category'] for issue in issues],
            "recommendations_count": len(recommendations)
        }
        
        self.save_data_to_file(
            analysis_log,
            f"analysis_summary_{timestamp_str}.json",
            "Analysis summary"
        )
        
        return result
