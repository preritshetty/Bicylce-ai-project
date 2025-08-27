import pandas as pd
import json
import os
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException



# ===============================
# Pydantic models for structured output
# ===============================
class DataIssue(BaseModel):
    category: str = Field(description="Category of the issue")
    description: str = Field(description="Detailed description of the issue")
    severity: str = Field(description="Severity level: high, medium, or low")
    affected_columns: List[str] = Field(description="List of affected column names")
    fix_approach: str = Field(description="Suggested approach to fix the issue")


class AnalysisResult(BaseModel):
    total_issues: int = Field(description="Total number of issues found")
    issues: List[DataIssue] = Field(description="List of identified issues")
    column_renaming: Dict[str, str] = Field(description="Column renaming suggestions")
    recommendations: List[str] = Field(description="General recommendations")
    data_quality_score: Optional[int] = Field(default=85, description="Overall data quality score from 1-100")


@dataclass
class LLMConfig:
    """Configuration for LLM interface"""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1500
    request_timeout: int = 30


# ===============================
# Main Interface
# ===============================
class LLMInterfaceLangChain:
    """LangChain-based LLM interface for data quality analysis"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.logger = None
        self.setup_logging()

        utils_env_path = os.path.join(os.path.dirname(__file__), '.env')
        root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(utils_env_path):
            load_dotenv(utils_env_path)
        elif os.path.exists(root_env_path):
            load_dotenv(root_env_path)
        else:
            print("Warning: No .env file found in utils or root directory")

        self.has_api_key = self._check_api_key()
        self.llm = None
        self.analysis_chain = None

        if not self.has_api_key:
            raise ValueError("OpenAI API key is required. Please check your .env file in the utils folder.")

        self._setup_langchain()

    # ---------------- API Key ----------------
    def _check_api_key(self) -> bool:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key.strip() == "":
            try:
                self._safe_log("❌ No OpenAI API key found in environment variables", "error")
            except:
                print("❌ No OpenAI API key found in environment variables")
            return False
        if not api_key.startswith('sk-'):
            try:
                self._safe_log("❌ Invalid OpenAI API key format", "error")
            except:
                print("❌ Invalid OpenAI API key format")
            return False
        try:
            self._safe_log("✅ Valid OpenAI API key found")
        except:
            print("✅ Valid OpenAI API key found")
        return True

    # ---------------- Logging ----------------
    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"llm_langchain_analysis_{timestamp}.log"
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", log_filename)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger = logging.getLogger(f'LLMInterfaceLangChain_{timestamp}')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
        self.logger.info("🚀 LangChain LLM Interface initialized")
        self.logger.info(f"📁 Log file: {log_path}")

    def _safe_log(self, message: str, level: str = "info"):
        if not hasattr(self, 'logger') or self.logger is None:
            print(f"[{level.upper()}] {message}")
            return
        emoji_map = {
            '🚀': '[ROCKET]', '📁': '[FOLDER]', '🤖': '[ROBOT]',
            '✅': '[CHECK]', '❌': '[X]', '⚠️': '[WARNING]',
            '🔍': '[SEARCH]', '📊': '[CHART]', '💡': '[BULB]',
            '🔧': '[WRENCH]', '📋': '[CLIPBOARD]'
        }
        safe_message = message
        for emoji, text in emoji_map.items():
            safe_message = safe_message.replace(emoji, text)
        safe_message = re.sub(r'[^\x00-\x7F]+', '[EMOJI]', safe_message)
        getattr(self.logger, level)(safe_message)

    # ---------------- Setup LangChain ----------------
    def _setup_langchain(self):
        try:
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                request_timeout=self.config.request_timeout
            )
            self.output_parser = PydanticOutputParser(pydantic_object=AnalysisResult)
            self.prompt_template = self._create_prompt_template()
            self.analysis_chain = self.prompt_template | self.llm | self.output_parser
            self._safe_log("LangChain components initialized successfully")
        except Exception as e:
            self._safe_log(f"Error setting up LangChain: {str(e)}", "error")
            self.has_api_key = False

    # ---------------- Prompt ----------------
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the analysis prompt template using LangChain"""

        system_message = (
            "You are a data quality expert who analyzes datasets for business logic violations, "
            "data consistency issues, and suggests improvements including business-friendly column names. "
            "Adapt your analysis to the data domain while focusing on realistic, detectable issues. "
            "Always return valid JSON only."
        )

        human_message = """
Analyze this dataset for data quality issues, business logic problems, and suggest improvements.

DATASET INFO:
- Sample Rows: {total_rows}
- Total Columns: {column_count}

COLUMN DETAILS:
{column_info}

SAMPLE DATA (first 3 rows):
{data_sample}

ANALYSIS REQUIREMENTS:

1. DATA CONSISTENCY
   - Detect contradictory values in the same row
   - Impossible combinations (e.g., window_seat=True AND aisle_seat=True)
   - Status vs operational data conflicts

2. BUSINESS LOGIC VIOLATIONS
   - Date/time inconsistencies (e.g., departure after arrival)
   - Negative or impossible numeric values
   - Required fields missing

3. DATA QUALITY ISSUES
   - Missing critical information
   - Inconsistent formats
   - Obvious outliers that look like data entry errors

4. COLUMN NAMING
   - Provide a mapping of ALL columns to business-friendly names
   - Unchanged ones must still map to themselves

5. CROSS-FIELD RELATIONSHIP VALIDATION (MANDATORY)

   Step 1: Identify Numeric Count/Quantity Columns
   - Detect numeric columns whose names suggest a count/quantity 
     (keywords: count, number, num, qty, total, size, length).

   Step 2: Identify Potential List/Text Columns
   - Detect text/string columns that contain comma/pipe/semicolon-separated values, arrays, or lists.
   - Detect columns with pluralized names or keywords like list, items, names, details, locations.

   Step 3: Pairing Analysis
   - Pair each numeric "count" column with semantically related list/text columns
     (shared roots, stems, or keywords).
   - Examples:
       * passengers (numeric) ↔ passenger_names (text)
       * items_count (numeric) ↔ items_list (text)
       * layovers (numeric) ↔ layover_locations (text)

   Step 4: Validation Rules
   - Compare numeric counts vs number of tokens in the list column.
   - Flag HIGH SEVERITY issues when mismatches are detected.
   - Apply this rule for all detected pairs.

   Step 5: Other Relationship Checks
   - Status fields vs operational data (e.g., status="cancelled" but payment_date exists).
   - Start/end dates must follow chronological order.
   - Duration fields must equal end_time - start_time where present.
   - Mutually exclusive boolean fields must not both be true.
   - Total/average/percentage fields must align with their components.

   IMPORTANT:
   - Use ONLY actual columns provided — never invent new fields.
   - Skip silently if no valid pair exists.
   - Report only programmatically verifiable issues.
   CRITICAL CODE GENERATION RULES:
- NEVER use direct comparisons like df['col'] < 0
- ALWAYS use: pd.to_numeric(df['col'], errors='coerce') < 0
- ALWAYS combine with .notna() checks
- Wrap complex operations in try-except blocks
- Always return a boolean Series of same length as df


OUTPUT REQUIREMENT:
Return ONLY valid JSON with:
- total_issues (integer)
- issues (list of objects with: category, severity, description, affected_columns, fix_approach)
- column_renaming (mapping of ALL columns, even unchanged ones)
- recommendations (list of improvements)
- data_quality_score (1–100)
"""


        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])


    # ---------------- Run Analysis ----------------
    def analyze_data_quality(self, df: pd.DataFrame, column_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._safe_log("Starting LangChain data quality analysis with OpenAI API")
            analysis_data = self._prepare_analysis_data(df, column_info)
            self._safe_log(f"Input data prepared: {len(df)} rows, {len(df.columns)} columns")
            self._safe_log("🌐 Sending request to OpenAI via LangChain...")
            result = self.analysis_chain.invoke({
                "total_rows": len(df),
                "column_count": len(df.columns),
                "column_info": analysis_data["column_info_json"],
                "data_sample": analysis_data["sample_rows_json"]
            })
            result_dict = result.model_dump()

            # Ensure score
            if 'data_quality_score' not in result_dict or result_dict['data_quality_score'] is None:
                total_issues = result_dict.get('total_issues', 0)
                result_dict['data_quality_score'] = max(100 - (total_issues * 15), 10)

            # Ensure column renaming covers all columns
            for col in df.columns:
                if col not in result_dict["column_renaming"]:
                    result_dict["column_renaming"][col] = col

            self._safe_log("✅ LangChain analysis completed successfully")
            self._safe_log(f"Found {result_dict['total_issues']} issues")
            self._safe_log(f"Data quality score: {result_dict['data_quality_score']}")
            return result_dict
        except OutputParserException as e:
            self._safe_log(f"❌ Output parsing error: {str(e)}", "error")
            raise Exception(f"Failed to parse LLM response: {str(e)}")
        except Exception as e:
            self._safe_log(f"❌ LangChain analysis error: {str(e)}", "error")
            raise Exception(f"Analysis failed: {str(e)}")

    # ---------------- Data Prep ----------------
    def _prepare_analysis_data(self, df: pd.DataFrame, column_info: Dict[str, Any]) -> Dict[str, Any]:
        column_info_json = json.dumps(column_info, indent=2)
        sample_rows = df.head(3).to_dict('records')
        sample_rows_json = json.dumps(sample_rows, indent=2)
        return {
            "column_info_json": column_info_json,
            "sample_rows_json": sample_rows_json
        }
