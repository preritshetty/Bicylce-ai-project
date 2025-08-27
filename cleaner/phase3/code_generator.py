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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException

# Import flag mapper for type hints
try:
    from .flag_mapper import IssueFlag
except ImportError:
    try:
        from flag_mapper import IssueFlag
    except ImportError:
        # Fallback - define a simple placeholder
        class IssueFlag:
            def __init__(self, flag_value, description, category="unknown"):
                self.flag_value = flag_value
                self.description = description
                self.category = category


# =========================
# Detection code guidelines
# =========================
CODE_GENERATION_GUIDELINES = """
CRITICAL RULES FOR DETECTION CODE:

1. BOOLEAN HANDLING:
   - Always normalize:
     col_bool = df['col'].astype(str).str.strip().str.lower().isin(['true','t','yes','y','1'])
   - For contradictory booleans: (col1_bool & col2_bool)

2. NUMERIC HANDLING:
   - Convert with: pd.to_numeric(df['col'], errors='coerce')
   - Negative check: (numeric < 0) & numeric.notna()

3. DATE HANDLING:
   - Convert with: pd.to_datetime(df['col'], errors='coerce')
   - Rule: (start > end) & start.notna() & end.notna()

4. COUNT vs LIST CHECK (GENERALIZED, SAFE, and MANDATORY):
   - Count column must be converted safely:
       count_field = pd.to_numeric(df['<exact_count_column_name>'], errors='coerce').fillna(0).astype(int)

   - List column must be normalized:
       list_field = df['<exact_list_column_name>'].fillna('').astype(str)

   - Use a safe counting function with multiple delimiters:
       def safe_count_items(x):
           try:
               if not x or str(x).strip() == '':
                   return 0
               return len([item.strip() for item in re.split(r'[;,|>]|->', str(x)) if item.strip()])
           except Exception:
               return 0

       actual_count = list_field.apply(safe_count_items)

   - Detection rule (must use the real column names exactly as in `Affected Columns`):
       df.loc[(count_field != actual_count) & count_field.notna(), 'flag_status'] |= {flag_value}

   - CRITICAL: Never use placeholder names like count_col or list_col.
     Always use the *actual* column names provided in Affected Columns.

5. MISSING/EMPTY FIELD:
   - Rule: df['col'].isna() | (df['col'].astype(str).str.strip() == '')

6. SAFE OUTPUT (STRICT INLINE RULES):
   - ALWAYS return exactly ONE `df.loc[...] |= {flag_value}` statement.
   - NEVER define variables like `count_field`, `list_field`, `actual_count`, etc.
   - Inline everything in the condition:
       ✅ Example:
       df.loc[
         (pd.to_numeric(df['Layovers'], errors='coerce').fillna(0).astype(int) 
          != df['Layover_Locations'].fillna('').astype(str)
                .apply(lambda x: len([i for i in re.split(r'[;,|>]|->', str(x)) if i.strip()])))
         , 'flag_status'
       ] |= {flag_value}
   - Use **only** the exact column names from `Affected Columns`.
   - No placeholders (`count_field`, `list_field`) are allowed.




7. ENVIRONMENT:
   - Assume `import re` is already available in the execution environment.
   - Do not add imports inside the generated code.
"""



# Pydantic models for structured output
class DetectionCode(BaseModel):
    flag_value: int = Field(description="The binary flag value (1, 2, 4, 8, etc.)")
    detection_code: str = Field(description="Python pandas code to detect the issue")
    explanation: str = Field(description="Human-readable explanation of what the code does")
    test_description: str = Field(description="Description of how to test this detection")
    safety_notes: str = Field(description="Any safety considerations for this code")


class CodeGenerationResult(BaseModel):
    total_codes: int = Field(description="Total number of detection codes generated")
    detection_codes: List[DetectionCode] = Field(description="List of generated detection codes")
    generation_timestamp: str = Field(description="When the codes were generated")
    llm_model: str = Field(description="Which LLM model was used")


@dataclass
class CodeGenConfig:
    """Configuration for code generation"""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    request_timeout: int = 30


class CodeGenerator:
    """LangChain-powered code generator for data issue detection"""
    
    def __init__(self, config: Optional[CodeGenConfig] = None):
        self.config = config or CodeGenConfig()
        
        # Initialize logger
        self.logger = None
        self.setup_logging()
        
        # Load environment variables
        self._load_env()
        
        self.has_api_key = self._check_api_key()
        self.llm = None
        self.code_generation_chain = None
        
        if not self.has_api_key:
            raise ValueError("OpenAI API key is required for code generation. Please check your .env file.")
            
        self._setup_langchain()
    
    def _load_env(self):
        """Load environment variables from .env file"""
        utils_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', '.env')
        root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        
        if os.path.exists(utils_env_path):
            load_dotenv(utils_env_path)
            self._safe_log(f"Loaded environment from: {utils_env_path}")
        elif os.path.exists(root_env_path):
            load_dotenv(root_env_path)
            self._safe_log(f"Loaded environment from: {root_env_path}")
        else:
            self._safe_log("Warning: No .env file found", "warning")
    
    def _check_api_key(self) -> bool:
        """Check if OpenAI API key is available and valid"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key.strip() == "":
            self._safe_log("No OpenAI API key found in environment variables", "error")
            return False
        if not api_key.startswith('sk-'):
            self._safe_log("Invalid OpenAI API key format", "error")
            return False
        self._safe_log("Valid OpenAI API key found")
        return True
    
    def setup_logging(self):
        """Setup logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"code_generator_{timestamp}.log"
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", log_filename)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(f'CodeGenerator_{timestamp}')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
        
        self.logger.info("Code Generator initialized")
    
    def _safe_log(self, message: str, level: str = "info"):
        if not self.logger:
            print(f"[{level.upper()}] {message}")
            return
        safe_message = re.sub(r'[^\x00-\x7F]+', '[EMOJI]', message)
        getattr(self.logger, level)(safe_message)
    
    def _setup_langchain(self):
        """Setup LangChain components"""
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            request_timeout=self.config.request_timeout
        )
        self.output_parser = PydanticOutputParser(pydantic_object=DetectionCode)
        self.prompt_template = self._create_prompt_template()
        self.code_generation_chain = self.prompt_template | self.llm | self.output_parser
        self._safe_log("LangChain code generation components initialized successfully")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Prompt for code generation"""
        
        system_message = """You are an expert Python data engineer specializing in pandas data manipulation and quality detection. 
Your task is to generate safe, efficient pandas code to detect specific data quality issues."""

        human_message = """Generate pandas detection code for this data quality issue:

**Issue Details:**
- Flag Value: {flag_value}
- Category: {category}
- Description: {description}
- Severity: {severity}
- Affected Columns: {affected_columns}
- Fix Approach: {fix_approach}

**Sample Data Context:**
{sample_data_info}

MANDATORY RULES:
{CODE_GENERATION_GUIDELINES}
**IMPORTANT CODING RULES:**
- Use ONLY the exact column names listed in `Affected Columns`.
- Always reference them in pandas as: df['column_name'] (string literal, not a variable).
- NEVER invent placeholder names like count_col, list_col, start_col, etc.
- If multiple columns are listed, only use those exact names.
- The detection code must be a SINGLE valid Python statement.
- Always use exactly one opening '[' and one closing ']' for df.loc[…, 'flag_status'].
- Do not add extra brackets or unmatched parentheses.
- Final code must match the exact pattern:
  df.loc[(<boolean_condition>), 'flag_status'] |= {flag_value}

OUTPUT REQUIREMENT:
Return valid JSON:
{{
    "flag_value": {flag_value},
    "detection_code": "df.loc[<condition>, 'flag_status'] |= {flag_value}",
    "explanation": "what this code does",
    "test_description": "how to test it",
    "safety_notes": "safety considerations"
}}

{format_instructions}"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ]).partial(
            format_instructions=self.output_parser.get_format_instructions(),
            CODE_GENERATION_GUIDELINES=CODE_GENERATION_GUIDELINES
        )
    def _validate_detection_code(self, code: str, flag_value: int) -> bool:
            """
            Validate that generated code is safe, syntactically balanced, 
            and follows expected patterns.
            """
            try:
                # Forbid dangerous ops
                forbidden_patterns = [
                    'import ', 'exec(', 'eval(', '__import__',
                    'open(', 'os.', 'sys.', 'subprocess', 'shutil'
                ]
                for pattern in forbidden_patterns:
                    if pattern in code.lower():
                        self._safe_log(f"Validation failed: forbidden pattern '{pattern}'", "error")
                        return False

                # Bracket/parenthesis balance check
                if code.count('[') != code.count(']'):
                    self._safe_log("Validation failed: unbalanced square brackets", "error")
                    return False
                if code.count('(') != code.count(')'):
                    self._safe_log("Validation failed: unbalanced parentheses", "error")
                    return False

                # Ensure proper df.loc pattern
                expected_pattern = rf"df\.loc\[.*,\s*['\"]flag_status['\"]\s*\]\s*\|=\s*{flag_value}"
                if not re.search(expected_pattern, code):
                    self._safe_log(f"Validation failed: expected df.loc pattern missing for flag {flag_value}", "error")
                    return False

                return True
            except Exception as e:
                self._safe_log(f"Validation check error: {str(e)}", "error")
                return False

    def generate_detection_code(self, issue_flag: IssueFlag, sample_data_info: str = "") -> DetectionCode:
        try:
            self._safe_log(f"Generating detection code for flag {issue_flag.flag_value}: {issue_flag.description[:50]}...")
            chain_input = {
                "flag_value": issue_flag.flag_value,
                "category": issue_flag.category,
                "description": issue_flag.description,
                "severity": issue_flag.severity,
                "affected_columns": ", ".join(issue_flag.affected_columns),
                "fix_approach": issue_flag.fix_approach,
                "sample_data_info": sample_data_info or "Standard pandas DataFrame with the mentioned columns"
            }
            result = self.code_generation_chain.invoke(chain_input)
            self._safe_log(f"Generated code:\n{result.detection_code}")
            if self._validate_detection_code(result.detection_code, issue_flag.flag_value):
                return result
            else:
                return self._create_fallback_code(issue_flag)
        except Exception as e:
            self._safe_log(f"Code generation error for flag {issue_flag.flag_value}: {str(e)}", "error")
            return self._create_fallback_code(issue_flag)
    
        
    
    def _create_fallback_code(self, issue_flag: IssueFlag) -> DetectionCode:
        """
        Create fallback detection code if generation fails.
        For Cross-Field Relationship Validation, generate a real generalized check
        instead of a dummy placeholder.
        """
        if issue_flag.category.lower().strip() == "cross-field relationship validation":
            # Expecting exactly two columns: count and list
            if len(issue_flag.affected_columns) == 2:
                count_col, list_col = issue_flag.affected_columns
                detection_code = (
                    f"df.loc[(pd.to_numeric(df['{count_col}'], errors='coerce').fillna(0).astype(int) "
                    f"!= df['{list_col}'].fillna('').astype(str)"
                    f".apply(lambda x: len([i for i in re.split(r'[;,|>]|->', str(x)) if i.strip()])))"
                    f", 'flag_status'] |= {issue_flag.flag_value}"
                )
                return DetectionCode(
                    flag_value=issue_flag.flag_value,
                    detection_code=detection_code,
                    explanation=f"Checks for mismatch between {count_col} and {list_col}",
                    test_description=f"Create rows where {count_col} does not equal number of items in {list_col}",
                    safety_notes="Handles missing values, multiple delimiters (;,|,>,->)."
                )

        # Default fallback (dummy)
        fallback_code = f"# Fallback detection for flag {issue_flag.flag_value}\n"
        fallback_code += f"df.loc[df.index == -1, 'flag_status'] |= {issue_flag.flag_value}  # Placeholder"
        return DetectionCode(
            flag_value=issue_flag.flag_value,
            detection_code=fallback_code,
            explanation=f"Fallback for {issue_flag.description}",
            test_description="This selects no rows. Replace with proper logic.",
            safety_notes="Safe but non-functional."
        )

    
    def generate_all_detection_codes(self, flag_mapping: Dict[int, IssueFlag], sample_data_info: str = "") -> CodeGenerationResult:
        detection_codes = []
        for flag_value, issue_flag in flag_mapping.items():
            detection_code = self.generate_detection_code(issue_flag, sample_data_info)
            detection_codes.append(detection_code)
        return CodeGenerationResult(
            total_codes=len(detection_codes),
            detection_codes=detection_codes,
            generation_timestamp=datetime.now().isoformat(),
            llm_model=self.config.model_name
        )
    
    def save_detection_codes(self, generation_result: CodeGenerationResult, output_dir: str = "outputs") -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_codes_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(generation_result.model_dump(), f, indent=2)
        return filepath
