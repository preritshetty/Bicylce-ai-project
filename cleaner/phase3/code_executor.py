"""
Code Execution Engine for Phase 3 Detection Codes

This executor takes the JSON output from code_generator and executes the detection codes
on the actual flight booking data to flag quality issues.
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
import numpy as np
import re

class CodeExecutor:
    """Executes AI-generated detection codes on flight booking data"""
    
    def __init__(self, data_path: str = "phase outputs/phase1_cleaned_data.csv"):
        self.data_path = data_path
        self.data = None
        self.execution_log = []
        self.results = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup execution logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/code_executor_{timestamp}.log"
        
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Code Executor initialized")
    
    def load_data(self) -> bool:
        """Load the cleaned flight booking data"""
        try:
            if not os.path.exists(self.data_path):
                self.logger.error(f"Data file not found: {self.data_path}")
                return False
            
            self.data = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded data: {len(self.data)} rows, {len(self.data.columns)} columns")
            
            # Initialize flag_status column
            if 'flag_status' not in self.data.columns:
                self.data['flag_status'] = 0
                self.logger.info("Added flag_status column, initialized to 0")
            else:
                self.data['flag_status'] = 0
                self.logger.info("Reset flag_status column to 0")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
    
    def load_detection_codes(self, codes_file: str) -> List[Dict[str, Any]]:
        """Load detection codes from code_generator output"""
        try:
            with open(codes_file, 'r') as f:
                codes_data = json.load(f)
            
            # Extract detection codes from the code_generator format
            detection_codes = codes_data.get('detection_codes', [])
            
            self.logger.info(f"Loaded {len(detection_codes)} detection codes from {codes_file}")
            
            # Log the loaded codes
            for code in detection_codes:
                self.logger.info(f"  - Flag {code['flag_value']}: {code['explanation'][:50]}...")
            
            return detection_codes
            
        except Exception as e:
            self.logger.error(f"Error loading detection codes: {str(e)}")
            raise
    
    def execute_detection_code(self, detection_code: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single detection code on the data"""
        flag_value = detection_code['flag_value']
        code = detection_code['detection_code']
        explanation = detection_code['explanation']
        
        initial_flagged = (self.data['flag_status'] & flag_value).sum() // flag_value
        
        try:
            self.logger.info(f"Executing Flag {flag_value}: {explanation[:60]}...")
            
            # ✅ Expose pandas, regex, and numpy
            exec_globals = {
                'df': self.data,
                'pd': pd,
                're': re,
                'np': np
            }
            
            exec(code, exec_globals)
            
            final_flagged = (self.data['flag_status'] & flag_value).sum() // flag_value
            rows_detected = final_flagged - initial_flagged
            
            result = {
                'flag_value': flag_value,
                'success': True,
                'rows_detected': rows_detected,
                'explanation': explanation,
                'code': code,
                'error': None
            }
            
            self.logger.info(f"  ✅ Success: {rows_detected} rows flagged with Flag {flag_value}")
        
        except Exception as e:
            error_msg = str(e)
            result = {
                'flag_value': flag_value,
                'success': False,
                'rows_detected': 0,
                'explanation': explanation,
                'code': code,
                'error': error_msg
            }
            self.logger.error(f"  ❌ Failed: {error_msg}")
        
        self.execution_log.append(result)
        return result

    
    def execute_all_codes(self, codes_file: str) -> Dict[str, Any]:
        """Execute all detection codes from code_generator output"""
        start_time = datetime.now()
        
        # Load data and detection codes
        if not self.load_data():
            raise RuntimeError("Failed to load data")
        
        detection_codes = self.load_detection_codes(codes_file)
        
        if not detection_codes:
            raise RuntimeError("No detection codes found")
        
        self.logger.info(f"Starting execution of {len(detection_codes)} detection codes")
        
        # Execute each detection code
        execution_results = []
        successful_executions = 0
        failed_executions = 0
        total_detections = 0
        
        for detection_code in detection_codes:
            result = self.execute_detection_code(detection_code)
            execution_results.append(result)
            
            if result['success']:
                successful_executions += 1
                total_detections += result['rows_detected']
            else:
                failed_executions += 1
        
        # Calculate final statistics
        execution_time = (datetime.now() - start_time).total_seconds()
        total_rows = len(self.data)
        unique_flagged_rows = (self.data['flag_status'] > 0).sum()
        clean_rows = total_rows - unique_flagged_rows
        
        # Create flag breakdown
        flag_breakdown = self._analyze_flag_combinations()
        
        # Compile results
        self.results = {
            'execution_summary': {
                'total_rows': total_rows,
                'flagged_rows': unique_flagged_rows,
                'clean_rows': clean_rows,
                'flagged_percentage': round(unique_flagged_rows / total_rows * 100, 2),
                'total_detections': total_detections,
                'execution_time_seconds': round(execution_time, 2),
                'successful_codes': successful_executions,
                'failed_codes': failed_executions,
                'success_rate': round(successful_executions / len(detection_codes) * 100, 2),
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': execution_results,
            'flag_breakdown': flag_breakdown
        }
        
        self.logger.info(f"Execution completed: {unique_flagged_rows}/{total_rows} rows flagged")
        self.logger.info(f"Success rate: {successful_executions}/{len(detection_codes)} codes")
        
        return self.results
    
    def _analyze_flag_combinations(self) -> Dict[str, Any]:
        """Analyze flag combinations in the data"""
        flag_analysis = {}
        
        # Count individual flags
        individual_flags = {}
        for flag_value in [1, 2, 4, 8, 16, 32, 64, 128]:
            count = (self.data['flag_status'] & flag_value).sum() // flag_value
            if count > 0:
                individual_flags[flag_value] = count
        
        # Count combined flags
        combined_flags = {}
        for flag_status in self.data['flag_status'].unique():
            if flag_status > 0:
                count = (self.data['flag_status'] == flag_status).sum()
                combined_flags[flag_status] = count
        
        flag_analysis = {
            'individual_flags': individual_flags,
            'combined_flags': combined_flags,
            'flag_combinations': self._get_flag_combinations_description(combined_flags)
        }
        
        return flag_analysis
    
    def _get_flag_combinations_description(self, combined_flags: Dict[int, int]) -> List[Dict[str, Any]]:
        """Convert flag combinations to human-readable descriptions"""
        combinations = []
        
        for flag_status, count in combined_flags.items():
            # Decompose flag status into individual flags
            individual_flags = []
            temp_status = flag_status
            power = 1
            
            while temp_status > 0:
                if temp_status & 1:
                    individual_flags.append(power)
                temp_status >>= 1
                power <<= 1
            
            combinations.append({
                'flag_status': flag_status,
                'individual_flags': individual_flags,
                'flag_description': f"Flags {'+'.join(map(str, individual_flags))}",
                'row_count': count,
                'binary_representation': bin(flag_status)[2:]
            })
        
        return combinations
    
    def save_flagged_data(self, output_path: str = None) -> str:
        """Save the flagged data to CSV"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"phase outputs/flagged_data_{timestamp}.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the data with flags
        self.data.to_csv(output_path, index=False)
        
        self.logger.info(f"Flagged data saved to: {output_path}")
        return output_path
    
    def save_execution_report(self, output_path: str = None) -> str:
        """Save execution results to JSON"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"phase outputs/execution_report_{timestamp}.json"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to JSON-safe format
        json_safe_results = self._convert_to_json_safe(self.results)
        
        # Save the report
        with open(output_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        self.logger.info(f"Execution report saved to: {output_path}")
        return output_path
    
    def _convert_to_json_safe(self, obj):
        """Convert numpy/pandas types to JSON-safe types"""
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy/pandas scalar
            return obj.item()
        elif isinstance(obj, dict):
            # Convert both keys and values to JSON-safe types
            return {self._convert_to_json_safe(key): self._convert_to_json_safe(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            # For other types, try to convert to basic Python types
            try:
                return str(obj) if not isinstance(obj, (str, int, float, bool)) else obj
            except:
                return str(obj)
    
    def get_flagged_rows_sample(self, n: int = 10) -> pd.DataFrame:
        """Get a sample of flagged rows for inspection"""
        if self.data is None:
            return pd.DataFrame()
        
        flagged_rows = self.data[self.data['flag_status'] > 0]
        
        if len(flagged_rows) == 0:
            return pd.DataFrame()
        
        # Add flag breakdown column
        flagged_rows = flagged_rows.copy()
        flagged_rows['flag_breakdown'] = flagged_rows['flag_status'].apply(
            lambda x: self._describe_flag_status(x)
        )
        
        return flagged_rows.head(n)
    
    def _describe_flag_status(self, flag_status: int) -> str:
        """Convert flag status to human-readable description"""
        if flag_status == 0:
            return "No flags"
        
        flags = []
        power = 1
        temp_status = flag_status
        
        while temp_status > 0:
            if temp_status & 1:
                flags.append(str(power))
            temp_status >>= 1
            power <<= 1
        
        return f"Flags: {'+'.join(flags)}"
    
    def run_complete_pipeline(self, detection_codes_file: str) -> Dict[str, Any]:
        """Run the complete execution pipeline"""
        try:
            # Execute all detection codes
            results = self.execute_all_codes(detection_codes_file)
            
            # Save outputs
            flagged_data_path = self.save_flagged_data()
            execution_report_path = self.save_execution_report()
            
            # Add file paths to results
            results['output_files'] = {
                'flagged_data': flagged_data_path,
                'execution_report': execution_report_path
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise


def main():
    """Test the code executor"""
    try:
        # Find the latest detection codes file
        detection_files = [f for f in os.listdir("phase outputs") 
                          if f.startswith("detection_codes_") and f.endswith(".json")]
        
        if not detection_files:
            print("❌ No detection codes found. Please run code_generator first.")
            return
        
        latest_codes_file = sorted(detection_files, reverse=True)[0]
        codes_path = os.path.join("phase outputs", latest_codes_file)
        
        print(f"🚀 Running Code Executor")
        print(f"📁 Using detection codes: {latest_codes_file}")
        
        # Initialize and run executor
        executor = CodeExecutor()
        results = executor.run_complete_pipeline(codes_path)
        
        # Display results
        summary = results['execution_summary']
        print(f"\n📊 Execution Results:")
        print(f"  • Total rows: {summary['total_rows']:,}")
        print(f"  • Flagged rows: {summary['flagged_rows']:,} ({summary['flagged_percentage']}%)")
        print(f"  • Clean rows: {summary['clean_rows']:,}")
        print(f"  • Execution time: {summary['execution_time_seconds']} seconds")
        print(f"  • Success rate: {summary['success_rate']}%")
        
        print(f"\n📁 Output files:")
        print(f"  • Flagged data: {results['output_files']['flagged_data']}")
        print(f"  • Execution report: {results['output_files']['execution_report']}")
        
        print(f"\n✅ Code execution completed successfully!")
        
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")


if __name__ == "__main__":
    main()
