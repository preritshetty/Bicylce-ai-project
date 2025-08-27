"""
flag_mapper.py

Converts Phase 2 analysis issues into binary flag system for efficient tracking.
Each issue gets assigned a unique binary flag (1, 2, 4, 8, 16, 32, etc.)
Creates human-readable mapping and saves to JSON file.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class IssueFlag:
    """Represents a single issue flag with metadata"""
    flag_value: int
    description: str
    category: str
    severity: str
    affected_columns: List[str]
    fix_approach: str


class FlagMapper:
    """Maps Phase 2 issues to binary flags for efficient tracking"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.flag_mapping = {}
        self.issue_flags = []
        self.next_flag_value = 1
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_flag_mapping(self, phase2_issues: List[Dict[str, Any]]) -> Dict[int, IssueFlag]:
        """
        Create binary flag mapping from Phase 2 issues
        
        Args:
            phase2_issues: List of issues from Phase 2 analysis
            
        Returns:
            Dictionary mapping flag values to IssueFlag objects
        """
        print(f"🏁 Creating flag mapping for {len(phase2_issues)} issues...")
        
        # Reset mapping
        self.flag_mapping = {}
        self.issue_flags = []
        self.next_flag_value = 1
        
        for i, issue in enumerate(phase2_issues):
            # Create flag for this issue
            flag_value = self._get_next_flag_value()
            
            issue_flag = IssueFlag(
                flag_value=flag_value,
                description=issue.get('description', f'Issue {i+1}'),
                category=issue.get('category', 'unknown'),
                severity=issue.get('severity', 'medium'),
                affected_columns=issue.get('affected_columns', []),
                fix_approach=issue.get('fix_approach', 'Manual review required')
            )
            
            self.issue_flags.append(issue_flag)
            self.flag_mapping[flag_value] = issue_flag
            
            print(f"  📌 Flag {flag_value}: {issue_flag.description[:60]}...")
        
        print(f"✅ Flag mapping created with {len(self.flag_mapping)} flags")
        return self.flag_mapping
    
    def save_mapping(self, flag_mapping: Dict[int, IssueFlag], output_dir: str = "outputs") -> str:
        """
        Save flag mapping to JSON file
        
        Args:
            flag_mapping: The flag mapping dictionary
            output_dir: Directory to save the mapping file
            
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert flag mapping to JSON-serializable format
        json_mapping = {
            'flags': {},
            'metadata': {
                'total_issues': len(flag_mapping),
                'created_at': datetime.now().isoformat(),
                'description': f'Binary flag mapping for {len(flag_mapping)} data quality issues',
                'flag_values': list(flag_mapping.keys()),
                'max_combined_value': sum(flag_mapping.keys())
            }
        }
        
        # Convert each IssueFlag to dictionary
        for flag_value, issue_flag in flag_mapping.items():
            json_mapping['flags'][str(flag_value)] = {
                'flag_value': issue_flag.flag_value,
                'category': issue_flag.category,
                'description': issue_flag.description,
                'severity': issue_flag.severity,
                'affected_columns': issue_flag.affected_columns,
                'fix_approach': issue_flag.fix_approach,
                'binary_representation': f"{issue_flag.flag_value:08b}"
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flag_mapping_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(json_mapping, f, indent=2)
        
        print(f"💾 Flag mapping saved to: {filepath}")
        return filepath
    
    def _get_next_flag_value(self) -> int:
        """Get next binary flag value (1, 2, 4, 8, 16, 32, ...)"""
        current_value = self.next_flag_value
        self.next_flag_value *= 2
        return current_value
    
    def save_mapping_to_file(self, filename: Optional[str] = None) -> str:
        """
        Save flag mapping to JSON file
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flag_mapping_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to serializable format
        mapping_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_flags": len(self.flag_mapping),
                "max_flag_value": max(self.flag_mapping.keys()) if self.flag_mapping else 0,
                "description": "Binary flag mapping for Phase 2 data quality issues"
            },
            "flag_mapping": {},
            "issue_details": {}
        }
        
        # Add flag mappings
        for flag_value, issue_flag in self.flag_mapping.items():
            # Simple mapping for quick lookup
            mapping_data["flag_mapping"][str(flag_value)] = issue_flag.description
            
            # Detailed mapping with all metadata
            mapping_data["issue_details"][str(flag_value)] = {
                "description": issue_flag.description,
                "category": issue_flag.category,
                "severity": issue_flag.severity,
                "affected_columns": issue_flag.affected_columns,
                "fix_approach": issue_flag.fix_approach
            }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Flag mapping saved to: {filepath}")
        return filepath
    
    def load_mapping_from_file(self, filepath: str) -> Dict[int, IssueFlag]:
        """
        Load flag mapping from JSON file
        
        Args:
            filepath: Path to the mapping file
            
        Returns:
            Dictionary mapping flag values to IssueFlag objects
        """
        print(f"📂 Loading flag mapping from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # Reconstruct flag mapping
        self.flag_mapping = {}
        self.issue_flags = []
        
        issue_details = mapping_data.get("issue_details", {})
        
        for flag_str, details in issue_details.items():
            flag_value = int(flag_str)
            
            issue_flag = IssueFlag(
                flag_value=flag_value,
                description=details.get('description', ''),
                category=details.get('category', 'unknown'),
                severity=details.get('severity', 'medium'),
                affected_columns=details.get('affected_columns', []),
                fix_approach=details.get('fix_approach', '')
            )
            
            self.flag_mapping[flag_value] = issue_flag
            self.issue_flags.append(issue_flag)
        
        print(f"✅ Loaded {len(self.flag_mapping)} flag mappings")
        return self.flag_mapping
    
    def get_flag_combinations(self) -> Dict[int, List[str]]:
        """
        Get all possible flag combinations and their descriptions
        
        Returns:
            Dictionary mapping combined flag values to list of issue descriptions
        """
        if not self.flag_mapping:
            return {}
        
        combinations = {}
        max_flags = len(self.flag_mapping)
        
        # Generate all possible combinations (up to reasonable limit)
        max_combinations = min(2 ** max_flags, 256)  # Limit to prevent explosion
        
        for combo_value in range(1, max_combinations):
            issues = []
            for flag_value, issue_flag in self.flag_mapping.items():
                if combo_value & flag_value:  # Check if this flag is set
                    issues.append(issue_flag.description)
            
            if issues:
                combinations[combo_value] = issues
        
        return combinations
    
    def decode_flag_status(self, flag_status: int) -> List[IssueFlag]:
        """
        Decode a combined flag status into individual issues
        
        Args:
            flag_status: Combined flag value from dataset
            
        Returns:
            List of IssueFlag objects that are present
        """
        present_issues = []
        
        for flag_value, issue_flag in self.flag_mapping.items():
            if flag_status & flag_value:  # Check if this flag is set
                present_issues.append(issue_flag)
        
        return present_issues
    
    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of current flag mapping"""
        if not self.flag_mapping:
            return {"status": "No mapping created"}
        
        severity_counts = {}
        category_counts = {}
        
        for issue_flag in self.issue_flags:
            # Count by severity
            severity = issue_flag.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by category
            category = issue_flag.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_flags": len(self.flag_mapping),
            "flag_values": list(self.flag_mapping.keys()),
            "max_flag_value": max(self.flag_mapping.keys()),
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "possible_combinations": 2 ** len(self.flag_mapping) - 1
        }


def main():
    """Example usage of FlagMapper"""
    # Example Phase 2 issues (like from your JSON file)
    example_issues = [
        {
            "category": "consistency",
            "description": "Contradictory information in the same row: window_seat=True AND aisle_seat=True",
            "severity": "medium",
            "affected_columns": ["window_seat", "aisle_seat"],
            "fix_approach": "Review and correct the seating information to ensure consistency"
        },
        {
            "category": "business_logic",
            "description": "Departure date is after arrival date",
            "severity": "high",
            "affected_columns": ["departure_dt", "arrival_dt"],
            "fix_approach": "Investigate and correct the departure and arrival date entries"
        },
        {
            "category": "data_quality",
            "description": "Missing loyalty points for a loyalty program member",
            "severity": "medium",
            "affected_columns": ["loyalty_pts", "reward_program_member"],
            "fix_approach": "Ensure loyalty points are provided for loyalty program members"
        },
        {
            "category": "relationships",
            "description": "Mismatch between number of stops and layover locations",
            "severity": "low",
            "affected_columns": ["number_of_stops", "layover_locations"],
            "fix_approach": "Ensure consistency between the number of stops and layover locations"
        }
    ]
    
    # Create flag mapper
    mapper = FlagMapper()
    
    # Create mapping
    flag_mapping = mapper.create_flag_mapping(example_issues)
    
    # Save to file
    filepath = mapper.save_mapping_to_file()
    
    # Show summary
    summary = mapper.get_mapping_summary()
    print(f"\n📊 Flag Mapping Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Show some combinations
    print(f"\n🔢 Example Flag Combinations:")
    combinations = mapper.get_flag_combinations()
    for combo_value in [1, 2, 3, 5, 15]:  # Show a few examples
        if combo_value in combinations:
            print(f"  Flag {combo_value}: {combinations[combo_value]}")


if __name__ == "__main__":
    main()
