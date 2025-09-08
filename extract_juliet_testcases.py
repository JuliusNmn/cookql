#!/usr/bin/env python3
"""
Juliet Test Suite Testcase Extractor

This script analyzes the Juliet Test Suite structure and extracts information about
testcases, including their associated files and whether they are bad-only testcases.

Based on the Juliet Test Suite v1.2 for Java User Guide documentation.
"""

import os
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from pathlib import Path


@dataclass
class TestCase:
    """
    Represents a single testcase in the Juliet Test Suite.
    
    A testcase may consist of one or more Java files and can be one of several types:
    - Non-class-based flaw (most common)
    - Class-based flaw (separate _bad and _good1 files)
    - Abstract method (uses _base, _bad, _goodG2B, _goodB2G files)
    - Bad-only (contains only bad methods, no good implementations)
    """
    name: str  # Base testcase name without file extensions
    cwe_id: str  # CWE identifier (e.g., "CWE15")
    files: List[str] = field(default_factory=list)  # All files belonging to this testcase
    testcase_type: str = "non_class_based"  # Type: non_class_based, class_based, abstract_method, bad_only
    is_bad_only: bool = False  # Whether this is a bad-only testcase
    has_servlet: bool = False  # Whether this testcase involves servlets


@dataclass
class CWEGroup:
    """
    Represents a group of testcases for a specific CWE.
    """
    cwe_id: str
    testcases: List[TestCase] = field(default_factory=list)
    total_files: int = 0


class JulietTestSuiteAnalyzer:
    """
    Analyzes the Juliet Test Suite directory structure and extracts testcase information.
    """
    
    def __init__(self, juliet_path: str):
        self.juliet_path = Path(juliet_path)
        self.cwe_groups: Dict[str, CWEGroup] = {}
        
        # Regex patterns for different testcase types
        self.cwe_pattern = re.compile(r'juliet-cwe(\d+)')
        # More flexible pattern to match the actual file naming convention
        self.testcase_base_pattern = re.compile(r'^(CWE\d+_.+?)(?:_(\d+[a-z]?|bad|good\d*|goodG2B\d*|goodB2G\d*|base))?\.java$')
        
        # Known bad-only testcases from documentation (partial list)
        # Note: This should be expanded with the complete list from Appendix D
        self.known_bad_only_patterns = {
            'CWE561_',  # Dead Code
            'CWE563_',  # Assignment to Variable without Use
            'CWE570_',  # Expression is Always False
            'CWE571_',  # Expression is Always True
        }
    
    def analyze(self) -> Dict[str, CWEGroup]:
        """
        Main method to analyze the Juliet test suite structure.
        """
        if not self.juliet_path.exists():
            raise FileNotFoundError(f"Juliet path does not exist: {self.juliet_path}")
        
        # Find all CWE directories
        cwe_dirs = [d for d in self.juliet_path.iterdir() 
                   if d.is_dir() and d.name.startswith('juliet-cwe')]
        
        for cwe_dir in sorted(cwe_dirs):
            self._analyze_cwe_directory(cwe_dir)
        
        return self.cwe_groups
    
    def _analyze_cwe_directory(self, cwe_dir: Path):
        """
        Analyze a single CWE directory and extract testcases.
        """
        # Extract CWE ID from directory name
        match = self.cwe_pattern.search(cwe_dir.name)
        if not match:
            return
        
        cwe_id = f"CWE{match.group(1)}"
        
        # Find the testcases directory structure
        testcases_dir = cwe_dir / "src" / "main" / "java" / "juliet" / "testcases"
        if not testcases_dir.exists():
            return
        
        # Group files by testcase
        testcase_files = defaultdict(list)
        
        # Walk through all Java files in the testcases directory
        for java_file in testcases_dir.rglob("*.java"):
            relative_path = str(java_file.relative_to(self.juliet_path))
            testcase_name = self._extract_testcase_name(java_file.name)
            if testcase_name:
                testcase_files[testcase_name].append(relative_path)
        
        # Create TestCase objects
        cwe_group = CWEGroup(cwe_id=cwe_id)
        
        for testcase_name, files in testcase_files.items():
            testcase = self._create_testcase(testcase_name, cwe_id, files)
            cwe_group.testcases.append(testcase)
            cwe_group.total_files += len(files)
        
        # Sort testcases by name
        cwe_group.testcases.sort(key=lambda tc: tc.name)
        
        self.cwe_groups[cwe_id] = cwe_group
    
    def _extract_testcase_name(self, filename: str) -> Optional[str]:
        """
        Extract the base testcase name from a filename.
        
        Examples:
        - CWE15_External_Control_of_System_or_Configuration_Setting__connect_tcp_01.java -> CWE15_External_Control_of_System_or_Configuration_Setting__connect_tcp_01
        - CWE15_External_Control_of_System_or_Configuration_Setting__connect_tcp_22a.java -> CWE15_External_Control_of_System_or_Configuration_Setting__connect_tcp_22
        - CWE15_External_Control_of_System_or_Configuration_Setting__connect_tcp_81_bad.java -> CWE15_External_Control_of_System_or_Configuration_Setting__connect_tcp_81
        """
        if not filename.endswith('.java'):
            return None
        
        base_filename = filename[:-5]  # Remove .java extension
        
        # Check for special suffixes that indicate file variants within a testcase
        special_suffixes = ['_bad', '_good1', '_good2', '_good3', '_goodG2B', '_goodB2G', '_base']
        
        # Also check for patterns like _goodG2B1, _goodG2B2, etc.
        import re
        good_pattern = re.compile(r'_good[GB]2[GB]\d*$')
        if good_pattern.search(base_filename):
            # Find where the good pattern starts and remove it
            match = good_pattern.search(base_filename)
            return base_filename[:match.start()]
        
        # Check if it ends with a letter (like 22a, 22b, 54a, 54b, etc.)
        if len(base_filename) > 1 and base_filename[-1].isalpha() and base_filename[-2].isdigit():
            # This is a multi-file testcase like CWE15_..._22a
            return base_filename[:-1]  # Remove the letter suffix
        
        # Check for special suffixes
        for suffix in special_suffixes:
            if base_filename.endswith(suffix):
                return base_filename[:-len(suffix)]
        
        # If no special pattern found, return the whole base filename
        return base_filename
    
    def _create_testcase(self, name: str, cwe_id: str, files: List[str]) -> TestCase:
        """
        Create a TestCase object and determine its type and characteristics.
        """
        testcase = TestCase(name=name, cwe_id=cwe_id, files=sorted(files))
        
        # Determine testcase type based on file patterns
        file_basenames = [Path(f).name for f in files]
        
        # Check for class-based flaw pattern (_bad and _good1 files)
        has_bad_file = any('_bad.java' in f for f in file_basenames)
        has_good1_file = any('_good1.java' in f for f in file_basenames)
        
        # Check for abstract method pattern (_base, _bad, _goodG2B, and optionally _goodB2G)
        has_base_file = any('_base.java' in f for f in file_basenames)
        has_goodG2B_file = any('_goodG2B.java' in f for f in file_basenames)
        has_goodB2G_file = any('_goodB2G.java' in f for f in file_basenames)
        has_a_file = any(f.endswith('a.java') for f in file_basenames)  # Main file ending with 'a'
        
        # Check for servlet testcases
        testcase.has_servlet = any('Servlet' in f for f in file_basenames)
        
        # Abstract method testcases have: _base, _bad, _goodG2B, and the main 'a' file
        # Some may also have _goodB2G
        if has_base_file and has_bad_file and has_goodG2B_file and has_a_file:
            testcase.testcase_type = "abstract_method"
        elif has_bad_file and has_good1_file and not has_base_file:
            testcase.testcase_type = "class_based"
        else:
            testcase.testcase_type = "non_class_based"
        
        # Determine if it's bad-only
        testcase.is_bad_only = self._is_bad_only_testcase(testcase, files)
        
        if testcase.is_bad_only:
            testcase.testcase_type = "bad_only"
        
        return testcase
    
    def _is_bad_only_testcase(self, testcase: TestCase, files: List[str]) -> bool:
        """
        Determine if a testcase is bad-only based on patterns and file content analysis.
        """
        # Check against known bad-only patterns
        for pattern in self.known_bad_only_patterns:
            if testcase.name.startswith(pattern):
                return True
        
        # For more accurate detection, we would need to analyze the actual Java files
        # to check for empty good() methods or good methods with only unreachable code
        # This is a simplified heuristic-based approach
        
        # If it's a single file testcase, check if it might be bad-only
        if len(files) == 1:
            # This would require reading the file content to check for good method implementations
            # For now, we'll rely on the known patterns
            pass
        
        return False
    
    def print_summary(self):
        """
        Print a summary of all discovered testcases.
        """
        total_cwes = len(self.cwe_groups)
        total_testcases = sum(len(group.testcases) for group in self.cwe_groups.values())
        total_files = sum(group.total_files for group in self.cwe_groups.values())
        
        print(f"=== Juliet Test Suite Analysis Summary ===")
        print(f"Total CWEs: {total_cwes}")
        print(f"Total Testcases: {total_testcases}")
        print(f"Total Java Files: {total_files}")
        print()
        
        # Print details for each CWE
        for cwe_id in sorted(self.cwe_groups.keys()):
            group = self.cwe_groups[cwe_id]
            bad_only_count = sum(1 for tc in group.testcases if tc.is_bad_only)
            class_based_count = sum(1 for tc in group.testcases if tc.testcase_type == "class_based")
            abstract_method_count = sum(1 for tc in group.testcases if tc.testcase_type == "abstract_method")
            servlet_count = sum(1 for tc in group.testcases if tc.has_servlet)
            
            print(f"{cwe_id}:")
            print(f"  Testcases: {len(group.testcases)}")
            print(f"  Files: {group.total_files}")
            print(f"  Bad-only: {bad_only_count}")
            print(f"  Class-based: {class_based_count}")
            print(f"  Abstract method: {abstract_method_count}")
            print(f"  Servlet-based: {servlet_count}")
            
            # Show first few testcase examples
            if group.testcases:
                print(f"  Examples:")
                for testcase in group.testcases[:3]:
                    file_count = len(testcase.files)
                    type_info = f"[{testcase.testcase_type}]"
                    if testcase.is_bad_only:
                        type_info += " [BAD-ONLY]"
                    print(f"    - {testcase.name} ({file_count} files) {type_info}")
                
                if len(group.testcases) > 3:
                    print(f"    ... and {len(group.testcases) - 3} more")
            
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze Juliet Test Suite testcases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_juliet_testcases.py /path/to/juliet-java-test-suite
  python extract_juliet_testcases.py ../juliet-java-test-suite
        """
    )
    
    parser.add_argument(
        'juliet_path',
        help='Path to the Juliet Test Suite directory'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = JulietTestSuiteAnalyzer(args.juliet_path)
        cwe_groups = analyzer.analyze()
        analyzer.print_summary()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
