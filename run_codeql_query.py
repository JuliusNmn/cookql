#!/usr/bin/env python3
"""
CodeQL Query Runner for Juliet Testcases

This script runs CodeQL queries on generated testcase databases, with support for:
- Validating query syntax without execution (--validate-only)
- Running single queries (.ql files) or multiple queries (directories containing .ql files)
- Running queries on all databases or filtered subsets
- Filtering by CWE, project names, or limits
- Aggregating and reporting results
- Displaying query results after execution
- Parallel execution for performance
- Multiple output formats (JSON, CSV, SARIF)

Usage examples:
  # Validate a query without running it
  python run_codeql_query.py --query path/to/query.ql --validate-only
  
  # Run a single query file
  python run_codeql_query.py --query path/to/query.ql --db-root ./codeql_dbs
  
  # Run all queries in a directory
  python run_codeql_query.py --query path/to/queries/ --db-root ./codeql_dbs
  
  # Use CodeQL query packs
  python run_codeql_query.py --query codeql/java-queries --db-root ./extracted_testcases/codeql_dbs --codeql-path /opt/codeql/codeql
  
  # With filtering options
  python run_codeql_query.py --query queries/ --db-root ./codeql_dbs --cwe CWE15
  python run_codeql_query.py --query queries/ --db-root ./codeql_dbs --limit 10
  python run_codeql_query.py --query queries/ --db-root ./codeql_dbs --projects project1,project2
"""

import os
import re
import argparse
import subprocess
import json
import csv
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time
from datetime import datetime


@dataclass
class QueryResult:
    """Represents the result of running a query on a single database."""
    project_name: str
    database_path: str
    success: bool
    execution_time: float
    result_count: int
    results: List[Dict[str, Any]]
    error_message: Optional[str] = None
    cwe_id: Optional[str] = None
    variant: Optional[str] = None  # 'bad' or 'good'


@dataclass
class QuerySummary:
    """Summary statistics for a query run across multiple databases."""
    total_databases: int
    successful_runs: int
    failed_runs: int
    total_results: int
    total_execution_time: float
    average_execution_time: float
    query_file: str
    timestamp: str
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0


class CodeQLQueryRunner:
    """Main class for running CodeQL queries on testcase databases."""
    
    def __init__(self, codeql_path: str = "codeql", max_workers: int = 4):
        self.codeql_path = codeql_path
        self.max_workers = max_workers
        self.results: List[QueryResult] = []
        
    def find_databases(self, db_root: str, filters: Dict[str, Any]) -> List[Tuple[str, Path]]:
        """Find CodeQL databases based on filters."""
        db_root_path = Path(db_root)
        if not db_root_path.exists():
            raise FileNotFoundError(f"Database root directory does not exist: {db_root}")
        
        # Find all database directories (contain codeql-database.yml)
        all_dbs = []
        for item in db_root_path.iterdir():
            if item.is_dir():
                yml_file = item / "codeql-database.yml"
                if yml_file.exists():
                    all_dbs.append((item.name, item))
        
        if not all_dbs:
            print(f"No CodeQL databases found in {db_root}")
            return []
        
        print(f"Found {len(all_dbs)} total databases")
        
        # Apply filters
        filtered_dbs = self._apply_filters(all_dbs, filters)
        
        print(f"After filtering: {len(filtered_dbs)} databases selected")
        return filtered_dbs
    
    def _apply_filters(self, databases: List[Tuple[str, Path]], filters: Dict[str, Any]) -> List[Tuple[str, Path]]:
        """Apply various filters to the database list."""
        filtered = databases
        
        # Filter by CWE
        if filters.get('cwe'):
            cwe_filter = filters['cwe'].upper()
            if not cwe_filter.startswith('CWE'):
                cwe_filter = f"CWE{cwe_filter}"
            filtered = [(name, path) for name, path in filtered if cwe_filter in name.upper()]
            print(f"Filtered by CWE '{cwe_filter}': {len(filtered)} databases")
        
        # Filter by variant (bad/good)
        if filters.get('variant'):
            variant = filters['variant'].lower()
            if variant in ['bad', 'good']:
                filtered = [(name, path) for name, path in filtered if name.endswith(f"_{variant}")]
                print(f"Filtered by variant '{variant}': {len(filtered)} databases")
        
        # Filter by specific project names
        if filters.get('projects'):
            project_names = set(filters['projects'])
            filtered = [(name, path) for name, path in filtered if name in project_names]
            print(f"Filtered by project list: {len(filtered)} databases")
        
        # Filter by pattern
        if filters.get('pattern'):
            pattern = filters['pattern']
            filtered = [(name, path) for name, path in filtered if re.search(pattern, name, re.IGNORECASE)]
            print(f"Filtered by pattern '{pattern}': {len(filtered)} databases")
        
        # Apply limit
        if filters.get('limit'):
            limit = int(filters['limit'])
            filtered = filtered[:limit]
            print(f"Limited to first {limit} databases")
        
        return filtered
    
    def _extract_metadata(self, project_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract CWE ID and variant from project name."""
        cwe_match = re.search(r'(CWE\d+)', project_name, re.IGNORECASE)
        cwe_id = cwe_match.group(1).upper() if cwe_match else None
        
        variant = None
        if project_name.endswith('_bad'):
            variant = 'bad'
        elif project_name.endswith('_good'):
            variant = 'good'
        
        return cwe_id, variant
    
    def _run_single_query(self, project_name: str, db_path: Path, query_file: str, 
                         output_format: str = "sarif-latest") -> QueryResult:
        """Run a CodeQL query on a single database."""
        start_time = time.time()
        
        try:
            # Create temporary output file
            output_file = db_path.parent / f"{project_name}_results.{output_format}"
            
            # Build CodeQL command
            cmd = [
                self.codeql_path, "database", "analyze",
                str(db_path),
                query_file,
                f"--format={output_format}",
                f"--output={output_file}",
                "--no-sarif-add-snippets",  # Reduce output size
                "--search-path=/opt/codeql"  # Add CodeQL library search path
            ]
            print(f"Running command: {' '.join(cmd)}")
            
            # Run the query
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout per query
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse results
                results, result_count = self._parse_results(output_file, output_format)
                
                # Clean up output file
                if output_file.exists():
                    output_file.unlink()
                
                cwe_id, variant = self._extract_metadata(project_name)
                
                return QueryResult(
                    project_name=project_name,
                    database_path=str(db_path),
                    success=True,
                    execution_time=execution_time,
                    result_count=result_count,
                    results=results,
                    cwe_id=cwe_id,
                    variant=variant
                )
            else:
                # Print command output immediately on failure
                print(f"❌ CodeQL failed for {project_name} (exit code {result.returncode})")
                if result.stdout.strip():
                    print(f"STDOUT:\n{result.stdout.strip()}")
                if result.stderr.strip():
                    print(f"STDERR:\n{result.stderr.strip()}")
                print("-" * 50)
                
                # Clean up output file on error
                if output_file.exists():
                    output_file.unlink()
                
                cwe_id, variant = self._extract_metadata(project_name)
                
                # Store both stdout and stderr in error message for later reference
                error_parts = []
                if result.stdout.strip():
                    error_parts.append(f"STDOUT: {result.stdout.strip()}")
                if result.stderr.strip():
                    error_parts.append(f"STDERR: {result.stderr.strip()}")
                error_message = "\n".join(error_parts) if error_parts else f"Exit code {result.returncode}"
                
                return QueryResult(
                    project_name=project_name,
                    database_path=str(db_path),
                    success=False,
                    execution_time=execution_time,
                    result_count=0,
                    results=[],
                    error_message=error_message,
                    cwe_id=cwe_id,
                    variant=variant
                )
                
        except subprocess.TimeoutExpired:
            print(f"❌ CodeQL timeout for {project_name} (5 minute limit exceeded)")
            print("-" * 50)
            
            execution_time = time.time() - start_time
            cwe_id, variant = self._extract_metadata(project_name)
            
            return QueryResult(
                project_name=project_name,
                database_path=str(db_path),
                success=False,
                execution_time=execution_time,
                result_count=0,
                results=[],
                error_message="Query execution timeout (5 minutes)",
                cwe_id=cwe_id,
                variant=variant
            )
        except Exception as e:
            print(f"❌ CodeQL exception for {project_name}: {str(e)}")
            print("-" * 50)
            
            execution_time = time.time() - start_time
            cwe_id, variant = self._extract_metadata(project_name)
            
            return QueryResult(
                project_name=project_name,
                database_path=str(db_path),
                success=False,
                execution_time=execution_time,
                result_count=0,
                results=[],
                error_message=str(e),
                cwe_id=cwe_id,
                variant=variant
            )
    
    def _parse_results(self, output_file: Path, output_format: str) -> Tuple[List[Dict[str, Any]], int]:
        """Parse query results from output file."""
        if not output_file.exists():
            return [], 0
        
        try:
            if output_format.startswith("sarif"):
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle SARIF structure
                if isinstance(data, dict) and 'runs' in data:
                    if len(data['runs']) > 0:
                        results = data['runs'][0].get('results', [])
                        return results, len(results)
                return [], 0
            
            elif output_format == "csv":
                results = []
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        results.append(dict(row))
                return results, len(results)
            
            else:
                # For other formats, just count non-empty lines
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                non_empty_lines = [line for line in lines if line.strip()]
                # For CSV format, create simple dict entries
                if output_format == "csv" and non_empty_lines:
                    # If it's CSV but no header, create simple entries
                    results = []
                    for i, line in enumerate(non_empty_lines):
                        if i == 0 and any(sep in line for sep in [',', '\t']):
                            # Skip header line
                            continue
                        results.append({"result": line.strip(), "line_number": i + 1})
                    return results, len(results)
                return [], len(non_empty_lines)
                
        except Exception as e:
            print(f"Error parsing results from {output_file}: {e}")
            return [], 0
    
    def run_query(self, query_file: str, databases: List[Tuple[str, Path]], 
                  output_format: str = "sarif-latest", parallel: bool = True) -> List[QueryResult]:
        """Run a query or queries on multiple databases."""
        # Check if it's a query pack (contains ':'), a directory, or a file path
        query_path = Path(query_file)
        is_query_pack = ':' in query_file
        is_directory = query_path.exists() and query_path.is_dir()
        is_single_file = query_path.exists() and query_path.is_file() and query_file.endswith('.ql')
        
        if not is_query_pack and not query_path.exists():
            raise FileNotFoundError(f"Query file or directory does not exist: {query_file}")
        
        if is_query_pack:
            query_type = "query pack"
        elif is_directory:
            query_type = "query directory"
            # Count .ql files in directory for informational purposes
            ql_files = list(query_path.rglob("*.ql"))
            print(f"Found {len(ql_files)} .ql files in directory")
        elif is_single_file:
            query_type = "query file"
        else:
            query_type = "query"
        
        print(f"Running {query_type} '{query_file}' on {len(databases)} databases...")
        
        if parallel and len(databases) > 1:
            # Run queries in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_db = {
                    executor.submit(self._run_single_query, name, path, query_file, output_format): (name, path)
                    for name, path in databases
                }
                
                results = []
                for future in concurrent.futures.as_completed(future_to_db):
                    name, path = future_to_db[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        status = "✅" if result.success else "❌"
                        count_info = f" ({result.result_count} results)" if result.success else ""
                        print(f"{status} {len(results)}/{len(databases)} {result.project_name}{count_info} ")
                        
                    except Exception as e:
                        print(f"❌ {name}: Exception occurred: {e}")
        else:
            # Run queries sequentially
            results = []
            for name, path in databases:
                result = self._run_single_query(name, path, query_file, output_format)
                results.append(result)
                
                status = "✅" if result.success else "❌"
                count_info = f" ({result.result_count} results)" if result.success else ""
                print(f"{status} {result.project_name}{count_info}")
                print(f"Processed {len(results)}/{len(databases)} databases")
        
        self.results = results
        return results
    
    def generate_summary(self, query_file: str) -> QuerySummary:
        """Generate a summary of the query results."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        total_results = sum(r.result_count for r in successful)
        total_time = sum(r.execution_time for r in self.results)
        avg_time = total_time / len(self.results) if self.results else 0
        
        false_positives = sum(1 for r in self.results if r.variant == 'good' and r.result_count > 0)
        false_negatives = sum(1 for r in self.results if r.variant == 'bad' and r.result_count == 0)
        
        precision = total_results / (total_results + false_positives) if total_results + false_positives > 0 else 0
        recall = total_results / (total_results + false_negatives) if total_results + false_negatives > 0 else 0
        
        return QuerySummary(
            total_databases=len(self.results),
            successful_runs=len(successful),
            failed_runs=len(failed),
            total_results=total_results,
            total_execution_time=total_time,
            average_execution_time=avg_time,
            query_file=query_file,
            timestamp=datetime.now().isoformat(),
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall
        )
    
    def save_results(self, output_file: str, format: str = "json", include_summary: bool = True):
        """Save results to file in various formats."""
        output_path = Path(output_file)
        
        if format.lower() == "json":
            data = {
                "results": [asdict(r) for r in self.results]
            }
            
            if include_summary and self.results:
                # We need the query file name - try to extract from first result or use placeholder
                query_file = "unknown_query.ql"  # This should be passed from caller
                data["summary"] = asdict(self.generate_summary(query_file))
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if not self.results:
                    return
                
                fieldnames = ['project_name', 'cwe_id', 'variant', 'success', 'result_count', 
                             'execution_time', 'error_message']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    writer.writerow({
                        'project_name': result.project_name,
                        'cwe_id': result.cwe_id or '',
                        'variant': result.variant or '',
                        'success': result.success,
                        'result_count': result.result_count,
                        'execution_time': f"{result.execution_time:.2f}",
                        'error_message': result.error_message or ''
                    })
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self, query_file: str):
        """Print a summary of the results to console."""
        if not self.results:
            print("No results to summarize.")
            return
        
        summary = self.generate_summary(query_file)
        
        print(f"\n{'='*60}")
        print(f"QUERY EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Query File: {summary.query_file}")
        print(f"Timestamp: {summary.timestamp}")
        print(f"Total Databases: {summary.total_databases}")
        print(f"Successful Runs: {summary.successful_runs}")
        print(f"Failed Runs: {summary.failed_runs}")
        print(f"Total Results Found: {summary.total_results}")
        print(f"Total Execution Time: {summary.total_execution_time:.2f}s")
        print(f"Average Time per DB: {summary.average_execution_time:.2f}s")
        
        # Show breakdown by CWE
        cwe_stats = {}
        variant_stats = {'bad': 0, 'good': 0, 'unknown': 0}
        
        for result in self.results:
            if result.success:
                cwe = result.cwe_id or 'Unknown'
                if cwe not in cwe_stats:
                    cwe_stats[cwe] = {'total': 0, 'with_results': 0, 'result_count': 0}
                cwe_stats[cwe]['total'] += 1
                if result.result_count > 0:
                    cwe_stats[cwe]['with_results'] += 1
                cwe_stats[cwe]['result_count'] += result.result_count
                
                variant = result.variant or 'unknown'
                variant_stats[variant] += result.result_count
        
        if cwe_stats:
            print(f"\nBreakdown by CWE:")
            for cwe, stats in sorted(cwe_stats.items()):
                print(f"  {cwe}: {stats['with_results']}/{stats['total']} databases with results "
                      f"({stats['result_count']} total results)")
        
        if any(variant_stats.values()):
            print(f"\nBreakdown by Variant:")
            for variant, count in variant_stats.items():
                if count > 0:
                    print(f"  {variant}: {count} results")
        
        # Show failed runs
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\nFailed Runs ({len(failed_results)}):")
            for result in failed_results:
                error = result.error_message or "Unknown error"
                print(f"  ❌ {result.project_name}: {error}")
    
    def validate_query(self, query_file: str, timeout: int = 30) -> Tuple[bool, str]:
        """Validate a CodeQL query using the --check-only flag.
        
        Returns:
            Tuple[bool, str]: (success, message) where success is True if valid, False otherwise
        """
        import tempfile
        
        query_path = Path(query_file)
        if not query_path.exists():
            return False, f"Query file does not exist: {query_file}"
        
        
        if not query_path.is_dir() and not query_file.endswith('.ql'):
            return False, f"File is not a .ql query file: {query_file}"
        
        

        try:
            
            # Run CodeQL query compile with --check-only flag
            cmd = [
                self.codeql_path, "query", "compile", 
                "--check-only",
                "--threads=1",  # Use single thread for validation
                "--search-path=/opt/codeql",  # Add CodeQL library search path
                str(query_path)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return True, "Query compiled successfully"
            else:
                # Extract error information
                error_output = result.stderr.strip() or result.stdout.strip()
                return False, f"Compilation failed with error: {error_output}"
                
        except subprocess.TimeoutExpired:
            return False, f"Query validation timed out after {timeout} seconds"
        except Exception as e:
            return False, f"Exception during validation: {str(e)}"
        

    def print_results(self, limit_per_db: int = None, show_details: bool = True):
        """Print the actual query results to console."""
        if not self.results:
            print("No results to display.")
            return
        
        successful_results = [r for r in self.results if r.success and r.result_count > 0]
        
        if not successful_results:
            print("No successful query results found.")
            return
        
        print(f"\n{'='*60}")
        print(f"QUERY RESULTS")
        print(f"{'='*60}")
        
        total_displayed = 0
        for result in successful_results:
            print(f"\n📁 Database: {result.project_name}")
            print(f"   CWE: {result.cwe_id or 'Unknown'}")
            print(f"   Variant: {result.variant or 'Unknown'}")
            print(f"   Results: {result.result_count}")
            print(f"   Execution time: {result.execution_time:.2f}s")
            
            if show_details and result.results:
                print(f"   Details:")
                display_limit = min(len(result.results), limit_per_db) if limit_per_db else len(result.results)
                
                for i, res in enumerate(result.results[:display_limit]):
                    if isinstance(res, dict):
                        # Handle different result formats
                        if 'message' in res and 'locations' in res:
                            # SARIF format
                            message = res.get('message', {}).get('text', 'No message')
                            locations = res.get('locations', [])
                            print(f"     {i+1}. {message}")
                            if locations:
                                for loc in locations[:1]:  # Show first location
                                    phys_loc = loc.get('physicalLocation', {})
                                    artifact = phys_loc.get('artifactLocation', {}).get('uri', 'unknown')
                                    region = phys_loc.get('region', {})
                                    start_line = region.get('startLine', 'unknown')
                                    print(f"        📍 {artifact}:{start_line}")
                        elif 'result' in res:
                            # Simple format with result field
                            print(f"     {i+1}. {res['result']}")
                        else:
                            # Generic dict format - show key-value pairs
                            result_str = ", ".join([f"{v}" for k, v in res.items() if k not in ['line_number']])
                            if result_str:
                                print(f"     {i+1}. {result_str}")
                    else:
                        # String or other format
                        print(f"     {i+1}. {str(res)}")
                    
                    total_displayed += 1
                
                if limit_per_db and len(result.results) > limit_per_db:
                    remaining = len(result.results) - limit_per_db
                    print(f"     ... and {remaining} more results")
            
            print(f"   {'-'*40}")
        
        print(f"\nTotal results displayed: {total_displayed}")
        total_results = sum(r.result_count for r in successful_results)
        if limit_per_db and total_displayed < total_results:
            print(f"Total results available: {total_results} (showing limited view)")
        
        print(f"{'='*60}")


def main():
    import dotenv
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run CodeQL queries on Juliet testcase databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a query without running it
  python run_codeql_query.py --query security.ql --validate-only
  
  # Single query file
  python run_codeql_query.py --query security.ql --db-root ./codeql_dbs
  
  # Directory of queries
  python run_codeql_query.py --query ./security-queries/ --db-root ./codeql_dbs
  
  # With filtering and output options
  python run_codeql_query.py --query ./queries/ --db-root ./codeql_dbs --cwe 15 --show-results
  python run_codeql_query.py --query security.ql --db-root ./codeql_dbs --variant bad --limit 10 --show-results --result-limit 5
  python run_codeql_query.py --query ./queries/ --db-root ./codeql_dbs --projects "CWE15_test_bad,CWE15_test_good" --show-results
  python run_codeql_query.py --query security.ql --db-root ./codeql_dbs --pattern "connect_tcp" --output results.json --show-results
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--query', '-q',
        required=True,
        help='Path to the CodeQL query file (.ql), directory containing .ql files, or query pack name'
    )
    
    parser.add_argument(
        '--db-root',
        required=True,
        help='Root directory containing CodeQL databases'
    )
    
    # Filtering options
    parser.add_argument(
        '--cwe',
        help='Filter databases by CWE number (e.g., 15 or CWE15)'
    )
    
    parser.add_argument(
        '--variant',
        choices=['bad', 'good'],
        help='Filter databases by variant (bad or good)'
    )
    
    parser.add_argument(
        '--projects',
        help='Comma-separated list of specific project names to run on'
    )
    
    parser.add_argument(
        '--pattern',
        help='Regular expression pattern to filter database names'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit the number of databases to process'
    )
    
    # Execution options
    parser.add_argument(
        '--codeql-path',
        default=os.getenv("CODEQL_PATH"),
        help='Path to the CodeQL executable (default: CODEQL_PATH)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run queries sequentially instead of in parallel'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Output file for results (format determined by extension: .json, .csv)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['csv', 'sarif-latest', 'sarifv2.1.0', 'sarif'],
        default='sarif-latest',
        help='Format for CodeQL query output (default: csv)'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing summary to console'
    )
    
    parser.add_argument(
        '--show-results',
        action='store_true',
        help='Display the actual query results after execution'
    )
    
    parser.add_argument(
        '--result-limit',
        type=int,
        help='Limit the number of results displayed per database'
    )
    
    parser.add_argument(
        '--no-result-details',
        action='store_true',
        help='Show only result counts without detailed content'
    )
    
    # Query validation options
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the query syntax without running it on databases'
    )
    
    parser.add_argument(
        '--validation-timeout',
        type=int,
        default=30,
        help='Timeout for query validation in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create query runner
        runner = CodeQLQueryRunner(
            codeql_path=args.codeql_path,
            max_workers=args.max_workers
        )
        
        # Handle validation-only mode
        if args.validate_only:
            print(f"Validating query: {args.query}")
            success, message = runner.validate_query(args.query, args.validation_timeout)
            
            if success:
                print(f"VALID: {message}")
            else:
                print(f"INVALID: {message}")
            
            # Return appropriate exit code
            return 0 if success else 1
        
        # Prepare filters
        filters = {}
        if args.cwe:
            filters['cwe'] = args.cwe
        if args.variant:
            filters['variant'] = args.variant
        if args.projects:
            filters['projects'] = [p.strip() for p in args.projects.split(',')]
        if args.pattern:
            filters['pattern'] = args.pattern
        if args.limit:
            filters['limit'] = args.limit
        
        # Find databases
        databases = runner.find_databases(args.db_root, filters)
        
        if not databases:
            print("No databases found matching the specified criteria.")
            return 1
        
        # Run query
        results = runner.run_query(
            query_file=args.query,
            databases=databases,
            output_format=args.output_format,
            parallel=not args.sequential
        )
        
        # Print summary
        if not args.no_summary:
            runner.print_summary(args.query)
        
        # Display query results if requested
        if args.show_results:
            runner.print_results(
                limit_per_db=args.result_limit,
                show_details=not args.no_result_details
            )
        
        # Save results if requested
        if args.output:
            output_format = 'json'
            if args.output.endswith('.csv'):
                output_format = 'csv'
            
            runner.save_results(args.output, output_format)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nQuery execution interrupted by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
