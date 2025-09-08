#!/usr/bin/env python3
"""
Data Model for CodeQL Query Refinement Experiment

This module defines the data structures for tracking CodeQL query performance,
false positives/negatives, regression analysis, and experiment history.

Key Features:
1. 3-shot query generation tracking
2. False positive/negative analysis with optional summarization
3. Regression measurement between iterations
4. Performance metrics (precision, recall, F1)
5. Execution time tracking per testcase
6. Historical data for graphing performance over time
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics


class TestcaseVariant(Enum):
    """Enum for testcase variants"""
    BAD = "bad"
    GOOD = "good"


class QueryGenerationMethod(Enum):
    """Enum for different query generation methods"""
    SINGLE_SHOT = "single_shot"
    THREE_SHOT = "three_shot"
    REFINED = "refined"


@dataclass
class TestcaseResult:
    """Result for a single testcase execution"""
    project_name: str
    cwe_id: str
    variant: TestcaseVariant
    expected_positive: bool  # True if this testcase should trigger the query
    actual_positive: bool    # True if query found results in this testcase
    result_count: int
    execution_time: float
    error_message: Optional[str] = None
    
    @property
    def is_true_positive(self) -> bool:
        """True positive: expected and actual are both True"""
        return self.expected_positive and self.actual_positive
    
    @property
    def is_true_negative(self) -> bool:
        """True negative: expected and actual are both False"""
        return not self.expected_positive and not self.actual_positive
    
    @property
    def is_false_positive(self) -> bool:
        """False positive: expected False, actual True"""
        return not self.expected_positive and self.actual_positive
    
    @property
    def is_false_negative(self) -> bool:
        """False negative: expected True, actual False"""
        return self.expected_positive and not self.actual_positive


@dataclass
class QueryCandidate:
    """A single query candidate in 3-shot generation"""
    candidate_id: str
    query_text: str
    generation_prompt: str
    generation_timestamp: datetime
    llm_model: str
    llm_temperature: float
    
    def __post_init__(self):
        if isinstance(self.generation_timestamp, str):
            self.generation_timestamp = datetime.fromisoformat(self.generation_timestamp)


@dataclass
class FalsePositiveNegativeAnalysis:
    """Analysis of false positives and negatives by analysis agent"""
    testcase_results: List[TestcaseResult]
    analysis_prompt: str
    analysis_summary: str
    analysis_timestamp: datetime
    analysis_model: str
    common_patterns: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.analysis_timestamp, str):
            self.analysis_timestamp = datetime.fromisoformat(self.analysis_timestamp)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a query evaluation"""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_testcases: int
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / Total"""
        return (self.true_positives + self.true_negatives) / self.total_testcases if self.total_testcases > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)"""
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator > 0 else 0.0


@dataclass
class ExecutionStats:
    """Statistics about query execution times"""
    total_execution_time: float
    average_execution_time: float
    min_execution_time: float
    max_execution_time: float
    median_execution_time: float
    successful_executions: int
    failed_executions: int
    
    @classmethod
    def from_testcase_results(cls, results: List[TestcaseResult]) -> 'ExecutionStats':
        """Calculate execution stats from testcase results"""
        successful = [r for r in results if r.error_message is None]
        failed = [r for r in results if r.error_message is not None]
        
        if successful:
            times = [r.execution_time for r in successful]
            return cls(
                total_execution_time=sum(times),
                average_execution_time=statistics.mean(times),
                min_execution_time=min(times),
                max_execution_time=max(times),
                median_execution_time=statistics.median(times),
                successful_executions=len(successful),
                failed_executions=len(failed)
            )
        else:
            return cls(
                total_execution_time=0.0,
                average_execution_time=0.0,
                min_execution_time=0.0,
                max_execution_time=0.0,
                median_execution_time=0.0,
                successful_executions=0,
                failed_executions=len(failed)
            )


@dataclass
class RegressionAnalysis:
    """Analysis of performance changes between iterations"""
    previous_run_id: str
    current_run_id: str
    
    # Performance deltas
    precision_delta: float
    recall_delta: float
    f1_delta: float
    accuracy_delta: float
    
    # Testcase-level changes
    newly_broken_testcases: List[str]  # Previously working, now broken
    newly_fixed_testcases: List[str]   # Previously broken, now working
    
    # Summary metrics
    total_regressions: int
    total_improvements: int
    net_change_score: float  # Positive = improvement, negative = regression
    
    @classmethod
    def compare_runs(cls, previous_run: 'TestRun', current_run: 'TestRun') -> 'RegressionAnalysis':
        """Compare two test runs and generate regression analysis"""
        prev_metrics = previous_run.performance_metrics
        curr_metrics = current_run.performance_metrics
        
        # Calculate deltas
        precision_delta = curr_metrics.precision - prev_metrics.precision
        recall_delta = curr_metrics.recall - prev_metrics.recall
        f1_delta = curr_metrics.f1_score - prev_metrics.f1_score
        accuracy_delta = curr_metrics.accuracy - prev_metrics.accuracy
        
        # Find testcase-level changes
        prev_results = {r.project_name: r for r in previous_run.testcase_results}
        curr_results = {r.project_name: r for r in current_run.testcase_results}
        
        newly_broken = []
        newly_fixed = []
        
        for project_name in set(prev_results.keys()) | set(curr_results.keys()):
            prev_result = prev_results.get(project_name)
            curr_result = curr_results.get(project_name)
            
            if prev_result and curr_result:
                # Both exist, check for changes
                prev_correct = prev_result.is_true_positive or prev_result.is_true_negative
                curr_correct = curr_result.is_true_positive or curr_result.is_true_negative
                
                if prev_correct and not curr_correct:
                    newly_broken.append(project_name)
                elif not prev_correct and curr_correct:
                    newly_fixed.append(project_name)
        
        # Calculate net change score (weighted combination of metrics)
        net_change_score = (
            0.4 * f1_delta +           # F1 is most important
            0.3 * precision_delta +     # Precision is important for practical use
            0.2 * recall_delta +        # Recall is important for coverage
            0.1 * accuracy_delta        # Accuracy provides overall picture
        )
        
        return cls(
            previous_run_id=previous_run.run_id,
            current_run_id=current_run.run_id,
            precision_delta=precision_delta,
            recall_delta=recall_delta,
            f1_delta=f1_delta,
            accuracy_delta=accuracy_delta,
            newly_broken_testcases=newly_broken,
            newly_fixed_testcases=newly_fixed,
            total_regressions=len(newly_broken),
            total_improvements=len(newly_fixed),
            net_change_score=net_change_score
        )


@dataclass
class TestRun:
    """Complete test run with all data and analysis"""
    # Unique identifiers
    run_id: str
    experiment_id: str
    iteration_number: int
    timestamp: datetime
    
    # Query information
    query_text: str
    query_description: str
    target_cwe: str
    generation_method: QueryGenerationMethod
    
    # 3-shot generation data (if applicable)
    query_candidates: List[QueryCandidate] = field(default_factory=list)
    selected_candidate_id: Optional[str] = None
    
    # Test execution results
    testcase_results: List[TestcaseResult] = field(default_factory=list)
    performance_metrics: Optional[PerformanceMetrics] = None
    execution_stats: Optional[ExecutionStats] = None
    
    # Analysis data
    fp_fn_analysis: Optional[FalsePositiveNegativeAnalysis] = None
    regression_analysis: Optional[RegressionAnalysis] = None
    
    # Metadata
    llm_model: str = "unknown"
    llm_temperature: float = 0.7
    codeql_version: str = "unknown"
    total_databases_tested: int = 0
    
    def __post_init__(self):
        """Post-initialization to calculate derived metrics"""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        
        if self.testcase_results and not self.performance_metrics:
            self.calculate_performance_metrics()
        
        if self.testcase_results and not self.execution_stats:
            self.execution_stats = ExecutionStats.from_testcase_results(self.testcase_results)
        
        if not self.total_databases_tested:
            self.total_databases_tested = len(self.testcase_results)
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics from testcase results"""
        tp = sum(1 for r in self.testcase_results if r.is_true_positive)
        tn = sum(1 for r in self.testcase_results if r.is_true_negative)
        fp = sum(1 for r in self.testcase_results if r.is_false_positive)
        fn = sum(1 for r in self.testcase_results if r.is_false_negative)
        
        self.performance_metrics = PerformanceMetrics(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_testcases=len(self.testcase_results)
        )
    
    def get_false_positives(self) -> List[TestcaseResult]:
        """Get all false positive results"""
        return [r for r in self.testcase_results if r.is_false_positive]
    
    def get_false_negatives(self) -> List[TestcaseResult]:
        """Get all false negative results"""
        return [r for r in self.testcase_results if r.is_false_negative]
    
    def get_results_by_cwe(self) -> Dict[str, List[TestcaseResult]]:
        """Group results by CWE"""
        by_cwe = {}
        for result in self.testcase_results:
            if result.cwe_id not in by_cwe:
                by_cwe[result.cwe_id] = []
            by_cwe[result.cwe_id].append(result)
        return by_cwe
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['timestamp'] = self.timestamp.isoformat()
        
        for candidate in data.get('query_candidates', []):
            if 'generation_timestamp' in candidate:
                candidate['generation_timestamp'] = candidate['generation_timestamp'].isoformat()
        
        if data.get('fp_fn_analysis') and 'analysis_timestamp' in data['fp_fn_analysis']:
            data['fp_fn_analysis']['analysis_timestamp'] = data['fp_fn_analysis']['analysis_timestamp'].isoformat()
        
        # Convert enums to strings
        data['generation_method'] = self.generation_method.value
        for result in data.get('testcase_results', []):
            result['variant'] = result['variant'].value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestRun':
        """Create from dictionary (JSON deserialization)"""
        # Convert string back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert enum strings back to enums
        data['generation_method'] = QueryGenerationMethod(data['generation_method'])
        
        for result in data.get('testcase_results', []):
            result['variant'] = TestcaseVariant(result['variant'])
        
        # Handle nested objects
        if data.get('query_candidates'):
            data['query_candidates'] = [QueryCandidate(**candidate) for candidate in data['query_candidates']]
        
        if data.get('testcase_results'):
            data['testcase_results'] = [TestcaseResult(**result) for result in data['testcase_results']]
        
        if data.get('performance_metrics'):
            data['performance_metrics'] = PerformanceMetrics(**data['performance_metrics'])
        
        if data.get('execution_stats'):
            data['execution_stats'] = ExecutionStats(**data['execution_stats'])
        
        if data.get('fp_fn_analysis'):
            fp_fn_data = data['fp_fn_analysis']
            fp_fn_data['testcase_results'] = [TestcaseResult(**r) for r in fp_fn_data.get('testcase_results', [])]
            data['fp_fn_analysis'] = FalsePositiveNegativeAnalysis(**fp_fn_data)
        
        if data.get('regression_analysis'):
            data['regression_analysis'] = RegressionAnalysis(**data['regression_analysis'])
        
        return cls(**data)


@dataclass
class Experiment:
    """Complete experiment tracking multiple iterations"""
    experiment_id: str
    name: str
    description: str
    target_cwe: str
    created_timestamp: datetime
    
    # Configuration
    use_three_shot_generation: bool = True
    use_fp_fn_analysis: bool = True
    llm_model: str = "unknown"
    
    # Test runs (iterations)
    test_runs: List[TestRun] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.created_timestamp, str):
            self.created_timestamp = datetime.fromisoformat(self.created_timestamp)
    
    def add_test_run(self, test_run: TestRun):
        """Add a test run to the experiment"""
        test_run.experiment_id = self.experiment_id
        test_run.iteration_number = len(self.test_runs) + 1
        self.test_runs.append(test_run)
    
    def get_latest_run(self) -> Optional[TestRun]:
        """Get the most recent test run"""
        return self.test_runs[-1] if self.test_runs else None
    
    def get_performance_history(self) -> List[Tuple[int, PerformanceMetrics]]:
        """Get performance metrics history as (iteration, metrics) tuples"""
        return [(run.iteration_number, run.performance_metrics) 
                for run in self.test_runs if run.performance_metrics]
    
    def get_regression_history(self) -> List[RegressionAnalysis]:
        """Get all regression analyses"""
        return [run.regression_analysis for run in self.test_runs 
                if run.regression_analysis]
    
    def calculate_trend(self, metric_name: str) -> Tuple[float, str]:
        """Calculate trend for a specific metric (slope and direction)"""
        history = self.get_performance_history()
        if len(history) < 2:
            return 0.0, "insufficient_data"
        
        values = []
        for iteration, metrics in history:
            if hasattr(metrics, metric_name):
                values.append((iteration, getattr(metrics, metric_name)))
        
        if len(values) < 2:
            return 0.0, "insufficient_data"
        
        # Simple linear regression to calculate slope
        n = len(values)
        sum_x = sum(x for x, y in values)
        sum_y = sum(y for x, y in values)
        sum_xy = sum(x * y for x, y in values)
        sum_x2 = sum(x * x for x, y in values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"
        
        return slope, direction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['created_timestamp'] = self.created_timestamp.isoformat()
        data['test_runs'] = [run.to_dict() for run in self.test_runs]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create from dictionary (JSON deserialization)"""
        data['created_timestamp'] = datetime.fromisoformat(data['created_timestamp'])
        data['test_runs'] = [TestRun.from_dict(run_data) for run_data in data.get('test_runs', [])]
        return cls(**data)


class ExperimentStorage:
    """Handles saving and loading experiments to/from disk"""
    
    def __init__(self, storage_dir: str = "./experiment_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def save_experiment(self, experiment: Experiment):
        """Save experiment to disk"""
        file_path = self.storage_dir / f"{experiment.experiment_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(experiment.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from disk"""
        file_path = self.storage_dir / f"{experiment_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Experiment.from_dict(data)
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs"""
        return [f.stem for f in self.storage_dir.glob("*.json")]
    
    def save_test_run(self, test_run: TestRun):
        """Save individual test run (useful for incremental updates)"""
        run_file = self.storage_dir / "runs" / f"{test_run.run_id}.json"
        run_file.parent.mkdir(exist_ok=True)
        
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(test_run.to_dict(), f, indent=2, ensure_ascii=False)


def create_new_experiment(name: str, description: str, target_cwe: str, 
                         use_three_shot: bool = True, use_fp_fn_analysis: bool = True,
                         llm_model: str = "unknown") -> Experiment:
    """Factory function to create a new experiment"""
    experiment_id = str(uuid.uuid4())
    
    return Experiment(
        experiment_id=experiment_id,
        name=name,
        description=description,
        target_cwe=target_cwe,
        created_timestamp=datetime.now(timezone.utc),
        use_three_shot_generation=use_three_shot,
        use_fp_fn_analysis=use_fp_fn_analysis,
        llm_model=llm_model
    )


def create_new_test_run(query_text: str, query_description: str, target_cwe: str,
                       generation_method: QueryGenerationMethod = QueryGenerationMethod.SINGLE_SHOT,
                       llm_model: str = "unknown", llm_temperature: float = 0.7) -> TestRun:
    """Factory function to create a new test run"""
    run_id = str(uuid.uuid4())
    
    return TestRun(
        run_id=run_id,
        experiment_id="",  # Will be set when added to experiment
        iteration_number=0,  # Will be set when added to experiment
        timestamp=datetime.now(timezone.utc),
        query_text=query_text,
        query_description=query_description,
        target_cwe=target_cwe,
        generation_method=generation_method,
        llm_model=llm_model,
        llm_temperature=llm_temperature
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing CodeQL Experiment Data Model...")
    
    # Create a new experiment
    experiment = create_new_experiment(
        name="CWE89 SQL Injection Detection",
        description="Iterative refinement of CodeQL queries for SQL injection detection",
        target_cwe="CWE89",
        use_three_shot=True,
        use_fp_fn_analysis=True,
        llm_model="gpt-4"
    )
    
    print(f"Created experiment: {experiment.experiment_id}")
    
    # Create a test run
    test_run = create_new_test_run(
        query_text="import java\nfrom Method m\nwhere m.getName() = \"executeQuery\"\nselect m",
        query_description="Initial query to detect SQL execution methods",
        target_cwe="CWE89",
        generation_method=QueryGenerationMethod.THREE_SHOT,
        llm_model="gpt-4"
    )
    
    # Add some mock testcase results
    test_run.testcase_results = [
        TestcaseResult(
            project_name="CWE89_SQL_Injection_bad",
            cwe_id="CWE89",
            variant=TestcaseVariant.BAD,
            expected_positive=True,
            actual_positive=True,
            result_count=1,
            execution_time=0.5
        ),
        TestcaseResult(
            project_name="CWE89_SQL_Injection_good",
            cwe_id="CWE89",
            variant=TestcaseVariant.GOOD,
            expected_positive=False,
            actual_positive=False,
            result_count=0,
            execution_time=0.3
        )
    ]
    
    # Calculate metrics
    test_run.calculate_performance_metrics()
    
    print(f"Test run metrics: P={test_run.performance_metrics.precision:.2f}, "
          f"R={test_run.performance_metrics.recall:.2f}, "
          f"F1={test_run.performance_metrics.f1_score:.2f}")
    
    # Add to experiment
    experiment.add_test_run(test_run)
    
    # Test storage
    storage = ExperimentStorage("./test_experiment_data")
    storage.save_experiment(experiment)
    
    # Load and verify
    loaded = storage.load_experiment(experiment.experiment_id)
    assert loaded is not None
    assert loaded.name == experiment.name
    assert len(loaded.test_runs) == 1
    
    print("✅ Data model test completed successfully!")
