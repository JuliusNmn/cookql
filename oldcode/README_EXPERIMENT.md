# CodeQL Query Refinement Experiment System with LangGraph

This system implements an iterative feedback loop for refining CodeQL queries using **LangGraph agents** with LLM-based analysis and regression tracking. It supports the three key features you requested:

1. **3-shot query generation** with best candidate selection
2. **Optional false positive/negative analysis** with summarization  
3. **Regression measurement** and tracking across iterations

## 🆕 LangGraph Integration

The system now uses **LangGraph** for sophisticated agent-based query generation and refinement, providing:

- **State management** across iterations
- **Conditional workflows** for decision making
- **Tool-based architecture** for modular functionality
- **Robust error handling** and recovery
- **Transparent execution flow** with detailed logging

## Architecture Overview

### Core Components

1. **`experiment_data_model.py`** - Complete data model for experiments
   - `TestRun` - Individual query evaluation with results
   - `Experiment` - Multi-iteration experiment tracking
   - `PerformanceMetrics` - Precision, recall, F1, accuracy
   - `RegressionAnalysis` - Comparison between iterations
   - `FalsePositiveNegativeAnalysis` - LLM-based error analysis

2. **`experiment_runner.py`** - LangGraph-based experiment orchestration
   - `LangGraphQueryAgent` - Agent-based query generation and analysis
   - `ExperimentRunner` - Coordinates the entire process using LangGraph
   - **Tools**: `generate_codeql_query`, `analyze_false_positives_negatives`
   - **Workflow**: Multi-node graph with conditional edges
   - Integration with existing `CodeQLQueryRunner`

3. **`experiment_visualization.py`** - Analysis and reporting
   - Performance trend graphs over time
   - Regression analysis visualization
   - Comprehensive report generation
   - Multi-experiment comparison

## Data Model Details

### TestRun Structure
```python
@dataclass
class TestRun:
    # Identifiers
    run_id: str
    experiment_id: str
    iteration_number: int
    timestamp: datetime
    
    # Query information
    query_text: str
    query_description: str
    target_cwe: str
    generation_method: QueryGenerationMethod
    
    # 3-shot generation data
    query_candidates: List[QueryCandidate]
    selected_candidate_id: Optional[str]
    
    # Results and metrics
    testcase_results: List[TestcaseResult]
    performance_metrics: PerformanceMetrics
    execution_stats: ExecutionStats
    
    # Analysis
    fp_fn_analysis: Optional[FalsePositiveNegativeAnalysis]
    regression_analysis: Optional[RegressionAnalysis]
```

### Key Metrics Tracked

**Performance Metrics:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 Score = 2 * (precision * recall) / (precision + recall)
- Accuracy = (TP + TN) / Total

**Execution Stats:**
- Total/average/min/max execution time
- Success/failure rates
- Per-testcase timing

**Regression Analysis:**
- Performance deltas between iterations
- Newly broken/fixed testcases
- Net change score (weighted combination)
- Cumulative progress tracking

## Installation

First, install the LangGraph dependencies:

```bash
pip install -r requirements_langgraph.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage Examples

### Basic LangGraph Experiment
```bash
# Run a basic experiment with LangGraph agent
python experiment_runner.py \
    --name "CWE89 SQL Injection Detection" \
    --description "LangGraph-based iterative refinement for SQL injection queries" \
    --target-cwe CWE89 \
    --db-root ./extracted_testcases/codeql_dbs \
    --max-iterations 5
```

### Advanced Configuration
```bash
# With custom settings and API key
python experiment_runner.py \
    --name "Advanced SQL Injection with LangGraph" \
    --description "Using GPT-4 with LangGraph agent workflow" \
    --target-cwe CWE89 \
    --db-root ./extracted_testcases/codeql_dbs \
    --max-iterations 10 \
    --llm-model gpt-4 \
    --api-key "your-api-key" \
    --cwe CWE89 \
    --limit 50 \
    --storage-dir ./my_experiments
```

### Generate Reports
```bash
# List all experiments
python experiment_visualization.py --list-experiments

# Generate comprehensive report
python experiment_visualization.py \
    --experiment-id <experiment-id> \
    --generate-report \
    --output-dir ./reports

# Compare multiple experiments
python experiment_visualization.py \
    --compare exp1-id exp2-id exp3-id
```

## Features in Detail

### 1. 3-Shot Query Generation with LangGraph

The LangGraph agent generates and evaluates 3 query candidates per iteration:
- **Tool-based generation**: Uses `generate_codeql_query` tool for consistent query creation
- **Parallel evaluation**: All candidates are evaluated against the test database
- **Intelligent selection**: Best candidate chosen based on F1 score and precision
- **State tracking**: All candidates stored in agent state for analysis

```python
# LangGraph workflow for 3-shot generation
workflow.add_node("generate_candidates", self._generate_candidates_node)
workflow.add_node("evaluate_candidates", self._evaluate_candidates_node)  
workflow.add_node("select_best_candidate", self._select_best_candidate_node)

# Agent state tracks all candidates
state['query_candidates'] = candidates
state['evaluation_results'] = evaluation_results
state['best_candidate'] = selected_candidate
```

### 2. False Positive/Negative Analysis with LangGraph

Agent-based analysis of classification errors using structured tools:
- **Tool-based analysis**: Uses `analyze_false_positives_negatives` tool
- **Pattern identification**: Automatically identifies common error patterns
- **Structured feedback**: Provides categorized improvement suggestions
- **State persistence**: Analysis results stored in agent state for next iteration

```python
# LangGraph tool for FP/FN analysis
@tool
def analyze_false_positives_negatives(input_data: AnalysisInput) -> str:
    """Analyze false positives and negatives to provide improvement suggestions"""
    
# Agent workflow integration
workflow.add_node("analyze_errors", self._analyze_errors_node)

# State-based feedback loop
state['feedback_summary'] = analysis_summary
# Feedback automatically flows to next iteration
```

### 3. Regression Measurement

Comprehensive tracking of performance changes:
- Metric deltas (precision, recall, F1, accuracy)
- Testcase-level changes (newly broken/fixed)
- Net change score (weighted performance improvement)
- Cumulative progress tracking

```python
# Compare iterations
regression = RegressionAnalysis.compare_runs(previous_run, current_run)

print(f"F1 change: {regression.f1_delta:+.3f}")
print(f"Newly broken: {len(regression.newly_broken_testcases)}")
print(f"Net improvement: {regression.net_change_score:+.3f}")
```

## Visualization and Analysis

### Performance Trends
- F1, precision, recall, accuracy over iterations
- Time-based progression
- Trend analysis with slope calculation

### Regression Analysis
- Performance deltas visualization
- Testcase-level change tracking
- Cumulative improvement graphs

### Error Analysis
- False positive/negative trends
- Error reduction rates
- Pattern identification

### Execution Analysis
- Query execution times
- Success rates
- Efficiency metrics (results per second)

## Integration with Existing System

The experiment system integrates seamlessly with your existing CodeQL infrastructure:

- Uses existing `CodeQLQueryRunner` for query execution
- Works with existing database structure from `split_juliet_testcases.py`
- Maintains compatibility with current query formats
- Extends existing result structures

## Storage and Persistence

All experiment data is stored in JSON format:
- Individual experiment files: `{experiment_id}.json`
- Individual test runs: `runs/{run_id}.json`
- Automatic backup and versioning
- Easy data export and analysis

## Example Workflow

1. **Setup**: Create testcase databases using existing splitter
2. **Initialize**: Create new experiment with target CWE
3. **Iterate**: Run multiple refinement iterations
4. **Analyze**: Generate visualizations and reports
5. **Compare**: Compare different experimental approaches

## Data Model Benefits

This comprehensive model provides:

✅ **Complete Traceability** - Every query, result, and change is tracked  
✅ **Regression Detection** - Automatic identification of performance regressions  
✅ **Performance Trends** - Clear visualization of improvement over time  
✅ **Error Analysis** - Deep dive into false positives and negatives  
✅ **Experimental Comparison** - Compare different approaches and parameters  
✅ **Reproducibility** - Full experimental state can be recreated  
✅ **Scalability** - Efficient storage and retrieval of large experiments  

The system is designed to be both comprehensive for research purposes and practical for iterative query development.
