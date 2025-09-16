#!/usr/bin/env python3
"""
CodeQL Query Refinement Experiment Runner

This module orchestrates the iterative refinement of CodeQL queries using LLM feedback.
It integrates with the existing query runner and implements the experimental workflow.

Key Features:
1. 3-shot query generation with best candidate selection
2. Optional false positive/negative analysis and summarization
3. Regression tracking between iterations
4. Integration with existing CodeQL query runner
5. Comprehensive logging and data collection
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, TypedDict
from datetime import datetime, timezone
import dotenv
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

from experiment_data_model import (
    Experiment, TestRun, TestcaseResult, QueryCandidate, 
    FalsePositiveNegativeAnalysis, RegressionAnalysis,
    TestcaseVariant, QueryGenerationMethod,
    create_new_experiment, create_new_test_run,
    ExperimentStorage
)
from run_codeql_query import CodeQLQueryRunner, QueryResult

dotenv.load_dotenv()


def ensure_qlpack_yml(directory: Path):
    """Ensure qlpack.yml file exists in the given directory"""
    qlpack_file = directory / "qlpack.yml"
    if not qlpack_file.exists():
        # Copy from sample_queries directory
        sample_qlpack = Path(__file__).parent / "sample_queries" / "qlpack.yml"
        if sample_qlpack.exists():
            import shutil
            shutil.copy2(sample_qlpack, qlpack_file)
            print(f"Copied qlpack.yml to {directory}")
        else:
            # Create default qlpack.yml
            qlpack_content = """name: sample-queries
version: 1.0.0
dependencies:
  codeql/java-all: "*"
"""
            with open(qlpack_file, 'w', encoding='utf-8') as f:
                f.write(qlpack_content)
            print(f"Created default qlpack.yml in {directory}")


def extract_codeql_from_response(response_text: str) -> str:
    """Extract CodeQL query from LLM response that may contain markdown code blocks"""
    import re
    
    # First try to find a codeql code block
    codeql_pattern = r'```codeql\s*\n(.*?)\n```'
    match = re.search(codeql_pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fall back to any code block
    code_pattern = r'```[a-zA-Z]*\s*\n(.*?)\n```'
    match = re.search(code_pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code blocks found, return the whole response (maybe it's just the query)
    return response_text.strip()


# LangGraph State Definition
class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: List[HumanMessage | AIMessage | SystemMessage]
    experiment: Experiment
    current_iteration: int
    target_cwe: str
    databases: List[Tuple[str, Path]]
    previous_run: Optional[TestRun]
    query_candidates: List[QueryCandidate]
    best_candidate: Optional[QueryCandidate]
    evaluation_results: List[TestRun]
    current_run: Optional[TestRun]
    feedback_summary: Optional[str]
    should_continue: bool
    max_iterations: int
    validation_failed: bool
    # Retry mechanism fields
    max_validation_retries: int
    current_validation_attempt: int
    validation_feedback: Optional[str]
    # Validation history tracking
    validation_history: List[Tuple[str, str]]  # List of (query_text, validation_result) pairs


# Pydantic Models for Tool Inputs
class QueryGenerationInput(BaseModel):
    """Input for query generation"""
    target_cwe: str = Field(description="Target CWE identifier")
    previous_query: Optional[str] = Field(default=None, description="Previous query for refinement")
    feedback: Optional[str] = Field(default=None, description="Feedback from previous iteration")
    initial_query: Optional[str] = Field(default=None, description="Initial query to start with")
    validation_feedback: Optional[str] = Field(default=None, description="Feedback from validation failures")
    num_candidates: int = Field(default=3, description="Number of query candidates to generate")


class AnalysisInput(BaseModel):
    """Input for false positive/negative analysis"""
    query_text: str = Field(description="The CodeQL query text")
    false_positives: List[str] = Field(description="List of false positive testcase names")
    false_negatives: List[str] = Field(description="List of false negative testcase names")


class QueryValidationInput(BaseModel):
    """Input for CodeQL query compilation validation"""
    query_text: str = Field(description="The CodeQL query text to validate")
    codeql_path: str = Field(default="codeql", description="Path to CodeQL executable")
    timeout: int = Field(default=30, description="Timeout in seconds for compilation check")


# Tools for the LangGraph agent
@tool
def generate_codeql_query(input_data: QueryGenerationInput, model: str) -> str:
    """Generate a CodeQL query for vulnerability detection"""
    
    prompt = f"""
You are an expert CodeQL query developer. Refine and improve the following CodeQL query to better detect {input_data.target_cwe} vulnerabilities in Java code.

Initial query to improve:
```codeql
{input_data.initial_query}
```
Please provide your response with the CodeQL query wrapped in code blocks like this:
```codeql
your query here
```
"""
    

    docs = """
Documentation for the Dataflow Module:    

import semmle.code.java.dataflow.DataFlow
Predicates
exprNode	Gets the node corresponding to e.
getFieldQualifier	Gets the node that occurs as the qualifier of fa.
getInstanceArgument	Gets the instance argument of a non-static call.
hasNonlocalValue	Holds if the FieldRead is not completely determined by explicit SSA updates.
localExprFlow	Holds if data can flow from e1 to e2 in zero or more local (intra-procedural) steps.
localFlow	Holds if data can flow from node1 to node2 in zero or more local (intra-procedural) steps.
localFlowStep	Holds if data can flow from node1 to node2 in one local step.
localMustFlowStep	Holds if the value of node2 is given by node1.
parameterNode	Gets the node corresponding to p.
simpleAstFlowStep	Holds if there is a data flow step from e1 to e2 that only steps from child to parent in the AST.
simpleLocalFlowStep	INTERNAL: do not use.
Classes
AdditionalNode	A node introduced by an extension of AdditionalDataFlowNode.
ArrayContent	A reference through an array.
CapturedVariableContent	A captured variable.
CollectionContent	A reference through the contents of some collection-like container.
Content	A description of the way data may be stored inside an object. Examples include instance fields, the contents of a collection object, or the contents of an array.
ContentSet	An entity that represents a set of Contents.
ExplicitParameterNode	A parameter, viewed as a node in a data flow graph.
ExprNode	An expression, viewed as a node in a data flow graph.
FieldContent	A reference through an instance field.
FieldValueNode	A node representing the value of a field.
ImplicitInstanceAccess	An implicit read of this or A.this.
ImplicitVarargsArray	An implicit varargs array creation expression.
InstanceAccessNode	A node representing an InstanceAccessExt.
InstanceParameterNode	An instance parameter for an instance method or constructor.
MapKeyContent	A reference through a map key.
MapValueContent	A reference through a map value.
Node	An element, viewed as a node in a data flow graph. Either an expression, a parameter, or an implicit varargs array creation.
ParameterNode	An explicit or implicit parameter.
PostUpdateNode	A node associated with an object after an operation that might have changed its state.
SyntheticFieldContent	A reference through a synthetic instance field.

    """
    # Build the comprehensive prompt with all context
    if input_data.initial_query and not input_data.previous_query:
        # First iteration with initial query - focus on refinement
        prompt = f"Initial query to improve:\n```codeql\n{input_data.initial_query}\n```"
    elif input_data.previous_query:
        # Add previous query context (from iterations)
        prompt += f"\n\nPrevious query to improve:\n```codeql\n{input_data.previous_query}\n```"
    
    # Add feedback from analysis
    if input_data.feedback:
        prompt += f"\n\nFeedback from previous iteration:\n{input_data.feedback}"
    
    # Add validation feedback if there were compilation errors
    if input_data.validation_feedback:
        prompt += f"\n\nIMPORTANT: Previous attempts failed validation. Please address these issues:\n{input_data.validation_feedback}"
    # Use OpenRouter to generate the query
    try:
        # Initialize OpenRouter model using ChatOpenAI interface
        llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=2000,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Generate response using chat format
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        # Extract content from the response
        response_text = response.content

        # Extract the CodeQL query from the response
        codeql_query = extract_codeql_from_response(response_text)
        
        return codeql_query
        
    except Exception as e:
        # If OpenRouter fails, provide a helpful error message
        raise RuntimeError(f"Failed to generate CodeQL query using OpenRouter: {str(e)}. Please check your OPENROUTER_API_KEY and network connection.")


@tool
def analyze_false_positives_negatives(input_data: AnalysisInput) -> str:
    """Analyze false positives and negatives to provide improvement suggestions"""
    
    analysis_prompt = f"""
Analyze the CodeQL query results to identify patterns and suggest improvements:

Query:
{input_data.query_text}

False Positives: {len(input_data.false_positives)}
False Negatives: {len(input_data.false_negatives)}

Problematic testcases:
FP: {', '.join(input_data.false_positives[:5])}{'...' if len(input_data.false_positives) > 5 else ''}
FN: {', '.join(input_data.false_negatives[:5])}{'...' if len(input_data.false_negatives) > 5 else ''}
"""
    
    # Mock analysis - in practice, this would be LLM-generated
    fp_patterns = []
    fn_patterns = []
    
    if input_data.false_positives:
        fp_patterns.extend([
            "Test code generating false alarms",
            "Mock implementations triggering detection",
            "Safe wrapper methods being flagged"
        ])
    
    if input_data.false_negatives:
        fn_patterns.extend([
            "Complex data flow not tracked",
            "Indirect vulnerability patterns missed",
            "Alternative API methods not covered"
        ])
    
    suggestions = []
    if fp_patterns:
        suggestions.extend([
            "Add exclusions for test packages",
            "Refine sink definitions to avoid safe wrappers",
            "Improve context analysis"
        ])
    
    if fn_patterns:
        suggestions.extend([
            "Extend data flow analysis depth",
            "Include additional vulnerability patterns",
            "Add more comprehensive source/sink definitions"
        ])
    
    summary = f"""
Analysis Summary:

Common False Positive Patterns:
{chr(10).join(f'- {pattern}' for pattern in fp_patterns)}

Common False Negative Patterns:
{chr(10).join(f'- {pattern}' for pattern in fn_patterns)}

Improvement Suggestions:
{chr(10).join(f'- {suggestion}' for suggestion in suggestions)}

Priority: Focus on {'reducing false positives' if len(input_data.false_positives) > len(input_data.false_negatives) else 'improving coverage'}
"""
    
    return summary


@tool
def validate_codeql_query(input_data: QueryValidationInput) -> str:
    """Validate a CodeQL query by attempting to compile it"""
    import tempfile
    import subprocess
    from pathlib import Path
    
    try:
        # Create a temporary query file in the test_queries directory to use qlpack.yml
        test_queries_dir = Path(__file__).parent / "test_queries"
        test_queries_dir.mkdir(exist_ok=True)
        ensure_qlpack_yml(test_queries_dir)
        
        # Create temporary query file in the test directory
        import uuid
        temp_filename = f"temp_query_{uuid.uuid4().hex[:8]}.ql"
        temp_query_path = test_queries_dir / temp_filename
        
        with open(temp_query_path, 'w', encoding='utf-8') as temp_file:
            temp_file.write(input_data.query_text)
        
        try:
            # Run CodeQL query compile with --check-only flag
            cmd = [
                input_data.codeql_path, "query", "compile", 
                "--check-only",
                "--threads=1",  # Use single thread for validation
                "--search-path=/opt/codeql",  # Add CodeQL library search path
                str(temp_query_path)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=input_data.timeout
            )
            
            if result.returncode == 0:
                return "VALID: Query compiled successfully"
            else:
                # Extract error information
                error_output = result.stderr.strip() or result.stdout.strip()
                return f"INVALID: Compilation failed with error: {error_output}"
                
        finally:
            # Clean up temporary file
            try:
                temp_query_path.unlink()
            except OSError:
                pass
                
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: Query compilation timed out after {input_data.timeout} seconds"
    except Exception as e:
        return f"ERROR: Failed to validate query: {str(e)}"


class LangGraphQueryAgent:
    """LangGraph-based agent for CodeQL query generation and refinement"""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7, api_key: Optional[str] = None, 
                 codeql_path: str = "codeql", max_validation_retries: int = 2):
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        self.max_validation_retries = max_validation_retries
        
        # Create tools
        self.tools = [generate_codeql_query, analyze_false_positives_negatives, validate_codeql_query]
        self.tool_executor = ToolNode(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        self.codeql_path = codeql_path
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generate_candidates", self._generate_candidates_node)
        workflow.add_node("validate_candidates", self._validate_candidates_node)
        workflow.add_node("regenerate_with_feedback", self._regenerate_candidates_with_feedback_node)
        workflow.add_node("evaluate_candidates", self._evaluate_candidates_node)
        workflow.add_node("select_best_candidate", self._select_best_candidate_node)
        workflow.add_node("analyze_errors", self._analyze_errors_node)
        workflow.add_node("check_convergence", self._check_convergence_node)
        workflow.add_node("finalize_iteration", self._finalize_iteration_node)
        
        # Define the flow
        workflow.set_entry_point("generate_candidates")
        
        workflow.add_edge("generate_candidates", "validate_candidates")
        
        # Conditional edges for validation retry logic
        workflow.add_conditional_edges(
            "validate_candidates",
            self._should_retry_validation,
            {
                "retry": "regenerate_with_feedback",
                "proceed": "evaluate_candidates"
            }
        )
        
        workflow.add_edge("regenerate_with_feedback", "validate_candidates")
        workflow.add_edge("evaluate_candidates", "select_best_candidate")
        workflow.add_edge("select_best_candidate", "analyze_errors")
        workflow.add_edge("analyze_errors", "check_convergence")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "check_convergence",
            self._should_continue,
            {
                "continue": "finalize_iteration",
                "stop": END
            }
        )
        
        workflow.add_edge("finalize_iteration", END)
        
        return workflow.compile()
    
    def _generate_candidates_node(self, state: AgentState) -> AgentState:
        """Generate query candidates using the LLM"""
        print(f"Generating query candidates for iteration {state['current_iteration']}...")
        
        # Prepare data for the query generation tool
        feedback = state.get('feedback_summary')
        previous_query = state['previous_run'].query_text if state['previous_run'] else None
        initial_query = None
        
        # Get initial query for first iteration
        if state['current_iteration'] == 1 and hasattr(state['experiment'], 'initial_query'):
            initial_query = state['experiment'].initial_query
        
        # Generate candidates
        candidates = []
        for i in range(3):  # 3-shot generation
            candidate_id = f"candidate_{i+1}_{int(time.time())}_{state['current_iteration']}"
            
            # Use the tool to generate query - all prompt logic is now in the tool
            query_input = QueryGenerationInput(
                target_cwe=state['target_cwe'],
                previous_query=previous_query,
                feedback=feedback,
                initial_query=initial_query,
                validation_feedback=state.get('validation_feedback'),
                num_candidates=1
            )
            
            query_text = generate_codeql_query.invoke({"input_data": query_input, "model": self.model})
            
            candidate = QueryCandidate(
                candidate_id=candidate_id,
                query_text=query_text,
                generation_prompt="Generated using consolidated prompt logic in generate_codeql_query tool",
                generation_timestamp=datetime.now(timezone.utc),
                llm_model=self.model,
                llm_temperature=self.temperature
            )
            candidates.append(candidate)
        
        state['query_candidates'] = candidates
        print(f"Generated {len(candidates)} query candidates")
        
        return state
    
    def _regenerate_candidates_with_feedback_node(self, state: AgentState) -> AgentState:
        """Regenerate query candidates using validation feedback"""
        print(f"Regenerating candidates with validation feedback (attempt {state['current_validation_attempt']})...")
        
        # Initialize validation history if not present
        if 'validation_history' not in state:
            state['validation_history'] = []
        
        # Collect validation errors from the full history, not just current attempt
        validation_history_text = []
        if state['validation_history']:
            validation_history_text.append("Complete validation history (query attempts and results):")
            for i, (query_text, validation_result) in enumerate(state['validation_history'], 1):
                # Truncate very long queries for readability
                query_preview = query_text[:200] + "..." if len(query_text) > 200 else query_text
                validation_history_text.append(f"\nAttempt {i}:")
                validation_history_text.append(f"Query: {query_preview}")
                validation_history_text.append(f"Result: {validation_result}")
        
        # Also collect current attempt errors for immediate context
        current_errors = []
        for candidate in state['query_candidates']:
            if hasattr(candidate, 'validation_result') and not candidate.is_valid:
                current_errors.append(f"Candidate {candidate.candidate_id}: {candidate.validation_result}")
        
        # Create comprehensive feedback message for the LLM
        validation_feedback = f"""
{chr(10).join(validation_history_text)}

Current attempt errors:
{chr(10).join(current_errors)}

Common issues to fix based on history:
- Ensure all imported modules are available (e.g., use 'import java' not specific submodules that may not exist)
- Check syntax for CodeQL query structure
- Verify that all referenced types and predicates are correctly defined
- Make sure the query follows CodeQL best practices
- Learn from previous failed attempts to avoid repeating the same errors

Please generate new query candidates that address these validation errors and avoid patterns that have consistently failed in the history above.
"""
        
        state['validation_feedback'] = validation_feedback
        
        # Prepare data for the query generation tool
        feedback = state.get('feedback_summary')
        previous_query = state['previous_run'].query_text if state['previous_run'] else None
        initial_query = None
        
        # Get initial query for first iteration
        if state['current_iteration'] == 1 and hasattr(state['experiment'], 'initial_query'):
            initial_query = state['experiment'].initial_query
        
        # Generate new candidates
        new_candidates = []
        for i in range(3):  # 3-shot generation
            candidate_id = f"retry_{state['current_validation_attempt']}_candidate_{i+1}_{int(time.time())}_{state['current_iteration']}"
            
            # Use the tool to generate query - all prompt logic is now in the tool
            query_input = QueryGenerationInput(
                target_cwe=state['target_cwe'],
                previous_query=previous_query,
                feedback=feedback,
                initial_query=initial_query,
                validation_feedback=validation_feedback,
                num_candidates=1
            )
            
            query_text = generate_codeql_query.invoke({"input_data": query_input, "model": self.model})
            
            candidate = QueryCandidate(
                candidate_id=candidate_id,
                query_text=query_text,
                generation_prompt="Generated using consolidated prompt logic in generate_codeql_query tool",
                generation_timestamp=datetime.now(timezone.utc),
                llm_model=self.model,
                llm_temperature=self.temperature
            )
            new_candidates.append(candidate)
        
        state['query_candidates'] = new_candidates
        print(f"Generated {len(new_candidates)} new query candidates with validation feedback")
        
        return state
    
    def _validate_candidates_node(self, state: AgentState) -> AgentState:
        """Validate query candidates using CodeQL compilation check"""
        print("Validating query candidates...")
        
        valid_candidates = []
        
        # Initialize validation history if not present
        if 'validation_history' not in state:
            state['validation_history'] = []
        
        for i, candidate in enumerate(state['query_candidates'], 1):
            print(f"Validating candidate {i}/{len(state['query_candidates'])}...")
            
            # Use the validation tool
            validation_input = QueryValidationInput(
                query_text=candidate.query_text,
                codeql_path=self.codeql_path,
                timeout=30
            )
            
            validation_result = validate_codeql_query.invoke({"input_data": validation_input})
            
            # Store validation result in candidate (we'll need to update the data model)
            candidate.validation_result = validation_result
            candidate.is_valid = validation_result.startswith("VALID:")
            
            # Add to validation history
            state['validation_history'].append((candidate.query_text, validation_result))
            
            if candidate.is_valid:
                valid_candidates.append(candidate)
                print(f"  ✅ Candidate {i}: Valid")
            else:
                print(f"  ❌ Candidate {i}: {validation_result}")
        
        # Update state with only valid candidates
        original_count = len(state['query_candidates'])
        if valid_candidates:
            state['query_candidates'] = valid_candidates
            state['validation_failed'] = False
            print(f"Validation complete: {len(valid_candidates)} valid candidates out of {original_count}")
        else:
            print("⚠️  No valid candidates found! All queries failed compilation.")
            # Check if we can retry
            if state['current_validation_attempt'] < state['max_validation_retries']:
                print(f"Will retry validation (attempt {state['current_validation_attempt'] + 1} of {state['max_validation_retries']})")
                state['validation_failed'] = True
                state['current_validation_attempt'] += 1
                # Keep invalid candidates for feedback generation
            else:
                print(f"Maximum validation retries ({state['max_validation_retries']}) reached. Proceeding with failed candidates.")
                state['validation_failed'] = True
                # Keep invalid candidates for debugging
        
        return state
    
    def _evaluate_candidates_node(self, state: AgentState) -> AgentState:
        """Evaluate all query candidates"""
        print("Evaluating query candidates...")
        
        # Check if validation failed and we have no valid candidates
        if state.get('validation_failed', False):
            # Check if we have any valid candidates among the failed ones
            valid_candidates = [c for c in state['query_candidates'] if getattr(c, 'is_valid', True)]
            if not valid_candidates:
                print("⚠️  Skipping evaluation - no valid query candidates after retries")
                state['evaluation_results'] = []
                return state
            else:
                # Use only the valid candidates for evaluation
                state['query_candidates'] = valid_candidates
                print(f"Proceeding with {len(valid_candidates)} valid candidates after retry attempts")
        
        evaluation_results = []
        
        for i, candidate in enumerate(state['query_candidates'], 1):
            print(f"Evaluating candidate {i}/{len(state['query_candidates'])}...")
            
            # Create temporary query file in test_queries directory to use qlpack.yml
            test_queries_dir = Path(__file__).parent / "test_queries"
            test_queries_dir.mkdir(exist_ok=True)
            ensure_qlpack_yml(test_queries_dir)
            temp_query_file = test_queries_dir / f"temp_query_{candidate.candidate_id}.ql"
            try:
                with open(temp_query_file, 'w', encoding='utf-8') as f:
                    f.write(candidate.query_text)
                
                # Run the query using existing CodeQL runner
                query_runner = CodeQLQueryRunner(self.codeql_path)
                query_results = query_runner.run_query(
                    query_file=str(temp_query_file),
                    databases=state['databases'],
                    output_format="sarif-latest",
                    parallel=True
                )
                
                # Convert to our format
                testcase_results = self._convert_query_results_to_testcase_results(
                    query_results, state['target_cwe']
                )
                
                # Create test run
                test_run = create_new_test_run(
                    query_text=candidate.query_text,
                    query_description=f"Evaluation of candidate {candidate.candidate_id}",
                    target_cwe=state['target_cwe'],
                    generation_method=QueryGenerationMethod.THREE_SHOT,
                    llm_model=candidate.llm_model,
                    llm_temperature=candidate.llm_temperature
                )
                
                test_run.testcase_results = testcase_results
                test_run.query_candidates = [candidate]
                test_run.selected_candidate_id = candidate.candidate_id
                test_run.calculate_performance_metrics()
                
                evaluation_results.append(test_run)
                
                if test_run.performance_metrics:
                    print(f"  F1: {test_run.performance_metrics.f1_score:.3f}, "
                          f"P: {test_run.performance_metrics.precision:.3f}, "
                          f"R: {test_run.performance_metrics.recall:.3f}")
                
            finally:
                pass
                # Clean up temporary file
                #if temp_query_file.exists():
                #    temp_query_file.unlink()
        
        state['evaluation_results'] = evaluation_results
        return state
    
    def _select_best_candidate_node(self, state: AgentState) -> AgentState:
        """Select the best performing candidate"""
        print("Selecting best candidate...")
        
        if not state['evaluation_results']:
            if state.get('validation_failed', False):
                print("⚠️  No evaluation results due to validation failures")
                # Create a dummy test run to track the failure
                from experiment_data_model import create_new_test_run, QueryGenerationMethod
                failed_run = create_new_test_run(
                    query_text="# All query candidates failed validation",
                    query_description="Validation failed for all generated candidates",
                    target_cwe=state['target_cwe'],
                    generation_method=QueryGenerationMethod.THREE_SHOT,
                    llm_model=self.model,
                    llm_temperature=self.temperature
                )
                failed_run.query_candidates = state['query_candidates']
                state['current_run'] = failed_run
                state['best_candidate'] = None
                return state
            else:
                raise ValueError("No evaluation results available")
        
        # Sort by F1 score (primary), then by precision (secondary)
        def score_candidate(test_run: TestRun) -> Tuple[float, float]:
            if not test_run.performance_metrics:
                return (0.0, 0.0)
            return (test_run.performance_metrics.f1_score, test_run.performance_metrics.precision)
        
        best_run = max(state['evaluation_results'], key=score_candidate)
        best_candidate = state['query_candidates'][state['evaluation_results'].index(best_run)]
        
        print(f"Selected best candidate with F1={best_run.performance_metrics.f1_score:.3f}, "
              f"Precision={best_run.performance_metrics.precision:.3f}")
        
        # Update the best run with all candidates for tracking
        best_run.query_candidates = state['query_candidates']
        
        state['best_candidate'] = best_candidate
        state['current_run'] = best_run
        
        return state
    
    def _analyze_errors_node(self, state: AgentState) -> AgentState:
        """Analyze false positives and negatives"""
        print("Analyzing false positives and negatives...")
        
        current_run = state['current_run']
        if not current_run:
            return state
        
        false_positives = current_run.get_false_positives()
        false_negatives = current_run.get_false_negatives()
        
        if false_positives or false_negatives:
            # Use the analysis tool
            analysis_input = AnalysisInput(
                query_text=current_run.query_text,
                false_positives=[fp.project_name for fp in false_positives],
                false_negatives=[fn.project_name for fn in false_negatives]
            )
            
            analysis_summary = analyze_false_positives_negatives.invoke({"input_data": analysis_input})
            
            # Create FP/FN analysis object
            fp_fn_analysis = FalsePositiveNegativeAnalysis(
                testcase_results=false_positives + false_negatives,
                analysis_prompt=f"Analyze FP/FN for {current_run.query_description}",
                analysis_summary=analysis_summary,
                analysis_timestamp=datetime.now(timezone.utc),
                analysis_model=self.model,
                common_patterns=[],  # Would be extracted from LLM response
                suggested_improvements=[]  # Would be extracted from LLM response
            )
            
            current_run.fp_fn_analysis = fp_fn_analysis
            state['feedback_summary'] = analysis_summary
            
            print(f"  Found {len(false_positives)} false positives, {len(false_negatives)} false negatives")
        else:
            print("  No false positives or negatives to analyze")
            state['feedback_summary'] = None
        
        return state
    
    def _check_convergence_node(self, state: AgentState) -> AgentState:
        """Check if we should continue iterating"""
        current_run = state['current_run']
        
        # Check convergence criteria
        should_stop = False
        
        if current_run and current_run.performance_metrics:
            # Stop if we achieve excellent F1 score
            if current_run.performance_metrics.f1_score >= 0.95:
                print(f"Achieved excellent F1 score ({current_run.performance_metrics.f1_score:.3f}), stopping")
                should_stop = True
        
        # Check max iterations
        if state['current_iteration'] >= state['max_iterations']:
            print(f"Reached maximum iterations ({state['max_iterations']}), stopping")
            should_stop = True
        
        state['should_continue'] = not should_stop
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Conditional edge function"""
        return "stop" if not state['should_continue'] else "continue"
    
    def _should_retry_validation(self, state: AgentState) -> str:
        """Conditional edge function for validation retry logic"""
        if (state.get('validation_failed', False) and 
            state['current_validation_attempt'] < state['max_validation_retries']):
            return "retry"
        else:
            return "proceed"
    
    def _finalize_iteration_node(self, state: AgentState) -> AgentState:
        """Finalize the current iteration"""
        print(f"Finalizing iteration {state['current_iteration']}")
        
        current_run = state['current_run']
        if current_run:
            # Add to experiment
            state['experiment'].add_test_run(current_run)
            
            # Perform regression analysis if we have a previous run
            if state['previous_run']:
                regression_analysis = RegressionAnalysis.compare_runs(state['previous_run'], current_run)
                current_run.regression_analysis = regression_analysis
                
                print(f"  Performance change: F1 {regression_analysis.f1_delta:+.3f}")
            
            # Update state for next iteration
            state['previous_run'] = current_run
            state['current_iteration'] += 1
        
        return state
    
    def _convert_query_results_to_testcase_results(self, query_results: List[QueryResult],
                                                  target_cwe: str) -> List[TestcaseResult]:
        """Convert CodeQL query results to our testcase result format"""
        testcase_results = []
        
        for result in query_results:
            # Determine variant from project name
            if result.project_name.endswith('_bad'):
                variant = TestcaseVariant.BAD
            elif result.project_name.endswith('_good'):
                variant = TestcaseVariant.GOOD
            else:
                variant = TestcaseVariant.BAD if 'bad' in result.project_name.lower() else TestcaseVariant.GOOD
            
            expected_positive = result.project_name.endswith('_bad')
            actual_positive = result.success and result.result_count > 0
            
            testcase_result = TestcaseResult(
                project_name=result.project_name,
                cwe_id=result.cwe_id or target_cwe,
                variant=variant,
                expected_positive=expected_positive,
                actual_positive=actual_positive,
                result_count=result.result_count,
                execution_time=result.execution_time,
                error_message=result.error_message
            )
            
            testcase_results.append(testcase_result)
        
        return testcase_results
    
    def run_iteration(self, state: AgentState) -> AgentState:
        """Run a single iteration using the LangGraph workflow"""
        return self.graph.invoke(state)


class ExperimentRunner:
    """Main class for running CodeQL query refinement experiments using LangGraph"""
    
    def __init__(self, codeql_path: str = "codeql", max_workers: int = 4,
                 llm_model: str = "gpt-4", storage_dir: str = "./experiment_data",
                 api_key: Optional[str] = None, max_validation_retries: int = 2):
        self.query_runner = CodeQLQueryRunner(codeql_path, max_workers)
        self.agent = LangGraphQueryAgent(model=llm_model, api_key=api_key, codeql_path=codeql_path, 
                                       max_validation_retries=max_validation_retries)
        self.storage = ExperimentStorage(storage_dir)
    
    def run_iteration(self, experiment: Experiment, databases: List[Tuple[str, Path]],
                     initial_prompt: str, previous_run: Optional[TestRun] = None) -> TestRun:
        """Run a single iteration using LangGraph agent"""
        
        # If this is the very first iteration and an initial query is provided, evaluate it first
        try:
            if (previous_run is None and experiment.initial_query and not experiment.test_runs):
                baseline_run, feedback_summary = self._evaluate_initial_query(experiment, databases, experiment.initial_query)
                experiment.add_test_run(baseline_run)
                # Persist baseline before refinement
                self.storage.save_experiment(experiment)
                # Use baseline as previous run and carry feedback into refinement
                previous_run = baseline_run
        except Exception as e:
            print(f"Error during initial query baseline evaluation: {e}")
        
        print(f"\n{'='*60}")
        print(f"Running LangGraph iteration {len(experiment.test_runs) + 1}")
        print(f"{'='*60}")
        
        # Prepare initial state for LangGraph
        initial_state: AgentState = {
            "messages": [SystemMessage(content=initial_prompt)],
            "experiment": experiment,
            "current_iteration": len(experiment.test_runs) + 1,
            "target_cwe": experiment.target_cwe,
            "databases": databases,
            "previous_run": previous_run,
            "query_candidates": [],
            "best_candidate": None,
            "evaluation_results": [],
            "current_run": None,
            # If we created a baseline this turn, surface its feedback to the agent; otherwise None
            "feedback_summary": None,
            "should_continue": True,
            "max_iterations": 1,  # Single iteration per call
            "validation_failed": False,
            # Retry mechanism fields
            "max_validation_retries": self.agent.max_validation_retries,
            "current_validation_attempt": 1,
            "validation_feedback": None,
            # Validation history tracking
            "validation_history": []
        }
        
        # Run the LangGraph workflow
        final_state = self.agent.run_iteration(initial_state)
        
        # Extract the result
        current_run = final_state.get('current_run')
        if not current_run:
            raise RuntimeError("LangGraph agent failed to produce a test run")
        
        # Save experiment state
        self.storage.save_experiment(experiment)
        
        return current_run
    
    def run_experiment(self, experiment_config: Dict[str, Any], 
                      databases: List[Tuple[str, Path]],
                      max_iterations: int = 5) -> Experiment:
        """Run a complete experiment with multiple iterations using LangGraph"""
        
        # Create or load experiment
        if 'experiment_id' in experiment_config:
            experiment = self.storage.load_experiment(experiment_config['experiment_id'])
            if not experiment:
                raise ValueError(f"Experiment {experiment_config['experiment_id']} not found")
        else:
            experiment = create_new_experiment(**experiment_config)
        
        print(f"Starting LangGraph experiment: {experiment.name}")
        print(f"Target CWE: {experiment.target_cwe}")
        print(f"Max iterations: {max_iterations}")
        print(f"Databases: {len(databases)}")
        
        # Build the initial prompt, including initial query if provided
        initial_query = experiment_config.get('initial_query')
        if initial_query:
            initial_prompt = f"""
You are an expert CodeQL query developer. Refine and improve the following CodeQL query to better detect {experiment.target_cwe} vulnerabilities in Java code.

Initial query to improve:
```codeql
{initial_query}
```

Your refined query should:
- Use appropriate data flow analysis
- Include relevant security sinks and sources
- Minimize false positives while maintaining good coverage
- Follow CodeQL best practices
- Target {experiment.target_cwe} vulnerability patterns specifically

Focus on creating a query that balances precision and recall for effective vulnerability detection.
"""
        else:
            initial_prompt = f"""
You are an expert CodeQL query developer. Create a precise CodeQL query to detect {experiment.target_cwe} vulnerabilities in Java code.

Your query should:
- Use appropriate data flow analysis
- Include relevant security sinks and sources
- Minimize false positives while maintaining good coverage
- Follow CodeQL best practices
- Target {experiment.target_cwe} vulnerability patterns specifically

Focus on creating a query that balances precision and recall for effective vulnerability detection.
"""
        
        # If an initial query is provided and this is a new experiment, evaluate it first
        baseline_feedback_summary: Optional[str] = None
        if initial_query and not experiment.test_runs:
            try:
                baseline_run, baseline_feedback_summary = self._evaluate_initial_query(experiment, databases, initial_query)
                experiment.add_test_run(baseline_run)
                # Save after adding baseline
                self.storage.save_experiment(experiment)
                print("Baseline initial query evaluated and added as iteration 1.")
            except Exception as e:
                print(f"Error evaluating initial query baseline: {e}")
        
        # Prepare the LangGraph state for the full experiment
        initial_state: AgentState = {
            "messages": [SystemMessage(content=initial_prompt)],
            "experiment": experiment,
            "current_iteration": len(experiment.test_runs) + 1,
            "target_cwe": experiment.target_cwe,
            "databases": databases,
            "previous_run": experiment.get_latest_run(),
            "query_candidates": [],
            "best_candidate": None,
            "evaluation_results": [],
            "current_run": None,
            # Seed feedback from baseline evaluation, if available
            "feedback_summary": baseline_feedback_summary,
            "should_continue": True,
            "max_iterations": max_iterations,
            "validation_failed": False,
            # Retry mechanism fields
            "max_validation_retries": self.agent.max_validation_retries,
            "current_validation_attempt": 1,
            "validation_feedback": None,
            # Validation history tracking
            "validation_history": []
        }
        
        # Run the full experiment using LangGraph
        try:
            # Create a modified graph for full experiment
            full_experiment_graph = self._build_full_experiment_graph()
            final_state = full_experiment_graph.invoke(initial_state)
            
            # Update experiment with final state
            experiment = final_state['experiment']
            
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
        except Exception as e:
            print(f"Error in experiment: {e}")
        
        # Final summary
        self.print_experiment_summary(experiment)
        
        return experiment
    
    def _build_full_experiment_graph(self) -> StateGraph:
        """Build a LangGraph workflow for the full multi-iteration experiment"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("run_iteration", self._run_iteration_node)
        workflow.add_node("check_experiment_convergence", self._check_experiment_convergence_node)
        workflow.add_node("update_experiment_state", self._update_experiment_state_node)
        
        # Define the flow
        workflow.set_entry_point("run_iteration")
        
        workflow.add_edge("run_iteration", "update_experiment_state")
        workflow.add_edge("update_experiment_state", "check_experiment_convergence")
        
        # Conditional edges for continuing or stopping
        workflow.add_conditional_edges(
            "check_experiment_convergence",
            self._should_continue_experiment,
            {
                "continue": "run_iteration",
                "stop": END
            }
        )
        
        return workflow.compile()

    def _evaluate_initial_query(self, experiment: Experiment, databases: List[Tuple[str, Path]], initial_query: str) -> Tuple[TestRun, Optional[str]]:
        """Evaluate the provided initial query on the dataset and produce a baseline TestRun and feedback summary."""
        # Write the initial query to a temporary .ql file within test_queries so we can reuse qlpack.yml
        test_queries_dir = Path(__file__).parent / "test_queries"
        test_queries_dir.mkdir(exist_ok=True)
        ensure_qlpack_yml(test_queries_dir)
        temp_query_file = test_queries_dir / f"initial_query_baseline_{int(time.time())}.ql"
        with open(temp_query_file, 'w', encoding='utf-8') as f:
            f.write(initial_query)
        try:
            # Run the query across databases
            query_results = self.query_runner.run_query(
                query_file=str(temp_query_file),
                databases=databases,
                output_format="sarif-latest",
                parallel=True
            )
            # Convert results and assemble TestRun
            testcase_results = self.agent._convert_query_results_to_testcase_results(query_results, experiment.target_cwe)
            baseline_run = create_new_test_run(
                query_text=initial_query,
                query_description="Baseline evaluation of initial query",
                target_cwe=experiment.target_cwe,
                generation_method=QueryGenerationMethod.SINGLE_SHOT,
                llm_model=self.agent.model,
                llm_temperature=self.agent.temperature
            )
            baseline_run.testcase_results = testcase_results
            baseline_run.query_candidates = []
            baseline_run.selected_candidate_id = None
            baseline_run.calculate_performance_metrics()
            # Optional FP/FN analysis to generate feedback
            feedback_summary: Optional[str] = None
            false_positives = baseline_run.get_false_positives()
            false_negatives = baseline_run.get_false_negatives()
            if false_positives or false_negatives:
                analysis_input = AnalysisInput(
                    query_text=baseline_run.query_text,
                    false_positives=[fp.project_name for fp in false_positives],
                    false_negatives=[fn.project_name for fn in false_negatives]
                )
                analysis_summary = analyze_false_positives_negatives.invoke({"input_data": analysis_input})
                fp_fn_analysis = FalsePositiveNegativeAnalysis(
                    testcase_results=false_positives + false_negatives,
                    analysis_prompt=f"Analyze FP/FN for {baseline_run.query_description}",
                    analysis_summary=analysis_summary,
                    analysis_timestamp=datetime.now(timezone.utc),
                    analysis_model=self.agent.model,
                    common_patterns=[],
                    suggested_improvements=[]
                )
                baseline_run.fp_fn_analysis = fp_fn_analysis
                feedback_summary = analysis_summary
            return baseline_run, feedback_summary
        finally:
            try:
                if temp_query_file.exists():
                    temp_query_file.unlink()
            except OSError:
                pass
    
    def _run_iteration_node(self, state: AgentState) -> AgentState:
        """Run a single iteration within the full experiment"""
        print(f"\n{'='*60}")
        print(f"Running LangGraph iteration {state['current_iteration']}")
        print(f"{'='*60}")
        
        # Use the single iteration agent
        iteration_state = self.agent.run_iteration(state)
        
        # Merge results back
        state.update(iteration_state)
        
        return state
    
    def _update_experiment_state_node(self, state: AgentState) -> AgentState:
        """Update experiment state after iteration"""
        current_run = state.get('current_run')
        if current_run:
            # Add to experiment
            state['experiment'].add_test_run(current_run)
            
            # Update state for next iteration
            state['previous_run'] = current_run
            state['current_iteration'] += 1
            
            # Reset validation state for next iteration but keep history
            state['validation_failed'] = False
            state['current_validation_attempt'] = 1
            state['validation_feedback'] = None
            # Note: validation_history is preserved across iterations
            
            # Print iteration summary
            if current_run.performance_metrics:
                print(f"\nIteration {current_run.iteration_number} Summary:")
                print(f"  F1 Score: {current_run.performance_metrics.f1_score:.3f}")
                print(f"  Precision: {current_run.performance_metrics.precision:.3f}")
                print(f"  Recall: {current_run.performance_metrics.recall:.3f}")
                if current_run.execution_stats:
                    print(f"  Execution time: {current_run.execution_stats.total_execution_time:.1f}s")
            
            # Save experiment state
            self.storage.save_experiment(state['experiment'])
        
        return state
    
    def _check_experiment_convergence_node(self, state: AgentState) -> AgentState:
        """Check if the full experiment should continue"""
        current_run = state.get('current_run')
        
        # Check convergence criteria
        should_stop = False
        
        if current_run and current_run.performance_metrics:
            # Stop if we achieve excellent F1 score
            if current_run.performance_metrics.f1_score >= 0.95:
                print(f"Achieved excellent F1 score ({current_run.performance_metrics.f1_score:.3f}), stopping experiment")
                should_stop = True
        
        # Check max iterations
        if state['current_iteration'] > state['max_iterations']:
            print(f"Reached maximum iterations ({state['max_iterations']}), stopping experiment")
            should_stop = True
        
        state['should_continue'] = not should_stop
        return state
    
    def _should_continue_experiment(self, state: AgentState) -> str:
        """Conditional edge function for full experiment"""
        return "stop" if not state['should_continue'] else "continue"
    
    def print_experiment_summary(self, experiment: Experiment):
        """Print a comprehensive summary of the experiment"""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY: {experiment.name}")
        print(f"{'='*80}")
        
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Target CWE: {experiment.target_cwe}")
        print(f"Total iterations: {len(experiment.test_runs)}")
        print(f"Duration: {experiment.test_runs[-1].timestamp - experiment.created_timestamp}")
        
        # Performance progression
        print(f"\nPerformance Progression:")
        print(f"{'Iter':<4} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Acc':<6} {'Change':<8}")
        print("-" * 40)
        
        for i, run in enumerate(experiment.test_runs, 1):
            if run.performance_metrics:
                metrics = run.performance_metrics
                change = ""
                if run.regression_analysis:
                    change = f"{run.regression_analysis.f1_delta:+.3f}"
                
                print(f"{i:<4} {metrics.f1_score:<6.3f} {metrics.precision:<6.3f} "
                      f"{metrics.recall:<6.3f} {metrics.accuracy:<6.3f} {change:<8}")
        
        # Trend analysis
        f1_slope, f1_direction = experiment.calculate_trend('f1_score')
        precision_slope, precision_direction = experiment.calculate_trend('precision')
        
        print(f"\nTrend Analysis:")
        print(f"  F1 Score: {f1_direction} (slope: {f1_slope:+.4f})")
        print(f"  Precision: {precision_direction} (slope: {precision_slope:+.4f})")
        
        # Best performing iteration
        best_run = max(experiment.test_runs, 
                      key=lambda r: r.performance_metrics.f1_score if r.performance_metrics else 0)
        
        print(f"\nBest Iteration: {best_run.iteration_number}")
        if best_run.performance_metrics:
            print(f"  F1: {best_run.performance_metrics.f1_score:.3f}")
            print(f"  Precision: {best_run.performance_metrics.precision:.3f}")
            print(f"  Recall: {best_run.performance_metrics.recall:.3f}")
        
        # Regression summary
        total_regressions = sum(r.regression_analysis.total_regressions 
                              for r in experiment.test_runs if r.regression_analysis)
        total_improvements = sum(r.regression_analysis.total_improvements 
                               for r in experiment.test_runs if r.regression_analysis)
        
        print(f"\nRegression Summary:")
        print(f"  Total regressions: {total_regressions}")
        print(f"  Total improvements: {total_improvements}")
        print(f"  Net improvement: {total_improvements - total_regressions}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CodeQL query refinement experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--description', required=True, help='Experiment description')
    parser.add_argument('--target-cwe', required=True, help='Target CWE (e.g., CWE89)')
    parser.add_argument('--db-root', required=True, help='Root directory containing CodeQL databases')
    parser.add_argument('--initial-query', help='Path to file containing initial CodeQL query to start with (optional)')
    
    parser.add_argument('--max-iterations', type=int, default=5, help='Maximum iterations')
    parser.add_argument('--no-three-shot', action='store_true', help='Disable 3-shot generation')
    parser.add_argument('--no-fp-fn-analysis', action='store_true', help='Disable FP/FN analysis')
    
    parser.add_argument('--codeql-path', default='codeql', help='Path to CodeQL executable')
    parser.add_argument('--max-workers', type=int, default=4, help='Max parallel workers')
    parser.add_argument('--llm-model', default='openai/gpt-4o', help='LLM model to use')
    parser.add_argument('--storage-dir', default='./experiment_data', help='Data storage directory')
    parser.add_argument('--max-validation-retries', type=int, default=4, help='Maximum number of retry attempts when query validation fails')
    
    # Database filtering
    parser.add_argument('--limit', type=int, help='Limit number of databases')
    
    args = parser.parse_args()
    
    try:
        # Initialize experiment runner with LangGraph
        runner = ExperimentRunner(
            codeql_path=args.codeql_path,
            max_workers=args.max_workers,
            llm_model=args.llm_model,
            storage_dir=args.storage_dir,
            api_key=os.getenv('OPENAI_API_KEY'),
            max_validation_retries=args.max_validation_retries
        )
        
        # Find databases
        filters = {}
        if args.limit:
            filters['limit'] = args.limit
        
        databases = runner.query_runner.find_databases(args.db_root, filters)
        if not databases:
            print("No databases found matching criteria")
            return 1
        
        # Read initial query from file if provided
        initial_query = None
        if args.initial_query:
            try:
                with open(args.initial_query, 'r', encoding='utf-8') as f:
                    initial_query = f.read().strip()
                print(f"Loaded initial query from: {args.initial_query}")
            except FileNotFoundError:
                print(f"Error: Initial query file not found: {args.initial_query}")
                return 1
            except Exception as e:
                print(f"Error reading initial query file {args.initial_query}: {e}")
                return 1
        
        # Configure experiment
        experiment_config = {
            'name': args.name,
            'description': args.description,
            'target_cwe': args.target_cwe,
            'use_three_shot': not args.no_three_shot,
            'use_fp_fn_analysis': not args.no_fp_fn_analysis,
            'llm_model': args.llm_model,
            'initial_query': initial_query
        }
        
        # Run experiment
        experiment = runner.run_experiment(experiment_config, databases, args.max_iterations)
        
        print(f"\nExperiment completed: {experiment.experiment_id}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

