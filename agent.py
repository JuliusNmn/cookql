
# pip install langchain langchain-openai langchain-community
from langchain_openai import ChatOpenAI
from langchain_community.tools import WriteFileTool, ReadFileTool, ListDirectoryTool
from memory_file_tools import FileStore, WriteFileTool, ReadFileTool, ListDirectoryTool
# pip install langgraph langchain-openai
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from run_codeql_query import CodeQLQueryRunner
import os
import json
import subprocess
from datetime import datetime
import dotenv
dotenv.load_dotenv()
from prompting import validation_error_repair_prompt, describe_failures, build_refinement_prompt
from langchain_core.runnables import RunnableConfig


class CookQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.8,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.file_store = FileStore()
        self.write_file_tool = WriteFileTool(self.file_store)
        self.read_file_tool = ReadFileTool(self.file_store)
        self.list_directory_tool = ListDirectoryTool(self.file_store)

        self.tools = [self.write_file_tool, self.read_file_tool, self.list_directory_tool]

        self.agent = create_react_agent(self.llm, self.tools)
        self.query_runner = CodeQLQueryRunner(codeql_path=os.getenv("CODEQL_PATH"), max_workers=32)

        # Enable console streaming of LLM outputs (set COOKQL_STREAM=0 to disable)
        self.stream_llm = str(os.getenv("COOKQL_STREAM", "1")).lower() not in ("0", "false", "no")

        # Prepare filters
        self.filters = {}
        self.filters['cwe'] = "CWE36" # "CWE89"
        self.filters['limit'] = 20
        # Find databases
        self.databases = self.query_runner.find_databases(os.getenv("DATASET_PATH"), self.filters)

        self.app = self._build_graph()
        self.event_counter = 0
        self.prev_event_dir = None

    def write_draft(self, state):
        return state

    def _ensure_log_base_dir(self):
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        base_dir = os.path.join("/home/julius/cookql", "logs", date_str)
        os.makedirs(base_dir, exist_ok=True)
        self.log_base_dir = base_dir

    def _safe_relpath(self, path):
        try:
            return path.lstrip(os.sep) if os.path.isabs(path) else path
        except Exception:
            return str(path)

    def _write_text_file(self, filepath, content):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else str(content))

    def _log_event(self, state, step_name, queries: dict, additional_data: dict = {}):
        if not hasattr(self, "log_base_dir"):
            self._ensure_log_base_dir()

        run_num = state.get("run_count")
        val_num = state.get("validation_count")
        event_dir_name = f"event_{self.event_counter}_{step_name}"
        event_dir = os.path.join(self.log_base_dir, event_dir_name)
        os.makedirs(event_dir, exist_ok=True)

        for key, value in additional_data.items():
            if isinstance(value, str):
                self._write_text_file(os.path.join(event_dir, f"{key}.txt"), value)
            else:
                try:
                    self._write_text_file(os.path.join(event_dir, f"{key}.json"), json.dumps(value, indent=2))
                except Exception:
                    self._write_text_file(os.path.join(event_dir, f"{key}.txt"), str(value))

        # Queries snapshot (JSON and individual files)
        try:
            self._write_text_file(os.path.join(event_dir, "queries.json"), json.dumps(queries, indent=2))
        except Exception:
            self._write_text_file(os.path.join(event_dir, "queries.txt"), str(queries))

        files_root = os.path.join(event_dir, "files")
        for filename, file_content in queries.items():
            rel_path = self._safe_relpath(filename)
            dest_path = os.path.join(files_root, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            try:
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
            except Exception:
                # Fallback to writing a string representation
                with open(dest_path + ".txt", "w", encoding="utf-8") as f:
                    f.write(str(file_content))

        # Generate diffs versus previous step's snapshot (if any)
        prev_event_dir = self.prev_event_dir
        try:
            if prev_event_dir and os.path.isdir(prev_event_dir):
                prev_files_dir = os.path.join(prev_event_dir, "files")
                curr_files_dir = files_root
                # Record previous path for traceability
                self._write_text_file(os.path.join(event_dir, "previous_log_dir.txt"), prev_event_dir)

                # Unified diff of file contents and structure
                diff_patch_path = os.path.join(event_dir, "diff.patch")
                try:
                    proc = subprocess.run(["diff", "-ruN", prev_files_dir, curr_files_dir], capture_output=True, text=True)
                    diff_output = proc.stdout or ""
                    if not diff_output:
                        diff_output = "No differences"
                    self._write_text_file(diff_patch_path, diff_output)
                except Exception as e:
                    self._write_text_file(diff_patch_path, f"Failed to run diff: {e}")

                # Quick list of changed files
                changed_list_path = os.path.join(event_dir, "changed_files.txt")
                try:
                    proc2 = subprocess.run(["diff", "-qr", prev_files_dir, curr_files_dir], capture_output=True, text=True)
                    changed_output = proc2.stdout or ""
                    if not changed_output:
                        changed_output = "No file changes"
                    self._write_text_file(changed_list_path, changed_output)
                except Exception as e:
                    self._write_text_file(changed_list_path, f"Failed to run diff -qr: {e}")
        except Exception:
            # Do not fail the flow on logging issues
            pass

        # Update pointer to previous event dir for this step
        self.prev_event_dir = event_dir
        self.event_counter += 1

    def validate(self, state):
        print("Validating queries")
        queries = state["queries_to_validate"]
        self.file_store.set_all_files(queries)
        success, result = self.file_store.write_files_to_tmp_dir()
        print(f"Write result: {result}")
        success, message = self.query_runner.validate_query(result)
        state["validation_success"] = "success" if success else "failed"
        state["validation_message"] = message
        print(f"Validation message: {message}")
        state["validation_count"] = state.get("validation_count") + 1
        self._log_event(state, "validated_query", queries, {"validation_success": success, "validation_message": message})
        if success:
            state["validated_queries"] = queries
        return state
    
    def evaluate_query(self, state):
        queries = state["validated_queries"]
        state["evaluated_queries"] = queries
        self.file_store.set_all_files(queries)
        success, result = self.file_store.write_files_to_tmp_dir()
        #print(f"Write result: {result}")
        results = self.query_runner.run_query(result, self.databases)
        state["results"] = results
        self.query_runner.save_results(result + "/results.json", format="json", include_summary=True)
        state["run_count"] = state.get("run_count", 0) + 1
        state["validation_count"] = 0
        # Log outputs and LLM I/O
        self._log_event(state, "evaluated_query", queries, {"results": results})
        return state

    def refine_evaluated_query(self, state):
        validation_message = state["validation_message"]
        print(validation_message)
        # Build refinement prompt from the results analysis
        analysis = state.get("results_analysis")
        current_files = state.get("evaluated_queries")
        prompt = build_refinement_prompt(analysis, current_files)
        print(prompt)

        # Ensure in-memory store reflects current state before invoking the agent
        self.file_store.set_all_files(current_files)

        # Invoke the LLM agent to refine the queries (with optional streaming)
        agent_result = None
        try:
            print("INVOKING AGENT")
            if self.stream_llm:
                agent_result = self._invoke_agent_with_console_stream(prompt)
            else:
                agent_result = self.agent.invoke({"messages": [HumanMessage(content=prompt)]})
        except Exception as e:
            agent_result = f"Agent invocation failed (refine): {e}"
            print(agent_result)
        print("AGENT INVOKED")
        # Sync modified files back into state
        updated_files = self.file_store.get_all_files()
        state["refined_query"] = updated_files
        state["queries_to_validate"] = updated_files
        state["validation_count"] = 0
        # Extract assistant text for logging when available
        try:
            msgs = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
            if msgs:
                last_msg = msgs[-1]
                agent_result = getattr(last_msg, "content", str(last_msg))
        except Exception:
            pass
        # Log outputs and LLM I/O
        self._log_event(state, "refined_evaluated_query", updated_files, {"llm_input": prompt, "llm_output": agent_result})
        return state

    def fix_validation_errors(self, state):
        errors = state.get("validation_message")
        # Build a repair prompt from validation errors and current files
        current_files = state.get("queries_to_validate")
        prompt = validation_error_repair_prompt(errors, current_files)

        # Ensure in-memory store reflects current state before invoking the agent
        self.file_store.set_all_files(current_files)

        # Invoke the LLM agent with the constructed prompt to modify files via tools (with optional streaming)
        agent_result = None
        try:
            if self.stream_llm:
                agent_result = self._invoke_agent_with_console_stream(prompt)
            else:
                agent_result = self.agent.invoke({"messages": [HumanMessage(content=prompt)]})
        except Exception as e:
            agent_result = f"Agent invocation failed: {e}"
            print(agent_result)

        # Sync modified files back into state
        updated_files = self.file_store.get_all_files()
        state["queries_to_validate"] = updated_files
        # Extract assistant text for logging when available
        try:
            msgs = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
            if msgs:
                last_msg = msgs[-1]
                agent_result = getattr(last_msg, "content", str(last_msg))
        except Exception:
            pass
        self._log_event(state, "fixed_validation_errors", updated_files, {"llm_input": prompt, "llm_output": agent_result})
        return state

    def _invoke_agent_with_console_stream(self, prompt):
        """Invoke the agent and stream AI messages to console as they are produced.

        Returns a dict matching the agent's usual return shape: {"messages": [...]}
        """
        final_messages = []
        try:
            for step in self.agent.stream({"messages": [HumanMessage(content=prompt)]}, stream_mode="messages"):
                # Two possible shapes depending on library version:
                # 1) step is a dict with {"messages": [BaseMessage, ...]}
                # 2) step is a single BaseMessage (message-level streaming)
                if isinstance(step, dict):
                    msgs = step.get("messages", [])
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                else:
                    msgs = [step]
                for msg in msgs:
                    try:
                        final_messages.append(msg)
                        role = getattr(msg, "type", None) or getattr(msg, "role", None)
                        if role == "ai":
                            text = getattr(msg, "content", "")
                            if text:
                                print(text, flush=True)
                    except Exception:
                        pass
            return {"messages": final_messages}
        except Exception as e:
            print(f"Agent streaming failed: {e}")
            return {"messages": []}

    def process_results(self, state):
        print("Processing results")
        results = state["results"]
        for result in results:
            print(result)

        
        print(state["results"])
        # Build analysis string for false positives/negatives and attach to state
        analysis = describe_failures(self.databases, results)
        state["results_analysis"] = analysis
        self._log_event(state, "process_results", state["evaluated_queries"], {"results":results, "results_analysis": analysis})
        print(analysis)
        return state

    def should_continue(self, state):
        if state.get("run_count", 0) >= 4: return "stop"
        return "continue"
    def process_validation(self, state):
        validation_success = state.get("validation_success")
        if validation_success == "success":
            return "success"
        else:
            if state.get("validation_count") >= 3:
                return "failed_max_retries"
            else:
                return "failed_retry"
    def _build_graph(self):
        g = StateGraph(dict)
        g.add_node("write_draft", self.write_draft)
        g.add_node("validate", self.validate)
        g.add_node("fix_validation_errors", self.fix_validation_errors)
        g.add_node("evaluate_query", self.evaluate_query)
        g.add_node("process_results", self.process_results)
        g.add_node("refine_evaluated_query", self.refine_evaluated_query)

        g.add_edge("write_draft", "validate")
        # if validation fixing failed twice, backtrack.
        g.add_conditional_edges("validate", self.process_validation, {"success": "evaluate_query", "failed_retry": "fix_validation_errors", "failed_max_retries": "process_results"})
        g.add_conditional_edges("evaluate_query", self.should_continue, {"continue": "process_results", "stop": END})
        g.add_edge("process_results", "refine_evaluated_query")

        g.add_edge("fix_validation_errors", "validate")
        g.add_edge("refine_evaluated_query", "validate")

        # enter feedback loop at validation of initial query
        g.set_entry_point("validate")

        return g.compile()

    def run(self):
        # Ensure log directory for this run's date exists
        self._ensure_log_base_dir()
        #self.file_store.load_files_from_dir("/home/julius/cookql/sqli_initial")
        self.file_store.load_files_from_dir("/home/julius/cookql/cwe36_initial")
        initial_files = self.file_store.get_all_files()

        initial_state = {"run_count": 0, "validation_count": 0, "queries_to_validate": initial_files}
        config = RunnableConfig(recursion_limit=50)
        result = self.app.invoke(initial_state, config=config)
        #print(result)


# Instantiate and execute immediately to preserve original runtime behavior
app_instance = CookQLAgent()
app_instance.run()

