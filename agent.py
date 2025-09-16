
# pip install langchain langchain-openai langchain-community
from langchain_openai import ChatOpenAI
from langchain_community.tools import WriteFileTool, ReadFileTool, ListDirectoryTool
from langchain.agents import initialize_agent, AgentType
from memory_file_tools import FileStore, WriteFileTool, ReadFileTool, ListDirectoryTool
# pip install langgraph langchain-openai
from langgraph.graph import StateGraph, END
from run_codeql_query import CodeQLQueryRunner
import os
import dotenv
dotenv.load_dotenv()


class CookQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, api_key=os.getenv("OPENAI_API_KEY"))
        self.file_store = FileStore()
        self.write_file_tool = WriteFileTool(self.file_store)
        self.read_file_tool = ReadFileTool(self.file_store)
        self.list_directory_tool = ListDirectoryTool(self.file_store)

        self.tools = [self.write_file_tool, self.read_file_tool, self.list_directory_tool]

        self.agent = initialize_agent(self.tools, self.llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        self.query_runner = CodeQLQueryRunner(codeql_path=os.getenv("CODEQL_PATH"))

        # Prepare filters
        self.filters = {}
        self.filters['cwe'] = "CWE89"
        # self.filters['limit'] = 4
        # Find databases
        self.databases = self.query_runner.find_databases(os.getenv("DATASET_PATH"), self.filters)

        self.app = self._build_graph()

    def write_draft(self, state):
        return state

    def validate(self, state):
        queries = state["queries"]
        self.file_store.set_all_files(queries)
        success, result = self.file_store.write_files_to_tmp_dir()
        print(f"Write result: {result}")
        success, message = self.query_runner.validate_query(result)
        state["validation_success"] = "success" if success else "failed"
        state["validation_message"] = message
        return state

    def evaluate_query(self, state):
        queries = state["queries"]
        self.file_store.set_all_files(queries)
        success, result = self.file_store.write_files_to_tmp_dir()
        print(f"Write result: {result}")
        results = self.query_runner.run_query(result, self.databases)
        self.query_runner.save_results(result + "/results.json", format="json", include_summary=True)

        state["run_count"] = state.get("run_count", 0) + 1
        return state

    def refine_evaluated_query(self, state):
        pass

    def fix_validation_errors(self, state):
        errors = state.get("validation_message", [])
        pass

    def process_results(self, state):
        pass

    def should_continue(self, state):
        return "stop"
        if state.get("run_count", 0) >= 3: return "stop"
        return "continue"

    def _build_graph(self):
        g = StateGraph(dict)
        g.add_node("write_draft", self.write_draft)
        g.add_node("validate", self.validate)
        g.add_node("fix_validation_errors", self.fix_validation_errors)
        g.add_node("evaluate_query", self.evaluate_query)
        g.add_node("process_results", self.process_results)
        g.add_node("refine_evaluated_query", self.refine_evaluated_query)

        g.add_edge("write_draft", "validate")

        g.add_conditional_edges("validate", lambda state: state.get("validation_success", "failed"), {"success": "evaluate_query", "failed": "fix_validation_errors"})
        g.add_conditional_edges("evaluate_query", self.should_continue, {"continue": "process_results", "stop": END})
        g.add_edge("process_results", "refine_evaluated_query")

        g.add_edge("fix_validation_errors", "validate")
        g.add_edge("refine_evaluated_query", "validate")

        # enter feedback loop at validation of initial query
        g.set_entry_point("validate")

        return g.compile()

    def run(self):
        self.file_store.load_files_from_dir("/home/julius/cookql/sqli_initial")
        initial_files = self.file_store.get_all_files()

        initial_state = {"run_count": 0, "queries": initial_files}

        result = self.app.invoke(initial_state)
        print(result["draft"])


# Instantiate and execute immediately to preserve original runtime behavior
app_instance = CookQLAgent()
app_instance.run()

