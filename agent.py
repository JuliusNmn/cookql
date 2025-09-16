
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


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, api_key=os.getenv("OPENAI_API_KEY"))
file_store = FileStore()
write_file_tool = WriteFileTool(file_store)
read_file_tool = ReadFileTool(file_store)
list_directory_tool = ListDirectoryTool(file_store)

tools = [write_file_tool, read_file_tool, list_directory_tool]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
query_runner = CodeQLQueryRunner(codeql_path=os.getenv("CODEQL_PATH"))


# Prepare filters
filters = {}

filters['cwe'] = "CWE89"
#filters['limit'] = 4
# Find databases
databases = query_runner.find_databases(os.getenv("DATASET_PATH"), filters)

def write_draft(state):    
    return state

def validate(state):
    queries = state["queries"]
    file_store.set_all_files(queries)
    success, result = file_store.write_files_to_tmp_dir()
    print(f"Write result: {result}")
    success, message = query_runner.validate_query(result)
    state["validation_success"] = "success" if success else "failed" 
    state["validation_message"] = message
    return state

def evaluate_query(state):


    queries = state["queries"]
    file_store.set_all_files(queries)
    success, result = file_store.write_files_to_tmp_dir()
    print(f"Write result: {result}")
    results = query_runner.run_query(result, databases)
    query_runner.save_results(result + "/results.json", format="json", include_summary=True)
    
    state["run_count"] = state.get("run_count", 0) + 1
    return state

def refine_evaluated_query(state):
    
    pass

def fix_validation_errors(state):
    errors = state.get("validation_message", [])
    pass
def process_results(state):
    pass

def should_continue(state):
    if state.get("run_count", 0) >= 3: return "stop"
    return "continue"


    
g = StateGraph(dict)
g.add_node("write_draft", write_draft)
g.add_node("validate", validate)
g.add_node("fix_validation_errors",fix_validation_errors)
g.add_node("evaluate_query",evaluate_query)
g.add_node("process_results", process_results)
g.add_node("refine_evaluated_query", refine_evaluated_query)


g.add_edge("write_draft", "validate")

g.add_conditional_edges("validate", lambda state: state.get("validation_success", "failed"), {"success": "evaluate_query", "failed": "fix_validation_errors"})
g.add_conditional_edges("evaluate_query", should_continue, {"continue": "process_results", "stop": END})
g.add_edge("process_results", "refine_evaluated_query")

g.add_edge("fix_validation_errors", "validate")
g.add_edge("refine_evaluated_query", "validate")

# enter feedback loop at validation of initial query
g.set_entry_point("validate")

app = g.compile()


file_store.load_files_from_dir("/home/julius/qlcook/extract_testcases/sqli_initial")
initial_files = file_store.get_all_files()

initial_state = {"run_count": 0, "queries": initial_files}

result = app.invoke(initial_state)
print(result["draft"])

