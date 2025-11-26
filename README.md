# CookQL: Static Analysis Agent 🧑‍🍳

This repo contains preliminary work for CookQL, a CodeQL writing LLM Agent.

## CodeQL standard Query evaluation

There is a script to benchmark the stock CodeQL CWE queries on the Juliet Dataset. 
Because the Juliet dataset bundles good and bad testcases in single classes, preprocesing is needed.
`split_juliet_testcases.py` (and `extract_juliet_testcases.py`) processes the Juliet dataset so that each testcase is split into multiple isolated good/bad testcases.
`build_cwe_matrix.py` runs CodeQL against the preprocessed dataset and outputs a csv table visualizing the coverage and success rate of CodeQL.
`run_codeql_query.py` is a utility to run a given CodeQL query/a set of queries on the processed dataset.

`results_smart.csv` is the result of `build_cwe_matrix.py` for a representative subset of Juliet. 

## Agent
`agent.py` contains a prototype for an agent to iteratively refine CodeQL queries. This doesn't work yet, as the stock queries are too complex for current LLMs to understand.
`memory_file_tools.py` contains langgraph tools that allow the agent to edit virtual files.
