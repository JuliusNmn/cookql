# memory_file_tools.py

import os
import tempfile
import shutil
from typing import Dict, Optional, Type, Tuple
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# -------------------------------
# FileStore: In-Memory File System
# -------------------------------

class FileStore:
    """In-memory file storage backend."""

    def __init__(self):
        self._files: Dict[str, str] = {}

    def write_file(self, filename: str, text: str, append: bool = False) -> str:
        if append and filename in self._files:
            self._files[filename] += text
        else:
            self._files[filename] = text
        return f"File written successfully to {filename}."

    def read_file(self, filename: str) -> str:
        if filename not in self._files:
            return f"Error: no such file or directory: {filename}"
        return self._files[filename]

    def list_files(self) -> str:
        if not self._files:
            return "No files found."
        return "\n".join(sorted(self._files.keys()))

    def get_all_files(self) -> Dict[str, str]:
        return dict(self._files)
    
    def set_all_files(self, files: Dict[str, str]):
        self._files = files

    def load_files_from_dir(self, directory: str) -> str:
        """Load all files from a directory recursively using os.walk."""
        if not os.path.exists(directory):
            return f"Error: Directory {directory} does not exist."
        
        if not os.path.isdir(directory):
            return f"Error: {directory} is not a directory."
        
        loaded_count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Store with relative path from the given directory
                    rel_path = os.path.relpath(file_path, directory)
                    self._files[rel_path] = content
                    loaded_count += 1
                except Exception as e:
                    # Skip files that can't be read (binary files, permission issues, etc.)
                    continue
        
        return f"Successfully loaded {loaded_count} files from {directory}."

    def write_files_to_tmp_dir(self) -> Tuple[bool, str]:
        """Write all files to a temporary directory and return the path."""
        if not self._files:
            return False,"Error: No files to write."
        
        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp(prefix="file_store_")
        
        try:
            for file_path, content in self._files.items():
                # Create the full path in the temp directory
                full_path = os.path.join(tmp_dir, file_path)
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                # Write file content
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return True, tmp_dir
        except Exception as e:
            # Clean up on error
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, f"Error writing files to temporary directory: {str(e)}"

    


# -------------------------------
# Tools
# -------------------------------

class WriteFileInput(BaseModel):
    file_path: str = Field(..., description="name of file")
    text: str = Field(..., description="text to write to file")
    append: bool = Field(
        default=False, description="Whether to append to an existing file."
    )


class WriteFileTool(BaseTool):
    name: str = "write_file"
    args_schema: Type[BaseModel] = WriteFileInput
    description: str = "Write file to the in-memory store"
    file_store: FileStore
    def __init__(self, file_store: FileStore):
        super().__init__(file_store=file_store)

    def _run(
        self,
        file_path: str,
        text: str,
        append: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        print("Writing file " + file_path)
        return self.file_store.write_file(file_path, text, append)


class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="name of file")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    args_schema: Type[BaseModel] = ReadFileInput
    description: str = "Read file from the in-memory store"
    file_store: FileStore
    def __init__(self, file_store: FileStore):
        super().__init__(file_store=file_store)

    def _run(
        self,
        file_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        print("Reading file " + file_path)
        return self.file_store.read_file(file_path)


class ListDirectoryTool(BaseTool):
    name: str = "list_directory"
    args_schema: Type[BaseModel] = BaseModel  # no args
    description: str = "List all files in the in-memory store"
    file_store: FileStore   
    def __init__(self, file_store: FileStore):
        super().__init__(file_store=file_store)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        print("Listing directory")
        return self.file_store.list_files()


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    # Create in-memory file store
    file_store = FileStore()

    # Initialize tools with the file store
    tools = [
        WriteFileTool(file_store),
        ReadFileTool(file_store),
        ListDirectoryTool(file_store),
    ]

    file_store.write_file("hello.py", "print('Goodbye, world!')")
    file_store.write_file("subdir/readme.md", "# Test Project\nThis is a test.")

    # Test the new methods
    print("Testing load_files_from_dir and write_files_to_tmp_dir:")
    
    # First, write current files to a temp directory
    result = file_store.write_files_to_tmp_dir()
    print(f"Write result: {result}")
    
    # Clear the file store
    file_store.set_all_files({})
    print(f"Files after clearing: {file_store.list_files()}")
    
    # Load files back from the temp directory (extract the path from the result)
    if "Files written to temporary directory:" in result:
        tmp_path = result.split("Files written to temporary directory: ")[1]
        load_result = file_store.load_files_from_dir(tmp_path)
        print(f"Load result: {load_result}")
        print(f"Files after loading: {file_store.list_files()}")
        
        # Clean up the temp directory
        shutil.rmtree(tmp_path, ignore_errors=True)
        print("Temporary directory cleaned up.")

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

    # Agent with tools
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

    # Example operation: create hello.py + README.md
    agent.invoke({
        "input": "update the python fileto print 'Hello world' instead of goodbye, and add a README.md file that explains how to run it"
    })

    # Inspect in-memory file store
    print("\nAll files in store:")
    for filename, content in file_store.get_all_files().items():
        print(f"{filename}: {repr(content)}")