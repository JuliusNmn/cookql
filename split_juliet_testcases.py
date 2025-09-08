#!/usr/bin/env python3
"""
Juliet Test Suite Testcase Splitter

This script uses the javalang library to parse Java files and extract good and bad
testcases into separate standalone Java projects.

For each testcase, it creates two versions:
- <testcase_name>_bad: Contains only the bad() method and supporting code
- <testcase_name>_good: Contains only the good() methods and supporting code

Each extracted testcase becomes a standalone Java project with proper package structure.
"""

import os
import re
import argparse
import shutil
import subprocess
import random
import concurrent.futures
import threading
import multiprocessing
import time
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import javalang
from dataclasses import dataclass
from extract_juliet_testcases import JulietTestSuiteAnalyzer, TestCase
import sys

@dataclass
class ExtractedMethod:
    """Represents an extracted method with its metadata."""
    name: str
    node: javalang.tree.MethodDeclaration
    is_main: bool = False
    is_helper: bool = False


class JavaCodeGenerator:
    """Generates Java source code from AST nodes."""
    
    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "  # 4 spaces
    
    def indent(self) -> str:
        """Get current indentation string."""
        return self.indent_str * self.indent_level
    
    def generate_compilation_unit(self, tree: javalang.tree.CompilationUnit, 
                                methods_to_keep: Set[str], 
                                class_suffix: str = "") -> str:
        """Generate Java source code for a compilation unit with filtered methods."""
        lines = []
        
        # Package declaration
        if tree.package:
            lines.append(f"package {tree.package.name};")
            lines.append("")
        
        # Imports
        for import_decl in tree.imports:
            if import_decl.static:
                lines.append(f"import static {import_decl.path};")
            else:
                lines.append(f"import {import_decl.path};")
        
        if tree.imports:
            lines.append("")
        
        # Type declarations (classes)
        for type_decl in tree.types:
            if isinstance(type_decl, javalang.tree.ClassDeclaration):
                class_code = self.generate_class(type_decl, methods_to_keep, class_suffix)
                lines.append(class_code)
        
        return "\n".join(lines)
    
    def generate_class(self, class_decl: javalang.tree.ClassDeclaration, 
                      methods_to_keep: Set[str], 
                      class_suffix: str = "") -> str:
        """Generate Java source code for a class with filtered methods."""
        lines = []
        
        # Class declaration
        modifiers = " ".join(class_decl.modifiers) if class_decl.modifiers else ""
        class_name = class_decl.name + class_suffix
        
        class_line = f"{modifiers} class {class_name}".strip()
        
        if class_decl.extends:
            class_line += f" extends {class_decl.extends.name}"
        
        if class_decl.implements:
            implements_list = [impl.name for impl in class_decl.implements]
            class_line += f" implements {', '.join(implements_list)}"
        
        lines.append(f"{class_line} {{")
        self.indent_level += 1
        
        # Process body declarations (includes inner classes, fields, methods)
        if hasattr(class_decl, 'body') and class_decl.body:
            for body_item in class_decl.body:
                if isinstance(body_item, javalang.tree.ClassDeclaration):
                    # Inner class - always include it
                    inner_class_code = self.generate_class(body_item, methods_to_keep, "")
                    lines.append(self.indent() + inner_class_code.replace('\n', '\n' + self.indent()))
                    lines.append("")
                elif isinstance(body_item, javalang.tree.FieldDeclaration):
                    field_code = self.generate_field(body_item)
                    if field_code:
                        lines.append(field_code)
                elif isinstance(body_item, javalang.tree.MethodDeclaration):
                    if body_item.name in methods_to_keep:
                        method_code = self.generate_method(body_item)
                        if method_code:
                            lines.append(method_code)
        else:
            # Fallback to old approach for compatibility
            # Fields
            for field in class_decl.fields:
                field_code = self.generate_field(field)
                if field_code:
                    lines.append(field_code)
            
            if class_decl.fields:
                lines.append("")
            
            # Methods (filtered)
            method_lines = []
            for method in class_decl.methods:
                if method.name in methods_to_keep:
                    method_code = self.generate_method(method)
                    if method_code:
                        method_lines.append(method_code)
            
            lines.extend(method_lines)
        
        self.indent_level -= 1
        lines.append("}")
        
        return "\n".join(lines)
    
    def generate_field(self, field: javalang.tree.FieldDeclaration) -> str:
        """Generate Java source code for a field declaration."""
        modifiers = " ".join(field.modifiers) if field.modifiers else ""
        type_name = self.get_type_name(field.type)
        
        field_lines = []
        for declarator in field.declarators:
            field_line = f"{self.indent()}{modifiers} {type_name} {declarator.name}".strip()
            if declarator.initializer:
                field_line += f" = {self.generate_expression(declarator.initializer)}"
            field_line += ";"
            field_lines.append(field_line)
        
        return "\n".join(field_lines)
    
    def generate_method(self, method: javalang.tree.MethodDeclaration) -> str:
        """Generate Java source code for a method declaration."""
        lines = []
        
        # Method signature
        modifiers = " ".join(method.modifiers) if method.modifiers else ""
        return_type = self.get_type_name(method.return_type) if method.return_type else "void"
        
        method_line = f"{self.indent()}{modifiers} {return_type} {method.name}(".strip()
        
        # Parameters
        if method.parameters:
            params = []
            for param in method.parameters:
                param_type = self.get_type_name(param.type)
                param_str = f"{param_type} {param.name}"
                params.append(param_str)
            method_line += ", ".join(params)
        
        method_line += ")"
        
        # Throws clause
        if method.throws:
            throws_list = [exc.name for exc in method.throws]
            method_line += f" throws {', '.join(throws_list)}"
        
        lines.append(method_line + " {")
        
        # Method body
        if method.body:
            self.indent_level += 1
            for stmt in method.body:
                stmt_code = self.generate_statement(stmt)
                if stmt_code:
                    lines.append(stmt_code)
            self.indent_level -= 1
        
        lines.append(f"{self.indent()}}}")
        lines.append("")  # Empty line after method
        
        return "\n".join(lines)
    
    def generate_statement(self, stmt) -> str:
        """Generate Java source code for a statement (simplified)."""
        # This is a simplified implementation - javalang doesn't provide
        # source code generation, so we'll use a basic approach
        if hasattr(stmt, 'children'):
            # For complex statements, we'll return a placeholder
            return f"{self.indent()}// TODO: Implement statement generation for {type(stmt).__name__}"
        else:
            return f"{self.indent()}// Statement: {type(stmt).__name__}"
    
    def generate_expression(self, expr) -> str:
        """Generate Java source code for an expression (simplified)."""
        if hasattr(expr, 'value'):
            return str(expr.value)
        elif hasattr(expr, 'member'):
            return expr.member
        else:
            return "/* expression */"
    
    def get_type_name(self, type_node) -> str:
        """Get the string representation of a type."""
        if hasattr(type_node, 'name'):
            return type_node.name
        elif hasattr(type_node, 'element_type'):
            # Array type
            return f"{self.get_type_name(type_node.element_type)}[]"
        else:
            return str(type_node)


class JavaSourceReconstructor:
    """Reconstructs Java source code by preserving original formatting where possible."""
    
    def __init__(self, original_source: str):
        self.original_lines = original_source.split('\n')
        self.line_mapping = {}
        
    def extract_method_source(self, method_node: javalang.tree.MethodDeclaration, empty_body: bool = False) -> str:
        """Extract the original source code for a method, optionally with empty body."""
        if hasattr(method_node, 'position') and method_node.position:
            start_line = method_node.position.line - 1  # Convert to 0-based
            
            # Find the end of the method by looking for the closing brace
            brace_count = 0
            end_line = start_line
            found_opening = False
            
            for i in range(start_line, len(self.original_lines)):
                line = self.original_lines[i]
                if '{' in line:
                    found_opening = True
                    brace_count += line.count('{')
                if found_opening:
                    brace_count -= line.count('}')
                    if brace_count == 0:
                        end_line = i
                        break
            
            if empty_body:
                # Generate empty method body while preserving signature
                return self.generate_empty_method(method_node, start_line)
            else:
                # Extract method lines as before
                method_lines = self.original_lines[start_line:end_line + 1]
                return '\n'.join(method_lines)
        
        return f"    // Could not extract source for method: {method_node.name}"
    
    def generate_empty_method(self, method_node: javalang.tree.MethodDeclaration, start_line: int) -> str:
        """Generate an empty method with preserved signature and throws clause."""
        # Extract the method signature line(s)
        signature_lines = []
        brace_found = False
        
        for i in range(start_line, len(self.original_lines)):
            line = self.original_lines[i]
            signature_lines.append(line)
            if '{' in line:
                brace_found = True
                break
        
        if not brace_found:
            # Fallback if we can't find the opening brace
            signature_lines.append('    {')
        
        # Get the last signature line and ensure it ends with {
        if signature_lines:
            last_line = signature_lines[-1]
            if not last_line.rstrip().endswith('{'):
                signature_lines[-1] = last_line.rstrip() + ' {'
        
        # Add appropriate method body based on return type
        if method_node.return_type and method_node.return_type != 'void':
            # Method has a non-void return type, add appropriate return statement
            return_value = self.get_default_return_value(method_node.return_type)
            signature_lines.append(f'        return {return_value};')
        
        # Add closing brace
        signature_lines.append('    }')
        
        return '\n'.join(signature_lines)
    
    def get_default_return_value(self, return_type) -> str:
        """Get appropriate default return value for a given return type."""
        if hasattr(return_type, 'name'):
            type_name = return_type.name
        else:
            type_name = str(return_type)
        
        # Handle primitive types
        if type_name in ['boolean']:
            return 'false'
        elif type_name in ['byte', 'short', 'int', 'long']:
            return '0'
        elif type_name in ['float', 'double']:
            return '0.0'
        elif type_name in ['char']:
            return "'\\0'"
        # Handle array types
        elif hasattr(return_type, 'element_type') or type_name.endswith('[]'):
            return 'null'
        # Handle all other object types (String, etc.)
        else:
            return 'null'
    
    def reconstruct_class(self, tree: javalang.tree.CompilationUnit, 
                         methods_to_keep: Set[str], 
                         methods_to_empty: Set[str] = None,
                         class_suffix: str = "") -> str:
        """Reconstruct Java source preserving original formatting."""
        if methods_to_empty is None:
            methods_to_empty = set()
        lines = []
        
        # Find key sections in original source
        package_line = None
        import_lines = []
        class_start_line = 0
        comment_lines = []
        
        in_header_comments = True
        for i, line in enumerate(self.original_lines):
            stripped = line.strip()
            
            # Collect header comments before package
            if in_header_comments and (stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('//') or stripped == ''):
                comment_lines.append(line)
                continue
            elif stripped.startswith('package '):
                in_header_comments = False
                package_line = line
            elif stripped.startswith('import '):
                import_lines.append(line)
            elif stripped.startswith('public class ') or stripped.startswith('class '):
                class_start_line = i
                break
        
        # Add header comments
        if comment_lines:
            lines.extend(comment_lines)
            lines.append("")
        
        # Add package and imports
        if package_line:
            lines.append(package_line)
            lines.append("")
        
        lines.extend(import_lines)
        if import_lines:
            lines.append("")
        
        # Class declaration - keep original name
        class_line = self.original_lines[class_start_line]
        
        # Ensure the class line has the opening brace
        if not class_line.rstrip().endswith('{'):
            class_line = class_line.rstrip() + ' {'
        
        lines.append(class_line)
        
        # Add body declarations (inner classes, fields, methods) in order
        for type_decl in tree.types:
            if isinstance(type_decl, javalang.tree.ClassDeclaration):
                self.add_class_body_declarations(type_decl, lines, methods_to_keep, methods_to_empty)
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def add_class_body_declarations(self, class_decl: javalang.tree.ClassDeclaration, 
                                  lines: list, methods_to_keep: Set[str], 
                                  methods_to_empty: Set[str]):
        """Add class body declarations (inner classes, fields, methods) in the correct order."""
        if hasattr(class_decl, 'body') and class_decl.body:
            # Process body declarations in order
            for body_item in class_decl.body:
                if isinstance(body_item, javalang.tree.ClassDeclaration):
                    # Inner class - extract its source
                    inner_class_source = self.extract_inner_class_source(body_item)
                    if inner_class_source:
                        lines.append(inner_class_source)
                        lines.append("")
                elif isinstance(body_item, javalang.tree.FieldDeclaration):
                    # Field
                    if hasattr(body_item, 'position') and body_item.position:
                        field_line = self.original_lines[body_item.position.line - 1]
                        lines.append(field_line)
                elif isinstance(body_item, javalang.tree.MethodDeclaration):
                    # Method
                    if body_item.name in methods_to_keep:
                        empty_body = body_item.name in methods_to_empty
                        method_source = self.extract_method_source(body_item, empty_body)
                        lines.append(method_source)
                        lines.append("")
        else:
            # Fallback to old approach for compatibility
            # Add fields
            for field in class_decl.fields:
                if hasattr(field, 'position') and field.position:
                    field_line = self.original_lines[field.position.line - 1]
                    lines.append(field_line)
            
            if class_decl.fields:
                lines.append("")
            
            # Add methods
            for method in class_decl.methods:
                if method.name in methods_to_keep:
                    empty_body = method.name in methods_to_empty
                    method_source = self.extract_method_source(method, empty_body)
                    lines.append(method_source)
                    lines.append("")
    
    def extract_inner_class_source(self, inner_class: javalang.tree.ClassDeclaration) -> str:
        """Extract the original source code for an inner class."""
        if hasattr(inner_class, 'position') and inner_class.position:
            start_line = inner_class.position.line - 1  # Convert to 0-based
            
            # Find the end of the inner class by looking for the closing brace
            brace_count = 0
            end_line = start_line
            found_opening = False
            
            for i in range(start_line, len(self.original_lines)):
                line = self.original_lines[i]
                if '{' in line:
                    found_opening = True
                    brace_count += line.count('{')
                if found_opening:
                    brace_count -= line.count('}')
                    if brace_count == 0:
                        end_line = i
                        break
            
            # Extract inner class lines
            inner_class_lines = self.original_lines[start_line:end_line + 1]
            return '\n'.join(inner_class_lines)
        
        return f"    // Could not extract source for inner class: {inner_class.name}"


class JulietTestcaseSplitter:
    """Main class for splitting Juliet testcases into good and bad variants."""
    
    def __init__(self, juliet_path: str, output_path: str):
        self.juliet_path = Path(juliet_path)
        self.output_path = Path(output_path)
        self.analyzer = JulietTestSuiteAnalyzer(juliet_path)
        self.cwe_groups = {}
        
        # Method name patterns
        self.bad_method_pattern = re.compile(r'^bad$')
        self.good_method_pattern = re.compile(r'^good$')
        self.secondary_good_pattern = re.compile(r'^good(\d+|G2B\d*|B2G\d*)$')
        self.helper_bad_pattern = re.compile(r'^helperBad$')
        self.helper_good_pattern = re.compile(r'^helperGood(G2B|B2G)?\d*$')
        self.source_sink_pattern = re.compile(r'^(bad|good[GB]2[GB]?\d*)(Source|Sink)$')
    
    def analyze_testcases(self):
        """Analyze the Juliet test suite structure."""
        print("Analyzing Juliet test suite structure...")
        self.cwe_groups = self.analyzer.analyze()
        
        total_testcases = sum(len(group.testcases) for group in self.cwe_groups.values())
        print(f"Found {len(self.cwe_groups)} CWEs with {total_testcases} testcases")
    
    def classify_methods(self, tree: javalang.tree.CompilationUnit) -> Tuple[Set[str], Set[str], Set[str]]:
        """Classify methods into bad, good, and shared categories."""
        bad_methods = set()
        good_methods = set()
        shared_methods = set()
        
        for type_decl in tree.types:
            if isinstance(type_decl, javalang.tree.ClassDeclaration):
                for method in type_decl.methods:
                    method_name = method.name
                    
                    if self.bad_method_pattern.match(method_name):
                        bad_methods.add(method_name)
                    elif self.good_method_pattern.match(method_name):
                        good_methods.add(method_name)
                    elif self.secondary_good_pattern.match(method_name):
                        good_methods.add(method_name)
                    elif self.helper_bad_pattern.match(method_name):
                        bad_methods.add(method_name)
                    elif self.helper_good_pattern.match(method_name):
                        good_methods.add(method_name)
                    elif self.source_sink_pattern.match(method_name):
                        # Determine if it's bad or good based on prefix
                        if method_name.startswith('bad'):
                            bad_methods.add(method_name)
                        else:
                            good_methods.add(method_name)
                    elif method_name == 'main':
                        shared_methods.add(method_name)
                    else:
                        # Other methods are typically shared
                        shared_methods.add(method_name)
        
        return bad_methods, good_methods, shared_methods
    
    def extract_testcase_variant(self, testcase: TestCase, variant: str) -> bool:
        """Extract a specific variant (good or bad) of a testcase."""
        try:
            # For now, only handle non-class-based and abstract method testcases
            if testcase.testcase_type == "class_based":
                return self.extract_class_based_testcase(testcase, variant)
            elif testcase.testcase_type == "abstract_method":
                return self.extract_abstract_method_testcase(testcase, variant)
            else:
                return self.extract_regular_testcase(testcase, variant)
                
        except Exception as e:
            print(f"Error extracting {variant} variant of {testcase.name}: {e}")
            return False
    
    def process_java_file_for_variant(self, file_path: str, variant: str, output_dir: Path, 
                                     primary_file: str = None) -> bool:
        """
        Process a single Java file for a specific variant (good/bad).
        
        Args:
            file_path: Path to the Java file relative to juliet_path
            variant: 'good' or 'bad'
            output_dir: Output directory for the processed file
            primary_file: Primary file path (used to determine if helper files should be skipped)
            
        Returns:
            True if processing was successful, False otherwise
        """
        full_path = self.juliet_path / file_path
        
        with open(full_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the Java file
        tree = javalang.parse.parse(source_code)
        
        # Classify methods for this file
        bad_methods, good_methods, shared_methods = self.classify_methods(tree)
        
        # Always keep all methods, but determine which ones to empty
        methods_to_keep = bad_methods | good_methods | shared_methods
        
        # Determine which methods to empty based on variant
        if variant == 'bad':
            methods_to_empty = good_methods  # Empty good methods in bad variant
        else:  # variant == 'good'
            methods_to_empty = bad_methods   # Empty bad methods in good variant
        
        # If this is a helper file (like 22b) and has no methods to keep, 
        # skip it for this variant
        if not methods_to_keep and primary_file and file_path != primary_file:
            return True  # Skip but don't fail
        
        # Generate the new Java source (no class suffix - keep original class names)
        class_suffix = ""
        reconstructor = JavaSourceReconstructor(source_code)
        new_source = reconstructor.reconstruct_class(tree, methods_to_keep, methods_to_empty, class_suffix)
        
        # Create the output file with original name
        file_name = Path(file_path).name
        new_file_name = file_name  # Keep original filename
        
        # Determine package structure and create file
        package_match = re.search(r'package\s+([\w.]+);', new_source)
        package_name = package_match.group(1) if package_match else "juliet.testcases"
        
        package_dirs = package_name.split('.')
        src_dir = output_dir / "src" / "main" / "java"
        package_dir = src_dir
        for dir_name in package_dirs:
            package_dir = package_dir / dir_name
        
        package_dir.mkdir(parents=True, exist_ok=True)
        
        output_file_path = package_dir / new_file_name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(new_source)
        
        return True

    def extract_regular_testcase(self, testcase: TestCase, variant: str) -> bool:
        """Extract regular (non-class-based) testcase variants."""
        if not testcase.files:
            return False
        
        # Find the primary file (usually the 'a' file)
        primary_file = None
        for file_path in testcase.files:
            if file_path.endswith('a.java') or len(testcase.files) == 1:
                primary_file = file_path
                break
        
        if not primary_file:
            primary_file = testcase.files[0]
        
        try:
            # Create output directory
            output_dir = self.output_path / f"{testcase.name}_{variant}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all files in the testcase
            for file_path in testcase.files:
                if not self.process_java_file_for_variant(file_path, variant, output_dir, primary_file):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error processing testcase {testcase.name}: {e}")
            return False
    
    def update_class_references(self, source_code: str, testcase: TestCase, class_suffix: str) -> str:
        """Update references to other classes in the same testcase."""
        updated_source = source_code
        
        for file_path in testcase.files:
            # Extract class name from file path
            file_name = Path(file_path).stem  # Remove .java extension
            
            # Update references to this class
            pattern = rf'\b{re.escape(file_name)}\b'
            replacement = f'{file_name}{class_suffix}'
            updated_source = re.sub(pattern, replacement, updated_source)
        
        return updated_source
    
    def extract_class_based_testcase(self, testcase: TestCase, variant: str) -> bool:
        """Extract class-based testcase variants."""
        # Class-based testcases already have separate _bad and _good1 files
        target_suffix = "_bad.java" if variant == "bad" else "_good1.java"
        
        target_file = None
        for file_path in testcase.files:
            if file_path.endswith(target_suffix):
                target_file = file_path
                break
        
        if not target_file:
            print(f"No {target_suffix} file found for class-based testcase {testcase.name}")
            return False
        
        full_path = self.juliet_path / target_file
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # For class-based testcases, we use the file as-is with original class names
            new_source = source_code
            
            # Create output directory and file
            output_dir = self.output_path / f"{testcase.name}_{variant}"
            self.create_java_project_structure(output_dir, testcase, new_source, "")
            
            return True
            
        except Exception as e:
            print(f"Error processing class-based testcase {target_file}: {e}")
            return False
    
    def extract_abstract_method_testcase(self, testcase: TestCase, variant: str) -> bool:
        """Extract abstract method testcase variants."""
        # Abstract method testcases need special handling
        # We need the base class and the appropriate implementation
        
        base_file = None
        main_file = None
        impl_files = []
        
        for file_path in testcase.files:
            filename = Path(file_path).name
            if filename.endswith('_base.java'):
                base_file = file_path
            elif filename.endswith('a.java'):  # Files ending with 'a.java' like *_81a.java
                main_file = file_path
            elif variant == 'bad' and filename.endswith('_bad.java'):
                impl_files.append(file_path)
            elif variant == 'good' and (filename.endswith('_goodG2B.java') or filename.endswith('_goodB2G.java')):
                impl_files.append(file_path)
        
        if not base_file or not main_file:
            print(f"Missing base or main file for abstract method testcase {testcase.name}: base={base_file}, main={main_file}")
            return False
        
        if len(impl_files) == 0:
            print(f"No implementation file found for {variant} variant of {testcase.name}")
            return False
        
        try:
            # Create output directory
            output_dir = self.output_path / f"{testcase.name}_{variant}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process the main file using the same logic as regular testcases
            if not self.process_java_file_for_variant(main_file, variant, output_dir):
                return False
            
            # For base and impl files, copy them as-is (they don't have good/bad methods to filter)
            src_dir = output_dir / "src" / "main" / "java"
            src_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in [base_file] + impl_files:
                full_path = self.juliet_path / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # Keep original class names and references
                new_source = source_code
                
                # Write the processed file with original name
                output_file_name = Path(file_path).name
                output_file_path = src_dir / output_file_name
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(new_source)
            
            return True
            
        except Exception as e:
            print(f"Error processing abstract method testcase: {e}")
            return False
    
    def create_java_project_structure(self, output_dir: Path, testcase: TestCase, 
                                    source_code: str, class_suffix: str):
        """Create a standalone Java project structure."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract package from source code
        package_match = re.search(r'package\s+([\w.]+);', source_code)
        package_name = package_match.group(1) if package_match else "juliet.testcases"
        
        # Create package directory structure
        package_dirs = package_name.split('.')
        src_dir = output_dir / "src" / "main" / "java"
        package_dir = src_dir
        for dir_name in package_dirs:
            package_dir = package_dir / dir_name
        
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the main Java file
        class_name = re.search(r'class\s+(\w+)', source_code)
        if class_name:
            java_file_name = f"{class_name.group(1)}.java"
        else:
            java_file_name = f"{testcase.name}.java"
        
        java_file_path = package_dir / java_file_name
        with open(java_file_path, 'w', encoding='utf-8') as f:
            f.write(source_code)
        
    def split_testcase(self, testcase: TestCase) -> Tuple[bool, bool]:
        """Split a testcase into good and bad variants."""
        print(f"Processing {testcase.name} ({testcase.testcase_type})...")
        
        bad_success = self.extract_testcase_variant(testcase, 'bad')
        good_success = self.extract_testcase_variant(testcase, 'good')
        
        return bad_success, good_success
    
    def split_all_testcases(self, limit: Optional[int] = None, randomize: bool = False):
        """Split all testcases into good and bad variants."""
        if not self.cwe_groups:
            self.analyze_testcases()
        
        total_processed = 0
        total_success = 0
        
        for cwe_id, group in self.cwe_groups.items():
            print(f"\nProcessing {cwe_id} ({len(group.testcases)} testcases)...")
            
            # Get testcases and optionally randomize order before applying limit
            testcases = list(group.testcases)
            if randomize:
                random.shuffle(testcases)
                print(f"Randomized order of {len(testcases)} testcases")
            
            for testcase in testcases:
                if limit and total_processed >= limit:
                    print(f"Reached limit of {limit} testcases")
                    return
                
                # Skip bad-only testcases for now
                if testcase.is_bad_only:
                    print(f"Skipping bad-only testcase: {testcase.name}")
                    continue
                
                bad_success, good_success = self.split_testcase(testcase)
                
                if bad_success or good_success:
                    total_success += 1
                
                total_processed += 1
                
                if total_processed % 10 == 0:
                    print(f"Processed {total_processed} testcases, {total_success} successful")
        
        print(f"\nCompleted processing {total_processed} testcases")
        print(f"Successfully extracted {total_success} testcases")
    
    def find_testcase_by_name(self, testcase_name: str) -> Optional[TestCase]:
        """Find a specific testcase by name across all CWE groups."""
        if not self.cwe_groups:
            self.analyze_testcases()
        
        for cwe_id, group in self.cwe_groups.items():
            for testcase in group.testcases:
                if testcase.name == testcase_name:
                    return testcase
        
        return None
    
    def split_single_testcase(self, testcase_name: str) -> bool:
        """Split a single testcase by name into good and bad variants."""
        testcase = self.find_testcase_by_name(testcase_name)
        
        if not testcase:
            print(f"❌ Testcase '{testcase_name}' not found")
            return False
        
        print(f"Found testcase: {testcase.name} ({testcase.testcase_type})")
        
        if testcase.is_bad_only:
            print(f"⚠️  Skipping bad-only testcase: {testcase.name}")
            return False
        
        bad_success, good_success = self.split_testcase(testcase)
        
        if bad_success and good_success:
            print(f"✅ Successfully extracted both variants of {testcase_name}")
            return True
        elif bad_success or good_success:
            variant = "bad" if bad_success else "good"
            print(f"⚠️  Partially successful - extracted {variant} variant of {testcase_name}")
            return True
        else:
            print(f"❌ Failed to extract {testcase_name}")
            return False


def create_single_codeql_database(testcase_dir: Path, db_root: Path, juliet_support_path: str, 
                                 codeql_path: str, progress_lock: threading.Lock) -> Tuple[str, bool, str]:
    """
    Create a CodeQL database for a single testcase project.
    
    Args:
        testcase_dir: Directory containing the testcase project
        db_root: Root directory where CodeQL databases will be created
        juliet_support_path: Path to the juliet-support directory
        codeql_path: Path to the CodeQL executable
        progress_lock: Lock for thread-safe progress updates
        
    Returns:
        Tuple of (project_name, success, error_message)
    """
    project_name = testcase_dir.name
    
    with progress_lock:
        print(f"🔄 Processing {project_name}...")
    
    # Add a small delay to prevent overwhelming the filesystem with concurrent operations
    time.sleep(0.1)
    
    try:
        # Find the main Java file in the project
        src_dir = testcase_dir / "src" / "main" / "java"
        if not src_dir.exists():
            return project_name, False, "No src/main/java directory found"
        
        # Find Java files in the project
        java_files = list(src_dir.rglob("*.java"))
        if not java_files:
            return project_name, False, "No Java files found"
        
        # Create a temporary source directory for CodeQL
        temp_src_dir = db_root / f"{project_name}_src"
        if temp_src_dir.exists():
            shutil.rmtree(temp_src_dir)
        temp_src_dir.mkdir()
        
        # Copy the project's Java files to temp directory
        for java_file in java_files:
            rel_path = java_file.relative_to(src_dir)
            dest_file = temp_src_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(java_file, dest_file)
        
        # Copy only essential juliet support files
        juliet_support_src = Path(juliet_support_path) / "src" / "main" / "java"
        if juliet_support_src.exists():
            # Copy to juliet/support to match the package structure
            support_dest = temp_src_dir / "juliet" / "support"
            support_dest.mkdir(parents=True, exist_ok=True)
            
            for support_file in juliet_support_src.rglob("*.java"):
                # Copy to juliet/support maintaining the original package structure
                rel_path = support_file.relative_to(juliet_support_src)
                dest_file = temp_src_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(support_file, dest_file)
        
        # Create CodeQL database
        db_dir = db_root / project_name
        if db_dir.exists():
            shutil.rmtree(db_dir)
        
        abs_temp_src_dir = temp_src_dir.resolve()
        abs_db_dir = db_dir.resolve()
        
        # Build the javac command for all Java files
        java_files_in_temp = list(temp_src_dir.rglob("*.java"))
        java_file_paths = [str(f.relative_to(temp_src_dir)) for f in java_files_in_temp]
        
        if java_file_paths:
            # Check if any files contain servlet imports to determine if we need servlet API
            needs_servlet_api = False
            for java_file in java_files_in_temp:
                try:
                    with open(java_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if ('import javax.servlet' in content or 
                            'extends AbstractTestCaseServlet' in content or
                            'HttpServletRequest' in content or
                            'HttpServletResponse' in content):
                            needs_servlet_api = True
                            break
                except Exception:
                    continue
            
            # Build classpath - include servlet API if needed
            classpath = "."
            if needs_servlet_api:
                # Look for servlet-api.jar in common locations
                servlet_jar_paths = [
                    Path.cwd() / "lib" / "servlet-api.jar",  # Current working directory
                    Path(__file__).parent.parent / "lib" / "servlet-api.jar",  # Project root
                    Path("/home/julius/qlcook/lib/servlet-api.jar")  # Absolute path from our setup
                ]
                
                servlet_jar = None
                for jar_path in servlet_jar_paths:
                    if jar_path.exists():
                        servlet_jar = str(jar_path)
                        break
                
                if servlet_jar:
                    classpath = f".:{servlet_jar}"
            
            javac_cmd = f"javac -cp {classpath} {' '.join(java_file_paths)}"
            
            cmd = [
                codeql_path, "database", "create", str(abs_db_dir),
                "--language=java",
                "--source-root", str(abs_temp_src_dir),
                "--command", javac_cmd
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=abs_temp_src_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                # Clean up temp source directory on success
                shutil.rmtree(temp_src_dir)
                return project_name, True, ""
            else:
                error_msg = f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
                print(f"Error creating CodeQL database for {project_name}: {error_msg}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Working directory: {abs_temp_src_dir}")
                print(f"Java files: {java_file_paths}")
                print(f"Classpath: {classpath}")
                print(f"Javac command: {javac_cmd}")
                sys.exit(1)
        else:
            if temp_src_dir.exists():
                shutil.rmtree(temp_src_dir)
            return project_name, False, "No Java files found to compile"
            
    except Exception as e:
        # Clean up on error
        temp_src_dir = db_root / f"{project_name}_src"
        db_dir = db_root / project_name
        if temp_src_dir.exists():
            shutil.rmtree(temp_src_dir)
        if db_dir.exists():
            shutil.rmtree(db_dir)
        return project_name, False, str(e)


def create_codeql_databases(output_root: str, juliet_support_path: str, db_root: str = None, 
                          codeql_path: str = "codeql", max_workers: int = None):
    """
    For each generated project in output_root, create a CodeQL database in parallel.
    
    Args:
        output_root: Root directory containing the generated testcase projects
        juliet_support_path: Path to the juliet-support directory 
        db_root: Root directory where CodeQL databases will be created (defaults to output_root/codeql_dbs)
        codeql_path: Path to the CodeQL executable (defaults to "codeql")
        max_workers: Maximum number of parallel workers (defaults to min(4, cpu_count))
    """
    output_path = Path(output_root)
    if not output_path.exists():
        print(f"❌ Output directory does not exist: {output_root}")
        return
    
    if db_root is None:
        db_root = output_path / "codeql_dbs"
    else:
        db_root = Path(db_root)
    
    db_root.mkdir(exist_ok=True)
    
    # Find all generated testcase projects
    testcase_dirs = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'codeql_dbs']
    
    # Auto-determine optimal worker count if not specified
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(4, cpu_count)  # Conservative default to avoid overwhelming the system
    
    # Limit workers based on number of projects to avoid unnecessary overhead
    max_workers = min(max_workers, len(testcase_dirs))
    
    print(f"Found {len(testcase_dirs)} testcase projects to process for CodeQL")
    print(f"Using {max_workers} parallel workers for database creation (CPU cores: {multiprocessing.cpu_count()})")
    
    if len(testcase_dirs) == 0:
        print("No testcase directories found to process.")
        return
    
    success_count = 0
    failed_count = 0
    progress_lock = threading.Lock()
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_project = {
            executor.submit(create_single_codeql_database, testcase_dir, db_root, 
                          juliet_support_path, codeql_path, progress_lock): testcase_dir.name
            for testcase_dir in testcase_dirs
        }
        
        # Process completed tasks with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(future_to_project):
            project_name = future_to_project[future]
            try:
                project_name_result, success, error_message = future.result()
                completed += 1
                
                with progress_lock:
                    if success:
                        print(f"✅ [{completed}/{len(testcase_dirs)}] CodeQL DB for {project_name_result}")
                        success_count += 1
                    else:
                        print(f"❌ [{completed}/{len(testcase_dirs)}] CodeQL DB failed for {project_name_result}")
                        if error_message and len(error_message) < 200:  # Only show short error messages
                            print(f"   Error: {error_message}")
                        failed_count += 1
                        
            except Exception as e:
                completed += 1
                with progress_lock:
                    print(f"❌ [{completed}/{len(testcase_dirs)}] Exception processing {project_name}: {e}")
                    failed_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"\n🎯 CodeQL database creation completed in {elapsed_time:.1f}s:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    if success_count > 0:
        print(f"   ⏱️  Average time per database: {elapsed_time/len(testcase_dirs):.1f}s")
    print(f"   📁 Databases location: {db_root}")


def create_codeql_databases_sequential(output_root: str, juliet_support_path: str, db_root: str = None, codeql_path: str = "codeql"):
    """
    Original sequential version of CodeQL database creation (kept for compatibility).
    
    Args:
        output_root: Root directory containing the generated testcase projects
        juliet_support_path: Path to the juliet-support directory 
        db_root: Root directory where CodeQL databases will be created (defaults to output_root/codeql_dbs)
        codeql_path: Path to the CodeQL executable (defaults to "codeql")
    """
    output_path = Path(output_root)
    if not output_path.exists():
        print(f"❌ Output directory does not exist: {output_root}")
        return
    
    if db_root is None:
        db_root = output_path / "codeql_dbs"
    else:
        db_root = Path(db_root)
    
    db_root.mkdir(exist_ok=True)
    
    # Find all generated testcase projects
    testcase_dirs = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'codeql_dbs']
    
    print(f"Found {len(testcase_dirs)} testcase projects to process for CodeQL (sequential mode)")
    
    success_count = 0
    failed_count = 0
    progress_lock = threading.Lock()  # Not needed for sequential but keeps interface consistent
    
    for testcase_dir in testcase_dirs:
        project_name_result, success, error_message = create_single_codeql_database(
            testcase_dir, db_root, juliet_support_path, codeql_path, progress_lock
        )
        
        if success:
            print(f"✅ CodeQL DB for {project_name_result}")
            success_count += 1
        else:
            print(f"❌ CodeQL DB failed for {project_name_result}: {error_message}")
            failed_count += 1
    
    print(f"\n🎯 CodeQL database creation completed:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Databases location: {db_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Split Juliet Test Suite testcases into separate good and bad variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with parallel CodeQL database creation
  python split_juliet_testcases.py ../juliet-java-test-suite ./dataset --limit 50 --randomize --cwe CWE89 --create-codeql-dbs --codeql-path /opt/codeql/codeql
  
  # Single testcase extraction
  python split_juliet_testcases.py ../juliet-java-test-suite ./dataset --testcase CWE89_SQL_Injection__Property_executeBatch_61b
  
  # Parallel database creation with custom worker count
  python split_juliet_testcases.py ../juliet-java-test-suite ./dataset --create-codeql-dbs --max-workers 8
  
  # Sequential database creation for debugging
  python split_juliet_testcases.py ../juliet-java-test-suite ./dataset --create-codeql-dbs --sequential-db-creation
        """
    )
    
    parser.add_argument(
        'juliet_path',
        help='Path to the Juliet Test Suite directory'
    )
    
    parser.add_argument(
        'output_path',
        help='Output directory for extracted testcases'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit the number of testcases to process (for testing)'
    )
    
    parser.add_argument(
        '--cwe',
        help='Process only testcases from a specific CWE (e.g., CWE15)'
    )
    
    parser.add_argument(
        '--testcase',
        help='Process only a specific testcase by name (e.g., CWE89_SQL_Injection__Property_executeBatch_61b)'
    )
    
    parser.add_argument(
        '--randomize',
        action='store_true',
        help='Randomize the order of testcases before applying the limit'
    )
    
    parser.add_argument(
        '--create-codeql-dbs',
        action='store_true',
        help='Create CodeQL databases for each generated testcase project'
    )
    
    parser.add_argument(
        '--codeql-db-root',
        help='Root directory for CodeQL databases (defaults to OUTPUT_PATH/codeql_dbs)'
    )
    
    parser.add_argument(
        '--juliet-support-path',
        default='../juliet-java-test-suite/juliet-support',
        help='Path to the juliet-support directory (default: ../juliet-java-test-suite/juliet-support)'
    )
    
    parser.add_argument(
        '--codeql-path',
        default='codeql',
        help='Path to the CodeQL executable (default: codeql)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of parallel workers for CodeQL database creation (default: auto-detect based on CPU cores, max 4)'
    )
    
    parser.add_argument(
        '--sequential-db-creation',
        action='store_true',
        help='Use sequential database creation instead of parallel (useful for debugging)'
    )
    
    args = parser.parse_args()
    
    try:
        splitter = JulietTestcaseSplitter(args.juliet_path, args.output_path)
        
        if args.testcase:
            # Process only specific testcase
            print(f"Processing single testcase: {args.testcase}")
            success = splitter.split_single_testcase(args.testcase)
            if not success:
                return 1
        elif args.cwe:
            # Process only specific CWE
            splitter.analyze_testcases()
            if args.cwe in splitter.cwe_groups:
                group = splitter.cwe_groups[args.cwe]
                print(f"Processing {len(group.testcases)} testcases from {args.cwe}")
                
                # Get testcases and optionally randomize order before applying limit
                testcases = list(group.testcases)
                if args.randomize:
                    random.shuffle(testcases)
                    print(f"Randomized order of {len(testcases)} testcases")
                
                for testcase in testcases[:args.limit] if args.limit else testcases:
                    if not testcase.is_bad_only:
                        splitter.split_testcase(testcase)
            else:
                print(f"CWE {args.cwe} not found. Available CWEs: {list(splitter.cwe_groups.keys())}")
        else:
            # Process all testcases
            splitter.split_all_testcases(args.limit, args.randomize)
        
        print("Testcase splitting completed!")
        
        # Create CodeQL databases if requested
        if args.create_codeql_dbs:
            print("\n" + "="*50)
            print("Creating CodeQL databases...")
            print("="*50)
            
            # Resolve juliet-support path
            juliet_support_path = Path(args.juliet_support_path)
            if not juliet_support_path.is_absolute():
                # Make it relative to the script location
                script_dir = Path(__file__).parent
                juliet_support_path = script_dir / juliet_support_path
            
            juliet_support_path = juliet_support_path.resolve()
            
            if not juliet_support_path.exists():
                print(f"❌ Juliet support path does not exist: {juliet_support_path}")
                print("   Please specify the correct path with --juliet-support-path")
                return 1
            
            if args.sequential_db_creation:
                print("Using sequential database creation mode")
                create_codeql_databases_sequential(
                    output_root=args.output_path,
                    juliet_support_path=str(juliet_support_path),
                    db_root=args.codeql_db_root,
                    codeql_path=args.codeql_path
                )
            else:
                create_codeql_databases(
                    output_root=args.output_path,
                    juliet_support_path=str(juliet_support_path),
                    db_root=args.codeql_db_root,
                    codeql_path=args.codeql_path,
                    max_workers=args.max_workers
                )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
