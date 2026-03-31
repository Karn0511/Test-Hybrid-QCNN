import os
import re
import json
from pathlib import Path

def get_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Simple regex for backend imports
    pattern = r"from (backend\.[a-zA-Z0-9_\.]+)|import (backend\.[a-zA-Z0-9_\.]+)"
    matches = re.findall(pattern, content)
    
    imports = []
    for m in matches:
        imp = m[0] if m[0] else m[1]
        if imp not in imports:
            imports.append(imp)
    return imports

def build_dependency_graph(root_dir):
    graph = {}
    root = Path(root_dir)
    
    for py_file in root.rglob("*.py"):
        if ".venv" in str(py_file) or ".git" in str(py_file) or "__pycache__" in str(py_file):
            continue
            
        rel_path = py_file.relative_to(root)
        module_name = str(rel_path).replace(os.path.sep, ".").replace(".py", "")
        
        # Clean up path to actual module
        graph[module_name] = {
            "path": str(rel_path),
            "depends_on": get_imports(py_file)
        }
    return graph

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    graph = build_dependency_graph(root)
    
    output_file = root / "dependency_graph.json"
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=4)
    print(f"✅ Dependency graph generated at {output_file}")
