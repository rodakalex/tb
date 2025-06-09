import ast
import os

class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            name = ast.unparse(node.func)
        self.calls.add(name)
        self.generic_visit(node)

def analyze_function(fn_node):
    visitor = FunctionCallVisitor()
    visitor.visit(fn_node)
    return visitor.calls

def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except Exception as e:
            return [f"{filepath}: failed to parse — {e}"]

    output = [f"\nFile: {filepath}"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            calls = analyze_function(node)
            output.append(f"  def {node.name}()")
            if calls:
                for call in sorted(calls):
                    output.append(f"    calls → {call}")
        elif isinstance(node, ast.ClassDef):
            output.append(f"  class {node.name}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    calls = analyze_function(item)
                    output.append(f"    def {item.name}()")
                    if calls:
                        for call in sorted(calls):
                            output.append(f"      calls → {call}")
    return output

def walk_project(directories):
    all_output = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    all_output.extend(analyze_file(path))
    return all_output

if __name__ == "__main__":
    dirs = ["trading_analysis", "strategy"]
    result = walk_project(dirs)
    with open("code_map.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(result))
    print("✅ Готово: смотри файл code_map.txt")
