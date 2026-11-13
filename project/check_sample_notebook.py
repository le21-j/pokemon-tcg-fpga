"""Check that the RL notebook still contains the original sample definitions."""

import ast
import hashlib
from pathlib import Path

import nbformat


def main():
    path = Path(__file__).with_name("reinforcement_learning.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    tree = ast.parse("\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code"))
    definitions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                   for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    original = notebook.metadata["sample_source"]["unchanged_definitions"]
    for name, digest in original.items():
        assert definitions.get(name) == digest, f"Original sample definition changed: {name}"
    assert notebook.metadata["sample_source"]["sha256"] == "1c2827a2f3afc73f06cb49a5d4279adc8e8d29335fd1940da7049122a8ee7ad4"
    print(f"PASS: {len(original)} original sample functions/classes, notebook schema, and executable Python syntax")


if __name__ == "__main__":
    main()
