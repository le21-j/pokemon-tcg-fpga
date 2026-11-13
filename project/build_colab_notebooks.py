"""Synchronize the small public-repository setup cell in all four notebooks."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("arena.ipynb", "sample_agent.ipynb", "reinforcement_learning.ipynb", "deep_sets_walkthrough.ipynb")
SETUP_END = "# --- End automatic project setup ---"


def main():
    bootstrap = (ROOT / "project/colab_setup.py").read_text()
    header = '''# @title 1. Automatic setup {display-mode: "form"}
# @markdown Run All clones the public project and installs dependencies automatically in Colab. Optional overrides can stay at their defaults.
REPOSITORY_PATH = "" # @param {type:"string"}
MOUNT_GOOGLE_DRIVE = False # @param {type:"boolean"}
INSTALL_LOCALLY = False # @param {type:"boolean"}

'''
    call = '''
if MOUNT_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

ROOT = setup_project(REPOSITORY_PATH, install_locally=INSTALL_LOCALLY)
'''
    for name in NOTEBOOKS:
        path = ROOT / "project" / name
        notebook = nbformat.read(path, as_version=4)
        cell = notebook.cells[1]
        tail = cell.source.split(SETUP_END, 1)[1].strip()
        cell.source = header + bootstrap + "\n" + call + "\n" + SETUP_END + "\n\n" + tail
        cell.metadata.update({"cellView": "form", "id": cell.id})
        notebook.metadata.pop("colab_project_snapshot", None)
        intro = notebook.cells[0].source.split("\n\n**Colab: open this notebook and choose", 1)[0]
        intro = intro.split("\n\n**Colab: choose Runtime", 1)[0]
        intro += "\n\n**Colab: choose Runtime → Run all.** Setup clones the public [project repository](https://github.com/le21-j/pokemon-tcg-fpga) and installs dependencies automatically."
        notebook.cells[0].source = intro
        for other in notebook.cells:
            if other.cell_type == "code":
                other.outputs = []
                other.execution_count = None
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Updated {name}: public Git clone, no embedded project snapshot")


if __name__ == "__main__":
    main()
