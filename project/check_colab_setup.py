"""Check direct Git cloning, automatic installation, and shared setup cells."""

import ast
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

import nbformat

from . import colab_setup
from .build_colab_notebooks import ROOT, NOTEBOOKS


def main():
    for name in NOTEBOOKS:
        notebook = nbformat.read(ROOT / "project" / name, as_version=4)
        nbformat.validate(notebook)
        setup = notebook.cells[1]
        ast.parse(setup.source)
        assert colab_setup.PUBLIC_REPOSITORY in setup.source
        assert "_PROJECT_SNAPSHOT" not in setup.source
        assert "colab_project_snapshot" not in notebook.metadata
        assert setup.metadata["cellView"] == "form"
        assert (ROOT / "project/colab_setup.py").read_text() in setup.source

    with TemporaryDirectory(prefix="tcg-public-clone-") as temporary:
        parent = Path(temporary)
        fixture = parent / "git-fixture"
        required = ("project/arena.py", "project/sample_sdk.py", "project/agent/model.py",
                    "project/rl_baseline/train.py", "project/agent_template/agent.py",
                    "project/requirements-colab.txt")
        for name in required:
            target = fixture / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# Git fixture\n")
        for args in (("init", "-b", "main"), ("add", "."),
                     ("-c", "user.name=Colab test", "-c", "user.email=test@example.invalid", "commit", "-m", "Fixture")):
            subprocess.run(["git", "-C", str(fixture), *args], capture_output=True, check=True)
        with patch.object(colab_setup.subprocess, "check_call") as pip:
            root = colab_setup.setup_project(repository_url=str(fixture), in_colab=True,
                                            workspace_parent=parent / "fresh")
            assert (root / ".git").is_dir() and colab_setup._project_ready(root)
            assert pip.call_count == 1 and "requirements-colab.txt" in pip.call_args.args[0][-1]
            again = colab_setup.setup_project(repository_url=str(fixture), in_colab=True,
                                             workspace_parent=parent / "fresh")
            assert root == again
            (root / "project/arena.py").unlink()
            (root / "keep-my-work.txt").write_text("Preserve an incomplete checkout")
            recovered = colab_setup.setup_project(repository_url=str(fixture), in_colab=True,
                                                 workspace_parent=parent / "fresh")
            assert recovered != root and (root / "keep-my-work.txt").exists()
            pip.reset_mock()
            colab_setup.setup_project(str(recovered), in_colab=False)
            pip.assert_not_called()
            with patch.object(colab_setup.subprocess, "run", side_effect=subprocess.CalledProcessError(128, "git")):
                try:
                    colab_setup.setup_project(repository_url="invalid", in_colab=True,
                                              workspace_parent=parent / "failure")
                except subprocess.CalledProcessError:
                    pass
                else:
                    raise AssertionError("An unavailable repository must report a clone failure")
    print("PASS: four snapshot-free notebooks, direct Git clone, pip command, caching, recovery and clone errors")


if __name__ == "__main__":
    main()
