"""Standard-library Colab setup for the public Pokémon TCG project repository."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


PUBLIC_REPOSITORY = "https://github.com/le21-j/pokemon-tcg-fpga.git"


def running_in_colab():
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        import google.colab
        return True
    except ImportError:
        return False


def _project_ready(root):
    return all((root / name).is_file() for name in (
        "project/arena.py", "project/sample_sdk.py", "project/agent/model.py",
        "project/rl_baseline/train.py", "project/agent_template/agent.py",
        "project/requirements-colab.txt"))


def setup_project(repository_path="", *, repository_url=PUBLIC_REPOSITORY, repository_ref="main",
                  in_colab=None, workspace_parent=None, install_locally=False):
    in_colab = running_in_colab() if in_colab is None else in_colab
    if repository_path.strip():
        root = Path(repository_path).expanduser().resolve()
        if not _project_ready(root):
            raise ValueError("REPOSITORY_PATH must contain the Pokémon TCG project")
    elif not in_colab:
        root = next((p for p in [Path.cwd(), *Path.cwd().parents] if _project_ready(p)), None)
        if root is None:
            raise ValueError("Run inside the project checkout or set REPOSITORY_PATH")
    else:
        parent = Path(workspace_parent) if workspace_parent is not None else (
            Path("/content") if Path("/content").is_dir() else Path.cwd())
        parent.mkdir(parents=True, exist_ok=True)
        name = "pokemon-tcg-fpga"
        root = parent / name
        retry = 0
        while root.exists() and not _project_ready(root):
            retry += 1
            root = parent / f"{name}-{retry}"
        if not root.exists():
            with tempfile.TemporaryDirectory(prefix="tcg-clone-", dir=parent) as temporary:
                checkout = Path(temporary) / "checkout"
                subprocess.run(["git", "clone", "--depth", "1", "--branch", repository_ref,
                                repository_url, str(checkout)],
                               env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), check=True)
                if not _project_ready(checkout):
                    raise RuntimeError("The downloaded repository is missing project files")
                checkout.rename(root)
            print("Cloned the public Pokémon TCG project")
    if in_colab or install_locally:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                               "-r", str(root / "project/requirements-colab.txt")])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print("Project ready:", root)
    return root
