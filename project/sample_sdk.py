"""Prepare the supplied notebook's cg.api helpers with the pinned native runtime."""

import hashlib
import importlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from .collect_matches import ENGINE_ROOT, download_runtime, load_runtime
from .agent.native_search import NativeSearch


HELPER_URL = "https://api.kaggle.com/v1/datasets/download/kiyotah/cg-lib/cg%2F{}?datasetVersionNumber=9"
HELPER_HASHES = {
    "api.py": "ff163d90fc7f0feb61beb037f8931b887f98e78a7e891bdfc46b2f9c8abbfbff",
    "utils.py": "60f29665cee0a88525d6f0383bc45959a6262d16fe35ef380aece1e0ea13c49b",
}


def prepare_sample_sdk():
    if not (ENGINE_ROOT / "cg/game.py").is_file():
        download_runtime(ENGINE_ROOT)
    game, sim = load_runtime(ENGINE_ROOT)
    # Register the verified search ABI before the original cg.api calls it.
    bindings = NativeSearch(sim.Battle.lib)
    helpers = ENGINE_ROOT.parent / "cg-helpers-v9" / "cg"
    helpers.mkdir(parents=True, exist_ok=True)
    for name, digest in HELPER_HASHES.items():
        target = helpers / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        with urlopen(HELPER_URL.format(name), timeout=45) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Unexpected cg-lib v9 helper hash: {name}")
        with NamedTemporaryFile(dir=helpers, delete=False) as temporary:
            temporary.write(content)
        Path(temporary.name).replace(target)
    cg = importlib.import_module("cg")
    if str(helpers) not in cg.__path__:
        cg.__path__.append(str(helpers))
    api = importlib.import_module("cg.api")
    if Path(api.__file__).resolve() != (helpers / "api.py").resolve():
        raise RuntimeError("A different cg.api is loaded; restart the notebook kernel")
    return game, sim, api, bindings
