"""Which files the olfsysm extension is compiled from, their hash and their git state.

Shared by setup.py, which bakes these into the extension at build time
(`olfsysm.build_info()`), and by `pyolfsysm.mb_model.olfsysm_clean_problems`, which compares
the build with the checkout's current C++ sources. Python-only changes to this repository
(pyolfsysm, scripts, tests) do not change any of these, so they need no rebuild.

Standard library only, and importable without importing pyolfsysm (setup.py loads this file
by path, before pyolfsysm is installed).
"""
import glob
import hashlib
from pathlib import Path
import subprocess
from typing import List, Optional, Union

# compiled sources (besides the generated bindings/python/olfsysm_build_info.cpp), relative
# to the repository root
SOURCES: List[str] = ['libolfsysm/src/olfsysm.cpp', 'bindings/python/pyolfsysm.cpp']
# directories whose files are all headers of the build
HEADER_DIRS: List[str] = ['libolfsysm/api', 'libolfsysm/include']


def headers(root: Union[str, Path]) -> List[str]:
    """Returns the header files of the build, relative to `root`, sorted."""
    root = Path(root)
    return sorted(str(Path(p).relative_to(root)) for d in HEADER_DIRS
        for p in glob.glob(str(root / d / '**' / '*'), recursive=True) if Path(p).is_file()
    )


def source_sha256(root: Union[str, Path]) -> Optional[str]:
    """SHA-256 over the compiled sources and headers under `root` (paths and contents).

    None if a compiled source is missing (e.g. `root` is not a source checkout).
    """
    root = Path(root)
    if not all((root / s).is_file() for s in SOURCES):
        return None
    h = hashlib.sha256()
    for rel in SOURCES + headers(root):
        h.update(rel.encode()); h.update(b'\0')
        h.update((root / rel).read_bytes()); h.update(b'\0')
    return h.hexdigest()


def _git(root: Union[str, Path], *args: str) -> Optional[str]:
    try:
        return subprocess.run(['git', *args], cwd=str(root), capture_output=True, text=True,
            check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def cpp_git_status(root: Union[str, Path]) -> Optional[str]:
    """`git status --porcelain` of the compiled sources and header directories under `root`.

    Includes untracked files there (an untracked header is part of the build). '' if they
    match HEAD, None if `root` is not a git checkout.
    """
    if _git(root, 'rev-parse', 'HEAD') is None:
        return None
    return _git(root, 'status', '--porcelain', '--untracked-files=all', '--',
        *SOURCES, *HEADER_DIRS
    )


def cpp_dirty(root: Union[str, Path]) -> int:
    """1 if the compiled sources / headers under `root` differ from HEAD, 0 if not, -1 if
    `root` is not a git checkout."""
    status = cpp_git_status(root)
    if status is None:
        return -1
    return int(bool(status.strip()))


def cpp_git_diff(root: Union[str, Path]) -> str:
    """`git diff HEAD` of the compiled sources and header directories under `root`."""
    return _git(root, 'diff', 'HEAD', '--', *SOURCES, *HEADER_DIRS) or ''
