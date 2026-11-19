from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


def make_repo(root: Path) -> Path:
    """A small git repo with sources in each directory the code state covers."""
    repo = root / "repo"
    for relative, text in {
        "src/pkg/a.py": "A = 1\n",
        "scripts/run.sh": "#!/bin/bash\necho run\n",
        "configs/c.yaml": "key: 1\n",
        "data/d.txt": "data\n",
        ".gitignore": "ignored/\n",
    }.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (repo / "scripts" / "run.sh").chmod(0o755)
    (repo / "src" / "pkg" / "link.py").symlink_to("a.py")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo
