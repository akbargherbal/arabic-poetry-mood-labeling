#!/usr/bin/env python
"""
checkpoint_utils.py

Chunked, resumable checkpointing for the (expensive, GPU-bound) embedding
step of label_moods.py, plus best-effort incremental `git commit` + `git
push` so that progress survives a lost Colab connection, a runtime
recycle, or a plain crash.

Why chunk only the embedding step?
-----------------------------------
Encoding is the only part of the pipeline that is slow enough to be at
real risk from a multi-hour Colab session (spot preemption, idle
timeout, disconnect). Scoring/calibration/tagging runs in seconds even
on ~24k rows and is fully deterministic given the embeddings, so it does
not need its own checkpoint -- if the process dies during that stage,
simply re-run the script; every embedding chunk is already on disk (and
pushed to GitHub), so it picks straight back up.

On-disk layout (relative to --checkpoint-dir):
    embeddings/chunk_000000.npy   float32 array, shape (rows_in_chunk, dim)
    embeddings/chunk_000001.npy
    ...
    meta.json                     run metadata, used to validate resumes

Design goals:
  - Every chunk write is atomic (tmp file + os.replace) so a crash
    mid-write never leaves a corrupt chunk that looks "done".
  - Resuming never re-embeds a chunk that is already present and valid.
  - Git push failures are logged and retried, but never crash the run --
    the embeddings are already safe on local disk either way, and the
    next successful push will pick them up.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


class CheckpointError(Exception):
    pass


def chunk_bounds(n_rows, chunk_size):
    """Yield (chunk_idx, start, end) tuples covering n_rows, in order."""
    idx = 0
    start = 0
    while start < n_rows:
        end = min(start + chunk_size, n_rows)
        yield idx, start, end
        start = end
        idx += 1


class EmbeddingCheckpoint:
    """Manages the on-disk chunk files + meta.json for one labeling run."""

    def __init__(self, checkpoint_dir, n_rows, chunk_size, model_name, embedding_dim=None):
        self.dir = Path(checkpoint_dir)
        self.emb_dir = self.dir / "embeddings"
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self.n_rows = n_rows
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.meta_path = self.dir / "meta.json"
        self._load_or_init_meta()

    def _load_or_init_meta(self):
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            mismatches = []
            if meta.get("n_rows") != self.n_rows:
                mismatches.append(f"n_rows: checkpoint={meta.get('n_rows')} current={self.n_rows}")
            if meta.get("chunk_size") != self.chunk_size:
                mismatches.append(f"chunk_size: checkpoint={meta.get('chunk_size')} current={self.chunk_size}")
            if meta.get("model_name") != self.model_name:
                mismatches.append(f"model_name: checkpoint={meta.get('model_name')} current={self.model_name}")
            if mismatches:
                raise CheckpointError(
                    f"Existing checkpoint at '{self.dir}' was created with different "
                    "settings, so resuming from it would silently misalign rows:\n  "
                    + "\n  ".join(mismatches)
                    + "\n\nEither point --checkpoint-dir at a fresh directory, or "
                    "re-run with the original settings to resume safely."
                )
            self.embedding_dim = meta.get("embedding_dim", self.embedding_dim)
        else:
            self._write_meta()

    def _write_meta(self):
        meta = {
            "n_rows": self.n_rows,
            "chunk_size": self.chunk_size,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
        }
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, self.meta_path)

    def chunk_path(self, chunk_idx):
        return self.emb_dir / f"chunk_{chunk_idx:06d}.npy"

    def is_done(self, chunk_idx, expected_rows):
        """True if a valid, complete chunk file already exists on disk."""
        p = self.chunk_path(chunk_idx)
        if not p.exists():
            return False
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:
            return False  # corrupt/partial file -> treat as not done, redo it
        if arr.shape[0] != expected_rows:
            return False
        if self.embedding_dim and arr.shape[1] != self.embedding_dim:
            return False
        return True

    def save_chunk(self, chunk_idx, embeddings):
        if self.embedding_dim is None:
            self.embedding_dim = int(embeddings.shape[1])
            self._write_meta()
        p = self.chunk_path(chunk_idx)
        tmp = p.with_name(p.name + ".tmp")
        # Write via an explicit file handle -- np.save() silently appends
        # ".npy" to bare filenames that don't already end in it, which would
        # turn "chunk_000000.npy.tmp" into "chunk_000000.npy.tmp.npy" and
        # break the atomic rename below.
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(embeddings, dtype=np.float32))
        os.replace(tmp, p)  # atomic on POSIX and on Windows (py3.3+)

    def load_chunk(self, chunk_idx):
        return np.load(self.chunk_path(chunk_idx))

    def load_all(self, total_chunks):
        return np.concatenate([self.load_chunk(i) for i in range(total_chunks)], axis=0)

    def progress(self, total_chunks):
        done = sum(1 for i in range(total_chunks) if self.chunk_path(i).exists())
        return done, total_chunks


# ---------------------------------------------------------------------------
# Best-effort git commit + push
# ---------------------------------------------------------------------------
def _run_git(args, cwd, timeout=60):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def is_git_repo(repo_dir):
    try:
        r = _run_git(["rev-parse", "--is-inside-work-tree"], repo_dir)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


def git_checkpoint_push(repo_dir, paths, message, retries=3, backoff=5):
    """
    Best-effort: `git add` the given paths, commit, and push.

    Never raises. Logs and returns False on failure so the caller can keep
    going -- the checkpoint data is already safe on local disk regardless
    of whether the push succeeds, and the next push attempt will include
    anything that failed to go out this time.
    """
    if not is_git_repo(repo_dir):
        print("  [git] not inside a git repo -> skipping push (checkpoint is still saved locally)")
        return False

    add = _run_git(["add", *paths], repo_dir)
    if add.returncode != 0:
        print(f"  [git] add failed: {add.stderr.strip()}")
        return False

    status = _run_git(["status", "--porcelain"], repo_dir)
    if not status.stdout.strip():
        return True  # nothing changed since last checkpoint push

    commit = _run_git(["commit", "-m", message], repo_dir)
    if commit.returncode != 0:
        print(f"  [git] commit failed: {commit.stderr.strip()}")
        return False

    for attempt in range(1, retries + 1):
        push = _run_git(["push"], repo_dir, timeout=120)
        if push.returncode == 0:
            print(f"  [git] pushed -> {message}")
            return True
        print(f"  [git] push attempt {attempt}/{retries} failed: {push.stderr.strip()}")
        if attempt < retries:
            time.sleep(backoff * attempt)

    print(
        "  [git] giving up on push for now (network hiccup?). The commit is "
        "safe in the local repo -- the next checkpoint push will retry it."
    )
    return False
