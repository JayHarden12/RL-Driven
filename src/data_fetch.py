from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional

import requests

from .utils import DEFAULT_DATASET_DIRNAME


DatasetPath = Path(".") / DEFAULT_DATASET_DIRNAME


def _sha256(path: Path, chunk_size: int = 2 ** 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path, progress: Optional[Callable[[int, Optional[int]], None]] = None, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        done = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=2 ** 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        tmp.rename(dest)
    return dest


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            # Prevent zip slip
            member_path = Path(out_dir, member.filename).resolve()
            if not str(member_path).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        z.extractall(out_dir)


def _unpack_archive(archive: Path, target_dir: Path) -> None:
    # Try generic unpack first, fallback to zip
    try:
        shutil.unpack_archive(str(archive), extract_dir=str(target_dir))
    except (shutil.ReadError, ValueError):
        if archive.suffix.lower() == ".zip":
            _safe_extract_zip(archive, target_dir)
        else:
            raise


def ensure_dataset(
    url: Optional[str],
    expected_dir: Path = DatasetPath,
    sha256: Optional[str] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Ensure BDG2 dataset exists locally; download+extract if missing.

    Returns absolute path to dataset directory, or None if unavailable.
    """
    # Already present
    if expected_dir.is_dir():
        return str(expected_dir.resolve())

    if not url:
        if logger:
            logger("No dataset URL configured (BDG2_DATASET_URL).")
        return None

    # Determine download destination
    downloads = Path("artifacts") / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    # Guess an archive name
    name = os.path.basename(url.split("?")[0]) or (DEFAULT_DATASET_DIRNAME + ".zip")
    archive_path = downloads / name

    if logger:
        logger(f"Downloading dataset archive to {archive_path} ...")
    _download(url, archive_path, progress=progress)

    if sha256 is not None:
        computed = _sha256(archive_path)
        if computed.lower() != sha256.lower():
            raise RuntimeError(f"SHA256 mismatch for {archive_path.name}: {computed} != {sha256}")

    # Extract next to project root, expecting directory name to be present or created
    if logger:
        logger("Extracting archive...")
    _unpack_archive(archive_path, target_dir=expected_dir.parent)

    # If the root folder didn't match, try to find it
    if not expected_dir.is_dir():
        # Scan for any folder that looks like BDG2 under parent
        for p in expected_dir.parent.glob("**/" + DEFAULT_DATASET_DIRNAME):
            if p.is_dir():
                return str(p.resolve())

    return str(expected_dir.resolve()) if expected_dir.is_dir() else None

