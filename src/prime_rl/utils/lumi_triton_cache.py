"""Publish shared Triton artifacts after flushing data to Lustre."""

import json
import os
import tempfile
from pathlib import Path

from triton.runtime.cache import FileCacheManager


class DurableFileCacheManager(FileCacheManager):
    def put(self, data, filename, binary=True):
        if Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError("Triton cache filenames must be single path components")
        if not self.cache_dir or self.lock_path is None:
            raise RuntimeError("Triton cache directory is not writable")
        is_binary = isinstance(data, bytes)
        payload = data if is_binary else str(data)
        mode = "wb" if is_binary else "w"
        fd, temporary = tempfile.mkstemp(prefix=".publishing-", dir=self.cache_dir)
        # Retain an interrupted private file for diagnosis; never sweep the cache.
        with os.fdopen(fd, mode) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        destination = self._make_path(filename)
        os.replace(temporary, destination)
        return destination

    def get_group(self, filename):
        manifest = self.get_file(f"__grp__{filename}")
        if manifest is None:
            return None
        try:
            group = json.loads(Path(manifest).read_text())
            if not isinstance(group, dict):
                return None
            child_paths = group["child_paths"]
            if not isinstance(child_paths, dict) or not child_paths:
                return None
            # FileCacheManager silently drops absent children. A partial group
            # can retain metadata but omit the GPU binary, crashing compilation.
            for name, path in child_paths.items():
                if not isinstance(name, str) or not isinstance(path, str):
                    return None
                if not Path(path).is_file():
                    return None
                if name.endswith(".json"):
                    json.loads(Path(path).read_text())
            return child_paths
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError, KeyError):
            return None
