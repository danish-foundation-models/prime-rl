"""Stage and verify the small, pinned English NLTK tables before training."""

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path


def prepare(root: Path):
    manifest = json.loads(Path(__file__).with_name("nltk_assets.json").read_text())
    missing = []
    for name, expected in manifest["files"].items():
        path = root / "tokenizers" / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Unexpected tokenizer content: {path}")
        else:
            missing.append(name)
    if not missing:
        return
    data = urllib.request.urlopen(manifest["url"], timeout=60).read()
    if hashlib.sha256(data).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Tokenizer archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in missing:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid tokenizer member: {name}")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Tokenizer member checksum mismatch: {name}")
            target = root / "tokenizers" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as file:
                file.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
