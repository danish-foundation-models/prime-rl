"""Add verified static search tools to new, immutable offline harness bundles."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

TOOLS = {
    "rg": ("15.1.0", "1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599", "ripgrep-15.1.0-x86_64-unknown-linux-musl/rg"),
    "fd": ("10.3.0", "2b6bfaae8c48f12050813c2ffe1884c61ea26e750d803df9c9114550a314cd14", "fd-v10.3.0-x86_64-unknown-linux-musl/fd"),
}


def build(source: Path, output: Path, archives: dict[str, Path]):
    binaries = {}
    for name, (_, digest, member) in TOOLS.items():
        blob = archives[name].read_bytes()
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"{name} upstream archive checksum mismatch")
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            entry = tar.getmember(member)
            if not entry.isfile():
                raise ValueError(f"{name} is not a regular file")
            binaries[name] = tar.extractfile(entry).read()
    output.mkdir(parents=True, exist_ok=True)
    for harness in ("opencode", "pi", "prime-agent"):
        src = source / f"{harness}-assets.tar.gz"
        manifest = json.loads(src.with_name(src.name + ".json").read_text())
        source_digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if source_digest != manifest["sha256"]:
            raise ValueError(f"{harness} source checksum mismatch")
        dest = output / src.name
        # Never replace a published bundle or silently reuse a partial build.
        if dest.exists() or dest.with_name(dest.name + ".json").exists():
            raise FileExistsError(dest)
        with tarfile.open(src) as old, tarfile.open(dest, "x:gz") as new:
            for entry in old:
                if entry.name in {f"var/tmp/vf-node/bin/{tool}" for tool in TOOLS}:
                    raise ValueError("Source already contains search tools")
                new.addfile(entry, old.extractfile(entry) if entry.isfile() else None)
            for name, data in binaries.items():
                entry = tarfile.TarInfo(f"var/tmp/vf-node/bin/{name}")
                entry.mode = 0o755
                entry.size = len(data)
                new.addfile(entry, io.BytesIO(data))
        manifest.update(source_sha256=source_digest, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                        search_tools={name: {"version": v, "archive_sha256": digest,
                                            "binary_sha256": hashlib.sha256(binaries[name]).hexdigest()}
                                      for name, (v, digest, _) in TOOLS.items()})
        dest.with_name(dest.name + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(harness, manifest["sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ripgrep", type=Path, required=True)
    parser.add_argument("--fd", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output, {"rg": args.ripgrep, "fd": args.fd})
