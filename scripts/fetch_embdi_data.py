#!/usr/bin/env python3
"""Download, verify, and extract the EmbDI reproducibility inputs from Zenodo."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "7930461"
ARCHIVE_KEY = "EmbDI datasets.zip"
RECORD_API = "https://zenodo.org/api/records/{}".format(RECORD_ID)
ARCHIVE_PREFIX = "EmbDI datasets/"
EXTRACT_PREFIXES = (
    "EmbDI datasets/pipeline/datasets/",
    "EmbDI datasets/pipeline/edgelists/",
    "EmbDI datasets/pipeline/test_dir/",
)
DEST_PREFIX = "pipeline/"
USER_AGENT = "embdi-repro-fetch/1.0"


def _request(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req)


def fetch_record():
    with _request(RECORD_API) as response:
        return json.load(response)


def record_archive(record):
    for entry in record.get("files", []):
        if entry["key"] == ARCHIVE_KEY:
            size = int(entry["size"])
            checksum = entry["checksum"]
            md5 = checksum.split(":", 1)[1] if ":" in checksum else checksum
            return {
                "key": entry["key"],
                "size": size,
                "md5": md5,
                "url": entry["links"]["self"],
            }
    raise SystemExit("Archive {!r} not found in Zenodo record {}".format(ARCHIVE_KEY, RECORD_ID))


def digest(path, algorithm):
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_archive(path, archive):
    size = path.stat().st_size
    if size != archive["size"]:
        return False, "size {} != expected {}".format(size, archive["size"])
    md5 = digest(path, "md5")
    if md5 != archive["md5"]:
        return False, "md5 {} != expected {}".format(md5, archive["md5"])
    return True, "size and md5 match"


def download(url, destination, archive, force=False):
    if destination.exists() and not force:
        valid, reason = verify_archive(destination, archive)
        if valid:
            print("Reusing verified archive: {}".format(destination))
            return
        print("Discarding invalid archive ({}): {}".format(reason, destination))
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading {}".format(url))
    with _request(url) as response, open(destination, "wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        last_report = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total and (done - last_report >= 100 * 1024 * 1024):
                print("  {:.0f}/{} MiB".format(done / 2**20, total / 2**20))
                last_report = done

    valid, reason = verify_archive(destination, archive)
    if not valid:
        raise SystemExit("Downloaded archive failed verification: {}".format(reason))
    print("Verified archive: {}".format(reason))


def extract_inputs(archive_path, repo_root):
    extracted = 0
    with zipfile.ZipFile(archive_path) as bundle:
        for name in bundle.namelist():
            if not name.startswith(EXTRACT_PREFIXES) or name.endswith("/"):
                continue
            relative = name[len(ARCHIVE_PREFIX):]
            target = (repo_root / relative).resolve()
            if repo_root.resolve() not in target.parents:
                raise SystemExit("Refusing to extract outside repository: {}".format(name))
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted += 1
    print("Extracted {} input files into {}".format(extracted, repo_root / "pipeline"))


def build_manifest(record, archive, repo_root):
    files = {}
    for prefix in EXTRACT_PREFIXES:
        directory = repo_root / prefix[len(ARCHIVE_PREFIX):]
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                relative = path.relative_to(repo_root).as_posix()
                files[relative] = {
                    "size": path.stat().st_size,
                    "sha256": digest(path, "sha256"),
                }
    metadata = record["metadata"]
    return {
        "record_id": RECORD_ID,
        "doi": record.get("doi"),
        "concept_doi": record.get("conceptdoi"),
        "title": metadata.get("title"),
        "archive": {key: archive[key] for key in ("key", "size", "md5")},
        "extract_prefixes": list(EXTRACT_PREFIXES),
        "files": files,
    }


def verify_manifest(manifest, repo_root):
    expected = manifest.get("files", {})
    missing, mismatched = [], []
    for relative, info in sorted(expected.items()):
        path = repo_root / relative
        if not path.exists():
            missing.append(relative)
            continue
        if path.stat().st_size != info["size"] or digest(path, "sha256") != info["sha256"]:
            mismatched.append(relative)
    if missing or mismatched:
        for item in missing:
            print("MISSING  {}".format(item))
        for item in mismatched:
            print("MISMATCH {}".format(item))
        raise SystemExit("Manifest verification failed")
    print("Verified {} files against manifest".format(len(expected)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "embdi-repro")
    parser.add_argument("--archive", type=Path, default=None, help="Use an existing local archive")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "embdi-data-manifest.json")
    parser.add_argument("--write-manifest", action="store_true", help="Rewrite the checksum manifest")
    parser.add_argument("--force", action="store_true", help="Re-download the archive")
    parser.add_argument("--verify-only", action="store_true", help="Verify the extracted tree without downloading")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()

    if args.verify_only:
        if not args.manifest.exists():
            raise SystemExit("Manifest not found: {}".format(args.manifest))
        verify_manifest(json.loads(args.manifest.read_text()), repo_root)
        return

    record = fetch_record()
    archive = record_archive(record)

    if args.archive:
        archive_path = args.archive
        if not archive_path.exists():
            raise SystemExit("Archive not found: {}".format(archive_path))
    else:
        archive_path = args.cache_dir / archive["key"]
        download(archive["url"], archive_path, archive, force=args.force)

    valid, reason = verify_archive(archive_path, archive)
    if not valid:
        raise SystemExit("Archive verification failed: {}".format(reason))
    print("Archive verified: {}".format(reason))

    extract_inputs(archive_path, repo_root)

    if args.write_manifest:
        manifest = build_manifest(record, archive, repo_root)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print("Wrote manifest: {}".format(args.manifest))
    elif args.manifest.exists():
        verify_manifest(json.loads(args.manifest.read_text()), repo_root)


if __name__ == "__main__":
    main()
