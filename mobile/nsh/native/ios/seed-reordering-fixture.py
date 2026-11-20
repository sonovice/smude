#!/usr/bin/env python3
"""Prepare the Reorder26 UI fixture in an explicitly named iOS simulator only.

Call after installation, before xcodebuild's Reordering26Tests. No physical
container APIs are used; existing simulator collections are retained.
"""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

simulator = sys.argv[1]
bundle = "com.notavium.smude"
subprocess.run(["xcrun", "simctl", "terminate", simulator, bundle], capture_output=True)
container = Path(subprocess.check_output([
    "xcrun", "simctl", "get_app_container", simulator, bundle, "data"
], text=True).strip())
root = container / "Library/Application Support"
repository = root / "SmudeProjects"
repository.mkdir(parents=True, exist_ok=True)
image = repository / "renders/reordering26-fixture.png"
image.parent.mkdir(exist_ok=True)

def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

image.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 300, 424, 8, 0, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff" * 300) * 424)) + chunk(b"IEND", b""))
library_file = repository / "library.json"
library = json.loads(library_file.read_text()) if library_file.exists() else {"imports": [], "projects": []}
template = dict(source_rotation=0, quality={"bands": [], "retake_suggested": False}, status="Ready",
                revision=0, discover=False, dimensions=[300, 424], side=0, rotation=0, input_rotation=0,
                layout="Single", model="Mesh", review=False, source="renders/reordering26-fixture.png",
                output="renders/reordering26-fixture.png")
pages = []
for identifier in range(26001, 26031):
    page = copy.deepcopy(template)
    page.update(id=identifier, capture=identifier)
    pages.append(page)
library["projects"] = [p for p in library["projects"] if p["id"] != 26000] + [
    dict(id=26000, title="Reorder26", draft=False, pages=pages, modified_at=int(time.time() * 1000))]
library_file.write_text(json.dumps(library))
preferences = root / "nsh" / bundle / "storage.json"
preferences.parent.mkdir(parents=True, exist_ok=True)
values = json.loads(preferences.read_text()) if preferences.exists() else {}
values["smude-project-view"] = "list"
preferences.write_text(json.dumps(values))
print(f"Prepared 30 fixture pages in simulator {simulator}")
