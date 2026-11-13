"""Small, explicit CMU downloads; never crawl the complete database."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://mocap.cs.cmu.edu"


def download_trial(subject: str, trial: str, output: str | Path):
    """Create a new directory containing originals, page evidence, and hashes."""
    if not all(re.fullmatch(r"[0-9]{2,3}", value) for value in (subject, trial)):
        raise ValueError("subject and trial must be zero-padded numeric CMU IDs")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    entries = []

    def fetch(remote, local):
        with urlopen(BASE_URL + remote, timeout=30) as response:
            data = response.read(20_000_001)
        if not data or len(data) > 20_000_000:
            raise ValueError("empty or unexpectedly large CMU response")
        with (directory / local).open("xb") as stream:
            stream.write(data)
        entries.append({"file": local, "url": BASE_URL + remote,
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return data.decode("utf-8")

    page = fetch(f"/search.php?subjectnumber={subject}", "source.html")
    motion = f"{subject}_{trial}"
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", page, re.I | re.S)
    matching = [row for row in rows if f"/subjects/{subject}/{motion}.amc" in row]
    if len(matching) != 1:
        raise ValueError("trial not uniquely listed on CMU page")
    cells = re.findall(r"<td\b[^>]*>(.*?)</td>", matching[0], re.I | re.S)
    if len(cells) != 10 or not cells[8].strip().isdigit():
        raise ValueError("cannot verify sample rate from CMU page")
    fps = int(cells[8].strip())
    if fps <= 0:
        raise ValueError("invalid CMU sample rate")
    asf = fetch(f"/subjects/{subject}/{subject}.asf", f"{subject}.asf")
    amc = fetch(f"/subjects/{subject}/{motion}.amc", f"{motion}.amc")
    if ":bonedata" not in asf or ":DEGREES" not in amc:
        raise ValueError("response is not an expected ASF/AMC file")
    manifest = {"format": "poseahead-cmu-download", "version": 1,
                "subject": subject, "trial": trial, "description": cells[2].strip(),
                "fps_numerator": fps, "files": entries,
                "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    with (directory / "download.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest
