import io
import json

import pytest

from poseahead.datasets.cmu_download import download_trial


def test_download_pins_sources_rate_and_refuses_overwrite(tmp_path, monkeypatch):
    page = ('<TR>' + '<TD></TD>' * 2 + '<TD>walk</TD>'
            + '<TD><A HREF="/subjects/07/07_01.amc">amc</A></TD>'
            + '<TD></TD>' * 4 + '<TD>120</TD><TD>Feedback</TD></TR>')
    replies = iter([page.encode(), b":bonedata", b":DEGREES"])
    monkeypatch.setattr("poseahead.datasets.cmu_download.urlopen",
                        lambda *a, **kw: io.BytesIO(next(replies)))
    output = tmp_path / "trial"
    report = download_trial("07", "01", output)
    assert report["fps_numerator"] == 120
    assert report["description"] == "walk"
    assert len(report["files"]) == 3
    assert json.loads((output / "download.json").read_text()) == report
    with pytest.raises(FileExistsError):
        download_trial("07", "01", output)


@pytest.mark.parametrize("subject", ["../07", "7", "https://example.org", "-1"])
def test_download_rejects_bad_id_before_writing(tmp_path, subject):
    output = tmp_path / "trial"
    with pytest.raises(ValueError):
        download_trial(subject, "01", output)
    assert not output.exists()


def test_download_does_not_guess_rate(tmp_path, monkeypatch):
    monkeypatch.setattr("poseahead.datasets.cmu_download.urlopen",
                        lambda *a, **kw: io.BytesIO(b"<html>Unavailable</html>"))
    output = tmp_path / "trial"
    with pytest.raises(ValueError, match="not uniquely listed"):
        download_trial("07", "01", output)
    assert not (output / "download.json").exists()
