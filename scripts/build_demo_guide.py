"""Dependency-free DOCX fallback for the presentation guide; not a visual renderer.

The document skill runtime and LibreOffice are unavailable on this machine.
Uses explicit compact_reference_guide tokens and memo_masthead header, real Word
numbering, fixed DXA tables. Only the small Markdown subset used by this guide is
supported. Named overrides: Title 24pt, Subtitle 12pt, Code 9pt/1.10, Table 10pt/1.10,
Citation 9pt/1.10, running furniture 9pt. Normal/heading/list geometry is unchanged.
"""

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("w", W)
ET.register_namespace("r", R)


def node(tag, attrs=None, text=None):
    result = ET.Element(f"{{{W}}}{tag}", {f"{{{W}}}{k}": str(v) for k, v in (attrs or {}).items()})
    if text is not None:
        result.text = text
    return result


def child(parent, tag, attrs=None, text=None):
    result = node(tag, attrs, text)
    parent.append(result)
    return result


def xml(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def styles():
    root = node("styles")
    # Explicit tokens, no reliance on Office template defaults.
    specs = {
        "Normal": ("Calibri", 22, "000000", 0, 120, 300, False),
        "Title": ("Calibri", 48, "0B2545", 0, 120, 300, True),
        "Subtitle": ("Calibri", 24, "555555", 0, 160, 300, False),
        "Heading1": ("Calibri", 32, "2E74B5", 360, 200, 300, True),
        "Heading2": ("Calibri", 26, "2E74B5", 280, 140, 300, True),
        "Heading3": ("Calibri", 24, "1F4D78", 200, 100, 300, True),
        "ListBullet": ("Calibri", 22, "000000", 0, 80, 300, False),
        "ListNumber": ("Calibri", 22, "000000", 0, 80, 300, False),
        "Code": ("Consolas", 18, "000000", 0, 80, 264, False),
        "TableText": ("Calibri", 20, "000000", 0, 0, 264, False),
        "TableCitation": ("Calibri", 18, "555555", 80, 80, 264, False),
        "Header": ("Calibri", 18, "555555", 0, 0, 264, False),
        "Footer": ("Calibri", 18, "555555", 0, 0, 264, False),
    }
    for name, (font, size, color, before, after, line, bold) in specs.items():
        style = child(root, "style", {"type": "paragraph", "styleId": name})
        if name == "Normal":
            style.set(f"{{{W}}}default", "1")
        child(style, "name", {"val": re.sub(r"Heading(\d)", r"heading \1", name)})
        if name != "Normal":
            child(style, "basedOn", {"val": "Normal"})
        child(style, "next", {"val": "Normal"})
        props = child(style, "pPr")
        child(props, "spacing", {"before": before, "after": after,
                                  "line": line, "lineRule": "auto"})
        child(props, "jc", {"val": "right" if name == "Footer" else "left"})
        child(props, "widowControl", {"val": 1})
        if name.startswith("Heading") or name in ("Title", "Subtitle"):
            child(props, "keepNext")
            child(props, "keepLines")
        if name.startswith("Heading"):
            child(props, "outlineLvl", {"val": int(name[-1]) - 1})
        if name in ("ListBullet", "ListNumber"):
            child(props, "ind", {"left": 540, "hanging": 271})
        if name == "Code":
            child(props, "shd", {"val": "clear", "fill": "F4F6F9"})
        run = child(style, "rPr")
        child(run, "rFonts", {"ascii": font, "hAnsi": font, "eastAsia": font, "cs": font})
        child(run, "sz", {"val": size})
        child(run, "szCs", {"val": size})
        child(run, "color", {"val": color})
        child(run, "b", {"val": int(bold)})
        child(run, "lang", {"val": "ru-RU"})
    return root


def numbering(step_lists):
    root = node("numbering")
    for index, fmt, label in ((0, "bullet", "•"), (1, "decimal", "%1.")):
        abstract = child(root, "abstractNum", {"abstractNumId": index})
        child(abstract, "multiLevelType", {"val": "singleLevel"})
        level = child(abstract, "lvl", {"ilvl": 0})
        child(level, "start", {"val": 1})
        child(level, "numFmt", {"val": fmt})
        child(level, "lvlText", {"val": label})
        child(level, "lvlJc", {"val": "left"})
        props = child(level, "pPr")
        child(child(props, "tabs"), "tab", {"val": "num", "pos": 540})
        child(props, "ind", {"left": 540, "hanging": 271})
        child(props, "spacing", {"before": 0, "after": 80, "line": 300, "lineRule": "auto"})
    child(child(root, "num", {"numId": 1}), "abstractNumId", {"val": 0})
    for num_id in range(2, step_lists + 2):
        num = child(root, "num", {"numId": num_id})
        child(num, "abstractNumId", {"val": 1})
        child(child(num, "lvlOverride", {"ilvl": 0}), "startOverride", {"val": 1})
    return root


def paragraph(text, style="Normal", *, number=None, page_break=False, bold=False):
    result = node("p")
    props = child(result, "pPr")
    child(props, "pStyle", {"val": style})
    if page_break:
        child(props, "pageBreakBefore")
    if number is not None:
        num = child(props, "numPr")
        child(num, "ilvl", {"val": 0})
        child(num, "numId", {"val": number})
    run = child(result, "r")
    if bold:
        child(child(run, "rPr"), "b")
    value = child(run, "t", text=text)
    value.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return result


def table(rows):
    if len(rows[0]) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Guide tables require exactly three columns.")
    widths = (1680, 3840, 3840)
    result = node("tbl")
    props = child(result, "tblPr")
    child(props, "tblW", {"w": 9360, "type": "dxa"})
    child(props, "tblInd", {"w": 120, "type": "dxa"})
    child(props, "tblLayout", {"type": "fixed"})
    borders = child(props, "tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        child(borders, edge, {"val": "single", "sz": 4, "color": "B7C7D9"})
    margins = child(props, "tblCellMar")
    for edge, size in (("top", 80), ("bottom", 80), ("start", 120), ("end", 120)):
        child(margins, edge, {"w": size, "type": "dxa"})
    grid = child(result, "tblGrid")
    for width in widths:
        child(grid, "gridCol", {"w": width})
    for index, values in enumerate(rows):
        row = child(result, "tr")
        row_props = child(row, "trPr")
        child(row_props, "cantSplit")
        if index == 0:
            child(row_props, "tblHeader")
        for width, value in zip(widths, values, strict=True):
            cell = child(row, "tc")
            cell_props = child(cell, "tcPr")
            child(cell_props, "tcW", {"w": width, "type": "dxa"})
            child(cell_props, "vAlign", {"val": "center"})
            if index == 0:
                child(cell_props, "shd", {"val": "clear", "fill": "E8EEF5"})
            content = paragraph(value, "TableText", bold=index == 0)
            child(content.find(f"{{{W}}}pPr"), "jc", {"val": "center"})
            cell.append(content)
    return result


def make_document(markdown):
    root = node("document")
    body = child(root, "body")
    lines = markdown.splitlines()
    pending_break, code, in_steps, step_lists = False, False, False, 0
    index, first_body = 0, True
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line.startswith("```"):
            code = not code
            in_steps = False
            continue
        if not line:
            in_steps = False
            continue
        if line == "<!-- pagebreak -->":
            pending_break = True
            continue
        if code:
            body.append(paragraph(line, "Code", page_break=pending_break))
        elif line.startswith("|"):
            rows = []
            index -= 1
            while index < len(lines) and lines[index].strip().startswith("|"):
                values = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in values):
                    rows.append(values)
                index += 1
            body.append(table(rows))
        elif line.startswith("# "):
            body.append(paragraph(line[2:], "Title", page_break=pending_break))
        elif line.startswith("## "):
            body.append(paragraph(line[3:], "Heading1", page_break=pending_break))
        elif line.startswith("### "):
            body.append(paragraph(line[4:], "Heading2", page_break=pending_break))
        elif line.startswith("- "):
            body.append(paragraph(line[2:], "ListBullet", number=1, page_break=pending_break))
        elif re.match(r"\d+\. ", line):
            if not in_steps:
                step_lists += 1
            in_steps = True
            body.append(paragraph(re.sub(r"^\d+\. ", "", line), "ListNumber",
                                  number=step_lists + 1, page_break=pending_break))
        else:
            style = ("Subtitle" if first_body else "TableCitation"
                     if line.startswith("Источник:") else "Normal")
            body.append(paragraph(line, style, page_break=pending_break))
            first_body = False
        pending_break = False
    if code:
        raise ValueError("Unclosed code block.")
    section = child(body, "sectPr")
    for kind, rid in (("headerReference", "rIdHeader"), ("footerReference", "rIdFooter")):
        ref = child(section, kind, {"type": "default"})
        ref.set(f"{{{R}}}id", rid)
    child(section, "pgSz", {"w": 12240, "h": 15840})
    child(section, "pgMar", {"top": 1440, "right": 1440, "bottom": 1440,
                              "left": 1440, "header": 708, "footer": 708, "gutter": 0})
    return root, step_lists


def audit_document(path):
    """OOXML geometry/content gate. Does not substitute for rendering."""
    ns = {"w": W}
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(archive.read(name))
        document = ET.fromstring(archive.read("word/document.xml"))
        section = document.find(".//w:sectPr", ns)
        assert section.find("w:pgSz", ns).get(f"{{{W}}}w") == "12240"
        assert section.find("w:pgSz", ns).get(f"{{{W}}}h") == "15840"
        for edge in ("top", "right", "bottom", "left"):
            assert section.find("w:pgMar", ns).get(f"{{{W}}}{edge}") == "1440"
        for edge in ("header", "footer"):
            assert section.find("w:pgMar", ns).get(f"{{{W}}}{edge}") == "708"
        for item in document.findall(".//w:tbl", ns):
            widths = [int(x.get(f"{{{W}}}w")) for x in item.findall("w:tblGrid/w:gridCol", ns)]
            assert sum(widths) == 9360
            assert item.find("w:tblPr/w:tblW", ns).get(f"{{{W}}}w") == "9360"
            assert item.find("w:tblPr/w:tblInd", ns).get(f"{{{W}}}w") == "120"
            for row in item.findall("w:tr", ns):
                actual = [int(x.get(f"{{{W}}}w")) for x in row.findall("w:tc/w:tcPr/w:tcW", ns)]
                assert widths == actual
            assert not item.findall(".//w:trHeight", ns)
        style_root = ET.fromstring(archive.read("word/styles.xml"))
        normal = style_root.find("w:style[@w:styleId='Normal']/w:pPr/w:spacing", ns)
        assert normal.get(f"{{{W}}}line") == "300"
        assert normal.get(f"{{{W}}}after") == "120"
        for name, size, before, after, color in (
            ("Heading1", 32, 360, 200, "2E74B5"),
            ("Heading2", 26, 280, 140, "2E74B5"),
            ("Heading3", 24, 200, 100, "1F4D78"),
        ):
            style = style_root.find(f"w:style[@w:styleId='{name}']", ns)
            assert style.find("w:rPr/w:sz", ns).get(f"{{{W}}}val") == str(size)
            assert style.find("w:rPr/w:color", ns).get(f"{{{W}}}val") == color
            spacing = style.find("w:pPr/w:spacing", ns)
            for key, value in (("before", before), ("after", after), ("line", 300)):
                assert spacing.get(f"{{{W}}}{key}") == str(value)
        numbers = ET.fromstring(archive.read("word/numbering.xml"))
        for indent in numbers.findall(".//w:ind", ns):
            assert indent.get(f"{{{W}}}left") == "540"
            assert indent.get(f"{{{W}}}hanging") == "271"
        declared_numbers = {x.get(f"{{{W}}}numId") for x in numbers.findall("w:num", ns)}
        for number in document.findall(".//w:numPr/w:numId", ns):
            assert number.get(f"{{{W}}}val") in declared_numbers
        rel_ns = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}
        relationships = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        for relationship in relationships.findall("p:Relationship", rel_ns):
            assert "word/" + relationship.get("Target") in archive.namelist()
        assert document.findall(".//w:numPr", ns)
        assert len(document.findall(".//w:pageBreakBefore", ns)) == 5
        text = " ".join(x.text or "" for x in document.findall(".//w:t", ns))
        assert "<!--" not in text and "codex-file-citation" not in text
        for item in document.findall(".//w:p", ns):
            value = "".join(x.text or "" for x in item.findall(".//w:t", ns))
            assert not re.match(r"^(?:- |• |\d+\. )", value), "Fake list marker"
    return {"status": "structurally_verified", "visual_qa": "not_completed_no_libreoffice"}


def build_guide(source, output):
    if Path(output).exists():
        raise FileExistsError("Choose a new DOCX path; existing files are preserved.")
    document, step_lists = make_document(Path(source).read_text(encoding="utf-8"))
    header = node("hdr")
    header.append(paragraph("PoseAhead | Демонстрация рабочего прототипа", "Header"))
    footer = node("ftr")
    footer.append(paragraph("13.10.2026  ·  ", "Footer"))
    child(footer[0], "fldSimple", {"instr": "PAGE"}).append(node("r"))
    child(footer[0][-1][0], "t", text="1")
    content_types = ET.Element(
        "Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types",
    )
    ET.SubElement(content_types, "Default", Extension="rels",
                  ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(content_types, "Default", Extension="xml", ContentType="application/xml")
    for name, kind in (("document", "document.main"), ("styles", "styles"),
                       ("numbering", "numbering"), ("header1", "header"), ("footer1", "footer")):
        ET.SubElement(content_types, "Override", PartName=f"/word/{name}.xml",
                      ContentType=f"application/vnd.openxmlformats-officedocument."
                                  f"wordprocessingml.{kind}+xml")
    package_rels = ET.Element(
        "Relationships", xmlns="http://schemas.openxmlformats.org/package/2006/relationships",
    )
    ET.SubElement(package_rels, "Relationship", Id="rId1", Type=f"{R}/officeDocument",
                  Target="word/document.xml")
    rels = ET.Element(
        "Relationships", xmlns="http://schemas.openxmlformats.org/package/2006/relationships",
    )
    for rid, kind, target in (
        ("rIdStyles", "styles", "styles.xml"), ("rIdNumbers", "numbering", "numbering.xml"),
        ("rIdHeader", "header", "header1.xml"), ("rIdFooter", "footer", "footer1.xml"),
    ):
        ET.SubElement(rels, "Relationship", Id=rid,
                      Type=f"{R}/{kind}", Target=target)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name, value in (
            ("[Content_Types].xml", xml(content_types)), ("_rels/.rels", xml(package_rels)),
            ("word/document.xml", xml(document)), ("word/styles.xml", xml(styles())),
            ("word/numbering.xml", xml(numbering(step_lists))),
            ("word/header1.xml", xml(header)), ("word/footer1.xml", xml(footer)),
            ("word/_rels/document.xml.rels", xml(rels)),
        ):
            archive.writestr(name, value)
    return audit_document(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/demo_guide.md"))
    parser.add_argument("--output", type=Path, default=Path("docs/demo_guide.docx"))
    args = parser.parse_args()
    print(build_guide(args.source, args.output))


if __name__ == "__main__":
    main()
