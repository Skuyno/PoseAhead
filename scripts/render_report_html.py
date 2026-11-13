"""Render the small, local advisor-report Markdown subset as printable HTML.

No network, JavaScript, document-runtime dependency, camera or application startup.
This is not a general Markdown engine or a DOCX converter. A standalone `---`
starts a new printed Letter page; all text is escaped before HTML insertion.
"""

import argparse
import html
import re
from pathlib import Path

STYLE = """
* { box-sizing: border-box; }
html { background: #edf0f3; color: #000; }
body { margin: 0; font: 11.5pt/1.32 Arial, sans-serif; }
.page { width: 8.5in; min-height: 11in; margin: 20px auto;
  padding: .75in .85in; background: #fff; box-shadow: 0 2px 12px #0001; }
h1, h2, h3 { color: #000; line-height: 1.18; break-after: avoid; }
h1 { font-size: 23pt; margin: 0 0 14pt; font-weight: 700; }
h2 { font-size: 16pt; margin: 18pt 0 9pt; }
h3 { font-size: 12.5pt; margin: 14pt 0 7pt; }
.page > :first-child { margin-top: 0; }
p { margin: 0 0 9pt; }
ul, ol { margin: 0 0 10pt; padding-left: 20pt; }
li { margin: 0 0 6pt; }
code { font: .92em Consolas, monospace; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0 12pt;
  font-size: 10.5pt; line-height: 1.25; table-layout: fixed; }
th, td { border: 1px solid #d9d9d9; padding: 6pt;
  vertical-align: middle; overflow-wrap: anywhere; }
th { background: #e3ecf4; color: #000; font-weight: 700; }
td { text-align: center; }
td:first-child, th:first-child { text-align: left; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
@media (max-width: 850px) {
  .page { width: 100%; min-height: 0; padding: 24px; margin: 0 0 12px; }
  table { font-size: 10pt; }
}
@page { size: Letter portrait; margin: .75in .85in; }
@media print {
  html { background: #fff; }
  body { margin: 0; }
  .page { width: auto; min-height: 0; padding: 0; margin: 0;
    box-shadow: none; break-after: page; }
  .page:last-child { break-after: auto; }
}
"""


def inline(text):
    tokens = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text)
    return "".join(
        "<code>" + html.escape(token[1:-1]) + "</code>" if token.startswith("`")
        else "<strong>" + html.escape(token[2:-2]) + "</strong>" if token.startswith("**")
        else html.escape(token) for token in tokens
    )


def table_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render(markdown):
    lines = markdown.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("report must start with a single document title")
    title = lines[0][2:].strip()
    if not title:
        raise ValueError("report title must not be empty")
    blocks = ['<section class="page">']
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "---":
            blocks.append('</section><section class="page">')
            index += 1
            continue
        heading = re.fullmatch(r"(#{1,3}) (.+)", line)
        if heading:
            level, text = len(heading[1]), heading[2].strip()
            if level == 1 and index != 0:
                raise ValueError("report must contain exactly one document title")
            blocks.append(f"<h{level}>{inline(text)}</h{level}>")
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(table_row(lines[index]))
                index += 1
            if (len(rows) < 3 or len(rows[0]) < 2
                    or not all(re.fullmatch(r":?-{3,}:?", item) for item in rows[1])
                    or any(len(row) != len(rows[0]) for row in rows)):
                raise ValueError("report table must have matching columns and a header separator")
            columns = len(rows[0])
            widths = {2: (65, 35), 3: (50, 25, 25), 4: (40, 20, 20, 20),
                      5: (16, 21, 21, 21, 21)}.get(columns, (100 / columns,) * columns)
            blocks.append("<table><colgroup>" + "".join(
                f'<col style="width:{width:g}%">' for width in widths
            ) + "</colgroup><thead><tr>" + "".join(
                '<th scope="col">' + inline(cell) + "</th>" for cell in rows[0]
            ) + "</tr></thead><tbody>")
            for row in rows[2:]:
                blocks.append("<tr>" + "".join("<td>" + inline(cell) + "</td>" for cell in row)
                              + "</tr>")
            blocks.append("</tbody></table>")
            continue
        list_item = re.fullmatch(r"(- |\d+\. )(.+)", line)
        if list_item:
            ordered = list_item[1] != "- "
            tag = "ol" if ordered else "ul"
            blocks.append(f"<{tag}>")
            while index < len(lines):
                item = re.fullmatch(r"(- |\d+\. )(.+)", lines[index].strip())
                if not item or (item[1] != "- ") != ordered:
                    break
                blocks.append("<li>" + inline(item[2]) + "</li>")
                index += 1
            blocks.append(f"</{tag}>")
            continue
        # Source paragraphs deliberately use one physical line each.
        blocks.append("<p>" + inline(line) + "</p>")
        index += 1
    blocks.append("</section>")
    return ("<!doctype html>\n<html lang=\"ru\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>" + html.escape(title) + "</title><style>" + STYLE
            + "</style></head><body>\n" + "\n".join(blocks) + "\n</body></html>\n")


def render_file(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("input and output must differ")
    document = render(source.read_text(encoding="utf-8"))
    # Refuse to overwrite a user's file or an earlier delivered report.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        path = render_file(args.input, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    print("Prepared local printable report: " + str(path))


if __name__ == "__main__":
    main()
