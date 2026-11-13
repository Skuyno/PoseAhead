"""The advisor handout is self-contained, escaped, and never overwrites its source."""

import importlib.util
from pathlib import Path

import pytest


def renderer():
    path = Path(__file__).resolve().parents[2] / "scripts/render_report_html.py"
    spec = importlib.util.spec_from_file_location("advisor_report_html", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plain_report_has_real_headings_lists_tables_and_print_pages():
    document = renderer().render("# Отчёт\n\nТекст **жирный** и `код`.\n\n"
                                 "## Результат\n- Первый\n- Второй\n\n---\n"
                                 "## Показ\n1. Открыть\n2. Запустить\n\n"
                                 "| Метод | Ошибка |\n| --- | ---: |\n| A | 12 |\n")
    assert '<html lang="ru">' in document
    assert '<h1>Отчёт</h1>' in document
    assert '<strong>жирный</strong>' in document and '<code>код</code>' in document
    assert '<ul><' not in document  # tags are separated into readable lines
    assert document.count('<li>') == 4
    assert document.count('<section class="page">') == 2
    assert '<th scope="col">Метод</th>' in document
    assert "size: Letter portrait" in document
    assert "<script" not in document and 'src="http' not in document


def test_raw_html_is_text_not_active_content():
    document = renderer().render('# <script>alert("x")</script>\n\n'
                                 '**<img src=x>** и `<b>` & текст\n')
    assert '<script>' not in document and '<img' not in document
    assert '&lt;script&gt;' in document
    assert '<strong>&lt;img src=x&gt;</strong>' in document
    assert '<code>&lt;b&gt;</code> &amp;' in document


@pytest.mark.parametrize("source", ("", "Текст", "# ", "# Название\n# Второе"))
def test_missing_empty_or_duplicate_title_is_rejected(source):
    with pytest.raises(ValueError, match="title"):
        renderer().render(source)


@pytest.mark.parametrize("table", (
    "| A | B |\n| --- |\n| 1 | 2 |",
    "| A | B |\n| words | words |\n| 1 | 2 |",
    "| A | B |\n| --- | --- |\n| 1 |",
))
def test_malformed_table_is_rejected(table):
    with pytest.raises(ValueError, match="table"):
        renderer().render("# Отчёт\n" + table)


def test_rendering_never_overwrites_source_or_existing_output(tmp_path):
    module = renderer()
    source, output = tmp_path / "report.md", tmp_path / "report.html"
    source.write_text("# Отчёт\n\nПример\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        module.render_file(source, source)
    assert module.render_file(source, output) == output
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.render_file(source, output)
    assert output.read_bytes() == original
    assert "Пример" in output.read_text(encoding="utf-8")


def test_invalid_source_leaves_no_output(tmp_path):
    source, output = tmp_path / "report.md", tmp_path / "report.html"
    source.write_text("нет заголовка", encoding="utf-8")
    with pytest.raises(ValueError):
        renderer().render_file(source, output)
    assert not output.exists()
