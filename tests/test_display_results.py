"""The results page. Free: it only reads the saved result files."""

from project.self_sacrifice.display import results


def test_markdown_turns_pipe_tables_into_html():
    page = results.markdown("## Title\n- a note\n| a | b |\n|---|---|\n| 1 | <2> |")
    assert "<h3>Title</h3>" in page and "<p>a note</p>" in page
    assert "<th>a</th>" in page and "<td>&lt;2&gt;</td>" in page
    assert "---" not in page


def test_the_sample_takes_the_first_gives_and_keeps_of_each_label():
    games = [("x", i % 2 == 0, {"n": i}) for i in range(20)]
    picked = results._sampled(games)
    assert [g["n"] for g in picked if g["gave"]] == [0, 2, 4]
    assert [g["n"] for g in picked if not g["gave"]] == [1, 3]


def test_the_page_renders_every_section_on_disk():
    page = results.render()
    assert page.count('<section class="card">') == len(results.sections()) + 1
    assert "__SECTIONS__" not in page and "__DATA__" not in page
    for sample in results.transcripts():
        assert sample["turns"] and sample["system"]
