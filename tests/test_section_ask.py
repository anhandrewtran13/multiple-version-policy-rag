"""Ask over tracker sections, including manuals that store no page numbers."""

import json
from pathlib import Path

from document_rag.config import RagError
from document_rag.section_ask import _blend, comparison_href, passages_for


def test_manual_sections_without_pages_still_search(tmp_path):
    data = tmp_path / "data"
    folder = data / "bhsa-manual"
    folder.mkdir(parents=True)
    versions = ["V1.0.0", "V1.2.0", "V1.5.1"]
    history = {
        "versions": versions,
        "texts": ["# Intro\n\nCounties submit the plan each year."],
        "sections": [
            {
                "chapter": "intro",
                "label": "1",
                "title": "Introduction",
                "texts": [0, 0, 0],
                "pages": [],
            }
        ],
    }
    (folder / "history.json").write_text(json.dumps(history))
    found = passages_for(data, "bhsa-manual", "What do counties submit in 1.5.1?")
    assert len(found) == 1
    assert found[0]["version"] == "V1.5.1"
    assert found[0]["page"] is None
    assert "Counties submit" in found[0]["text"]


def test_citation_link_uses_a_real_comparison_pair():
    versions = ["BHIN-25-013", "BHIN-26-015", "BHIN-26-023"]
    assert comparison_href("bhin-network-adequacy", versions, "BHIN-26-023", "network-adequacy/III.c") == (
        "#bhin-network-adequacy/BHIN-26-015~BHIN-26-023~network-adequacy/III.c"
    )
    assert comparison_href("bhin-network-adequacy", versions, "BHIN-25-013", "network-adequacy/I") == (
        "#bhin-network-adequacy/BHIN-25-013~BHIN-26-015~network-adequacy/I"
    )


def test_title_match_outranks_a_longer_unrelated_section(tmp_path):
    class Offline:
        def embeddings(self):
            raise RagError("Ollama is not running.")

    question = "What do counties submit for the integrated plan?"
    passages = [
        {
            "title": "Policy 2: Consistently Bill Commercial Insurance",
            "text": "Counties must require that providers submit claims to the commercial plan before billing.",
        },
        {
            "title": "Integrated Plan Submission",
            "text": "Counties submit the integrated plan and budget template to DHCS.",
        },
    ]
    ordered = _blend(question, passages, Offline(), tmp_path / "vectors.json")
    assert ordered[0]["title"] == "Integrated Plan Submission"


def test_live_histories_can_be_searched():
    data = Path("site/data")
    if not (data / "library.json").is_file():
        return
    library = json.loads((data / "library.json").read_text())
    for document in library["documents"]:
        found = passages_for(data, document["id"], "What changed in the current version?")
        assert found, document["id"]
        assert all("page" in item for item in found)
