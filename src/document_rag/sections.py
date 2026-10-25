"""Split policy manuals on numbered headings and compare those sections."""

import re

HEADING = re.compile(
    r"(?m)^(?P<id>(?:\d{1,2}|[A-Z])(?:\.\d+)*)\.?[ \t]+(?P<title>[A-Z][A-Za-z0-9,&'’/\-(): ]{1,160})\s*$"
)
PAGE_MARK = re.compile(r"\[\[page:(\d+)\]\]")
DOCUSIGN = re.compile(r"Docusign Envelope ID: \S+")


def document_text(pages):
    parts = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if text:
            parts.append(f"[[page:{page['page']}]]\n{text}")
    return "\n\n".join(parts)


def normalize_section(text):
    cleaned = DOCUSIGN.sub("", text)
    cleaned = PAGE_MARK.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().casefold()


def split_sections(pages):
    """Return sections keyed by heading id, in document order."""
    text = document_text(pages)
    matches = []
    for match in HEADING.finditer(text):
        following = text[match.end() : match.end() + 300]
        # Table-of-contents lines use dot leaders and a trailing page number.
        if ".." in match.group(0) or ".." in following:
            continue
        matches.append(match)
    sections = []
    seen = set()
    for index, match in enumerate(matches):
        section_id = match.group("id")
        if section_id in seen:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        page_match = PAGE_MARK.search(text[max(0, match.start() - 40) : match.start()])
        earlier = list(PAGE_MARK.finditer(text[: match.start()]))
        page = int(earlier[-1].group(1)) if earlier else 1
        if page_match:
            page = int(page_match.group(1))
        seen.add(section_id)
        sections.append(
            {
                "section_id": section_id,
                "title": re.sub(r"\s+", " ", match.group("title")).strip(),
                "page": page,
                "text": re.sub(r"\s+", " ", PAGE_MARK.sub("", DOCUSIGN.sub("", body))).strip(),
            }
        )
    return sections


def compare_sections(left, right):
    left_by_id = {section["section_id"]: section for section in left}
    right_by_id = {section["section_id"]: section for section in right}
    order = []
    for section in left + right:
        if section["section_id"] not in order:
            order.append(section["section_id"])
    changes = []
    for section_id in order:
        before = left_by_id.get(section_id)
        after = right_by_id.get(section_id)
        if before and after and normalize_section(before["text"]) == normalize_section(after["text"]):
            continue
        if before and not after:
            change = "removed"
        elif after and not before:
            change = "added"
        else:
            change = "modified"
        title = (after or before)["title"]
        changes.append(
            {
                "section_id": section_id,
                "title": title,
                "change": change,
                "from_page": before["page"] if before else None,
                "to_page": after["page"] if after else None,
                "before": before["text"] if before else "",
                "after": after["text"] if after else "",
            }
        )
    return changes
