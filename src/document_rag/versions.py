"""Version metadata parsed from PDF filenames."""

import re
from pathlib import Path

FILENAME_VERSION = re.compile(
    r"^(?P<family>.+)-V\.(?P<version>\d+\.\d+\.\d+)-(?P<date>\d{6}(?:\d{2})?)$",
    re.IGNORECASE,
)
MENTIONED_VERSION = re.compile(r"\b(\d+\.\d+\.\d+)\b")
CHANGE_QUESTION = re.compile(
    r"\b(chang\w*|differ\w*|revis\w*|between versions|what was (?:added|removed|updated))\b",
    re.IGNORECASE,
)


def parse_filename(filename):
    stem = Path(filename).stem
    match = FILENAME_VERSION.match(stem)
    if not match:
        return {"family": stem, "version": "", "version_date": "", "sort_key": ()}
    version = match.group("version")
    return {
        "family": match.group("family").strip(),
        "version": version,
        "version_date": match.group("date"),
        "sort_key": tuple(int(part) for part in version.split(".")),
    }


def annotate_versions(documents):
    groups = {}
    for doc in documents:
        doc.update(parse_filename(doc["filename"]))
        groups.setdefault(doc["family"], []).append(doc)
    for group in groups.values():
        group.sort(key=lambda item: (item["sort_key"], item["version_date"]))
        latest = group[-1]["version"]
        for doc in group:
            doc["is_latest"] = 1 if not latest or doc["version"] == latest else 0
    return documents


def is_change_question(question):
    return CHANGE_QUESTION.search(question) is not None


def scope_for_question(question, file_ids=None):
    """Return the document ids and Chroma filter for this question.

    Content questions search the latest version unless a version or file is named.
    Change questions search indexed section-change notes.
    """
    versions = MENTIONED_VERSION.findall(question)
    if is_change_question(question):
        where = {"kind": "change"}
        if len(versions) >= 2:
            where = {
                "$and": [
                    {"kind": "change"},
                    {"from_version": versions[0]},
                    {"to_version": versions[1]},
                ]
            }
        elif len(versions) == 1:
            where = {"$and": [{"kind": "change"}, {"to_version": versions[0]}]}
        return None, where
    if file_ids:
        return file_ids, {"kind": "content"}
    if len(versions) == 1:
        return None, {"$and": [{"kind": "content"}, {"version": versions[0]}]}
    return None, {"$and": [{"kind": "content"}, {"is_latest": 1}]}
