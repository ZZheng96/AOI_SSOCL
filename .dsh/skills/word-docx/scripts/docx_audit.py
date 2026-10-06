#!/usr/bin/env python3
"""Audit a DOCX package for fragile Word/OOXML structures."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def parse_xml(zf: zipfile.ZipFile, name: str) -> ET.Element | None:
    try:
        return ET.fromstring(zf.read(name))
    except (KeyError, ET.ParseError):
        return None


def count(root: ET.Element | None, xpath: str) -> int:
    if root is None:
        return 0
    return len(root.findall(xpath, NS))


def attr_value(elem: ET.Element, namespace: str, key: str) -> str | None:
    return elem.attrib.get(f"{{{NS[namespace]}}}{key}")


def visible_text(root: ET.Element | None, limit: int) -> str:
    if root is None or limit <= 0:
        return ""
    chunks = []
    length = 0
    for text_node in root.findall(".//w:t", NS):
        text = text_node.text or ""
        if not text:
            continue
        chunks.append(text)
        length += len(text)
        if length >= limit:
            break
    text = "".join(chunks)
    return text[:limit]


def rel_targets(root: ET.Element | None) -> dict[str, int]:
    if root is None:
        return {}
    totals: dict[str, int] = {}
    for rel in root.findall(".//rel:Relationship", NS):
        rel_type = rel.attrib.get("Type", "")
        key = rel_type.rsplit("/", 1)[-1] if rel_type else "unknown"
        totals[key] = totals.get(key, 0) + 1
    return dict(sorted(totals.items()))


def audit_docx(path: Path, text_limit: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a ZIP/DOCX package: {path}")

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        document = parse_xml(zf, "word/document.xml")
        rels = parse_xml(zf, "word/_rels/document.xml.rels")
        comments = parse_xml(zf, "word/comments.xml")
        footnotes = parse_xml(zf, "word/footnotes.xml")
        endnotes = parse_xml(zf, "word/endnotes.xml")
        styles = parse_xml(zf, "word/styles.xml")
        numbering = parse_xml(zf, "word/numbering.xml")

        headers = sorted(n for n in names if n.startswith("word/header") and n.endswith(".xml"))
        footers = sorted(n for n in names if n.startswith("word/footer") and n.endswith(".xml"))
        media = sorted(n for n in names if n.startswith("word/media/"))
        embeddings = sorted(n for n in names if n.startswith("word/embeddings/"))

        revision_counts = {
            "insertions": count(document, ".//w:ins"),
            "deletions": count(document, ".//w:del"),
            "moves_from": count(document, ".//w:moveFrom"),
            "moves_to": count(document, ".//w:moveTo"),
        }
        field_counts = {
            "simple_fields": count(document, ".//w:fldSimple"),
            "field_instructions": count(document, ".//w:instrText"),
        }

        section_count = count(document, ".//w:sectPr")
        paragraph_count = count(document, ".//w:p")
        table_count = count(document, ".//w:tbl")
        hyperlink_count = count(document, ".//w:hyperlink")
        bookmark_start_count = count(document, ".//w:bookmarkStart")

        comment_refs = count(document, ".//w:commentReference")
        footnote_refs = count(document, ".//w:footnoteReference")
        endnote_refs = count(document, ".//w:endnoteReference")

        styles_count = count(styles, ".//w:style")
        abstract_num_count = count(numbering, ".//w:abstractNum")
        num_count = count(numbering, ".//w:num")

        risks = []
        if any(revision_counts.values()):
            risks.append("tracked_changes")
        if comments is not None or comment_refs:
            risks.append("comments")
        if any(field_counts.values()):
            risks.append("fields")
        if table_count:
            risks.append("tables")
        if section_count > 1 or headers or footers:
            risks.append("sections_headers_footers")
        if abstract_num_count or num_count:
            risks.append("numbering")
        if media or embeddings:
            risks.append("media_or_embeddings")

        return {
            "file": str(path),
            "package": {
                "part_count": len(names),
                "has_document_xml": "word/document.xml" in names,
                "has_styles_xml": "word/styles.xml" in names,
                "has_numbering_xml": "word/numbering.xml" in names,
                "has_comments_xml": "word/comments.xml" in names,
                "has_footnotes_xml": "word/footnotes.xml" in names,
                "has_endnotes_xml": "word/endnotes.xml" in names,
                "header_parts": headers,
                "footer_parts": footers,
                "media_count": len(media),
                "embedding_count": len(embeddings),
            },
            "content": {
                "paragraph_count": paragraph_count,
                "table_count": table_count,
                "section_count": section_count,
                "hyperlink_count": hyperlink_count,
                "bookmark_start_count": bookmark_start_count,
                "comment_reference_count": comment_refs,
                "footnote_reference_count": footnote_refs,
                "endnote_reference_count": endnote_refs,
                "sample_visible_text": visible_text(document, text_limit),
            },
            "styles_numbering": {
                "style_count": styles_count,
                "abstract_numbering_count": abstract_num_count,
                "numbering_instance_count": num_count,
            },
            "review_and_fields": {
                **revision_counts,
                **field_counts,
                "comment_count": count(comments, ".//w:comment"),
                "footnote_count": count(footnotes, ".//w:footnote"),
                "endnote_count": count(endnotes, ".//w:endnote"),
            },
            "relationships": rel_targets(rels),
            "risk_flags": risks,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path, help="Path to a .docx file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--text", type=int, default=500, help="Visible text sample length")
    args = parser.parse_args()

    try:
        report = audit_docx(args.docx, args.text)
    except Exception as exc:
        print(f"docx_audit: {exc}", file=sys.stderr)
        return 2

    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
