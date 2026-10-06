---
name: word-docx
description: Create, inspect, edit, repair, and verify Microsoft Word .docx documents. Use when the user asks Codex to work with Word files, DOCX packages, templates, styles, numbering, tracked changes, comments, headers, footers, tables, fields, cross-references, page layout, or compatibility with Word, Google Docs, or LibreOffice.
---

# Word / DOCX

## Purpose

Work with `.docx` files as structured OOXML packages, not as plain text. Preserve the user's formatting, review metadata, numbering, references, and layout unless the user explicitly asks to simplify or rebuild the document.

## First Move

1. Identify the job type: read/extract, create, edit, repair, convert, or verify.
2. For an existing `.docx`, inspect the package before making risky edits.
3. Prefer the smallest structural change that satisfies the request.
4. Verify the final document by checking both text/content and OOXML risk areas.

Run the bundled audit script when an existing document might contain fragile structures:

```bash
python scripts/docx_audit.py path/to/file.docx --pretty
```

Use the audit output to decide whether to inspect `word/document.xml`, `word/styles.xml`, `word/numbering.xml`, headers, footers, relationships, comments, footnotes, endnotes, or media parts.

## Workflow

### Read or Extract

- Use a structure-preserving reader before touching XML.
- Remember that visible phrases may be split across runs, bookmarks, fields, hyperlinks, comments, or revision tags.
- If tracked changes are present, distinguish current visible text from inserted, deleted, and moved text.
- For summaries or content review, mention whether comments, unresolved revisions, or stale fields were present.

### Create

- Use named styles for headings, body text, captions, lists, and tables.
- Set page size, margins, and section properties explicitly.
- Use real Word numbering definitions for bullets and numbered lists; do not fake lists with Unicode bullets or typed numbers.
- Use table widths that are compatible with the page width and margins.
- Avoid empty paragraphs for spacing; set paragraph spacing instead.

### Edit

- Preserve the current style system. Extend existing styles only when needed.
- Make local span-level edits for review documents instead of rewriting whole paragraphs.
- Avoid rewriting paragraphs that contain comments, bookmarks, fields, hyperlinks, footnote anchors, or tracked changes unless the request requires it.
- Treat lists through `numbering.xml` and paragraph numbering properties, not only indentation.
- Treat layout through sections, margins, page size, headers, and footers, not manual spacing.
- When editing headers, footers, images, hyperlinks, or embedded objects, also update the corresponding relationship files.

### Repair

- Inspect broken relationships, missing media, duplicated style names, bad numbering restarts, orphaned comments, and malformed field anchors.
- Fix the underlying XML state rather than only matching the rendered appearance.
- If converting from legacy `.doc`, perform conversion first and then audit the resulting `.docx`.
- Treat `.docm` as macro-bearing and higher risk; do not remove or rewrite macro-related parts unless requested.

### Verify

- Re-open or re-parse the final `.docx` package after editing.
- Confirm requested text/content changes.
- Check for unresolved revisions, comments, stale fields, broken relationships, changed section properties, and numbering drift.
- For layout-sensitive documents, inspect tables, headers/footers, page size, margins, and section breaks.
- If compatibility matters, note residual risk across Microsoft Word, Google Docs, and LibreOffice.

## High-Risk Structures

- `word/document.xml`: main body, runs, paragraphs, tables, fields, revision tags.
- `word/styles.xml`: paragraph, character, table, and numbering styles.
- `word/numbering.xml`: abstract numbering, list instances, restarts, indentation.
- `word/_rels/document.xml.rels`: hyperlinks, images, headers, footers, footnotes, comments.
- `word/comments.xml`: comment bodies and anchors.
- `word/footnotes.xml` and `word/endnotes.xml`: note content and references.
- `word/header*.xml` and `word/footer*.xml`: section-linked header/footer content.
- `word/media/*`: images and embedded media.
- `docProps/*`: document metadata.

## Common Traps

- A `.docx` is a ZIP of XML parts; changing only visible text can corrupt hidden structure.
- A single word can be split across multiple runs.
- Deleted text may remain in the package when tracked changes are enabled.
- Field display text can be stale until the recipient updates fields.
- Header/footer images use part-specific relationship IDs.
- Copying between documents can import unwanted styles, themes, and numbering definitions.
- Tables that look acceptable in Word can drift in Google Docs or LibreOffice when widths are implicit.
- Page size defaults differ between A4 and US Letter and can change pagination.

## Delivery Notes

- State what was changed and what was verified.
- If any review metadata, comments, fields, macros, or layout-sensitive elements remain, call them out briefly.
- Keep generated helper artifacts out of the user's final directory unless they are requested deliverables.
