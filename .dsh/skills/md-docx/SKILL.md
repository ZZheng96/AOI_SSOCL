---
name: "md-to-docx"
description: "Converts Markdown files to formal Word documents (.docx). Invoke when user uploads or provides Markdown content and requests 'convert to Word', 'export document', 'make Word', 'generate docx', etc. Always use this skill for Markdown → Word conversion, even if user simply says 'help me make a Word document'."
tags: ["document", "conversion", "word", "markdown"]
inputs: ["markdown_file", "markdown_text"]
outputs: ["docx_file"]
---

# Markdown -> Word Document Conversion

## Overview

Use the `docx` npm package to convert structured Markdown content into formatted professional Word documents.
All text uses SimSun (宋体) font, primarily black color scheme, no colors.

---

## Execution Order

| Step | Operation |
|------|-----------|
| 1. Content Preprocessing | Read Markdown, perform encoding and symbol correction (see below) |
| 2. Structure Analysis | Identify heading levels, lists, tables, paragraphs |
| 3. Generate Script | Write Node.js script according to this skill's template |
| 4. Execute Output | Run script, output to user-specified path or current directory |

---

## Step 1: Encoding and Symbol Preprocessing

**Before generating any script, the following checks and corrections must be completed.**

### 1.1 Garbled Text Detection

If the following characteristics appear in the text, it indicates encoding errors and the file needs to be re-read with correct encoding:

- Latin letter mixed sequences like a, E, (tm), i3 appear -> UTF-8 incorrectly parsed as Latin-1
- Large number of ? replacing Chinese characters -> encoding loss
- Hexadecimal escapes like \x84, \x92 appear -> byte stream not properly decoded

Correction methods (bash):

```bash
# Detect file encoding
file -i original_file.md
python3 -c "import chardet; print(chardet.detect(open('original_file.md','rb').read()))"

# Re-read with correct encoding and save as UTF-8 (example: GBK -> UTF-8)
python3 -c "open('out.md','w',encoding='utf-8').write(open('original_file.md',encoding='gbk').read())"
```

### 1.2 Symbol Unification Correction

Before writing text content into JS scripts, use the following Python function for normalization.
Content inside code blocks (areas surrounded by three backticks) is skipped.

```python
import re

def clean_text(text):
    # Protect code blocks, skip replacement
    code_blocks = {}
    def protect(m):
        key = f'__CODE_{len(code_blocks)}__'
        code_blocks[key] = m.group(0)
        return key
    text = re.sub(r'```[\s\S]*?```', protect, text)

    # Paired straight quotes -> Chinese curved quotes
    text = re.sub(r'"([^"\n]{0,80})"', '\u201c\\1\u201d', text)
    text = re.sub(r"'([^'\n]{0,80})'", '\u2018\\1\u2019', text)

    # Dash normalization
    text = text.replace('--', '\u2014')
    text = re.sub(r'(?<=[\u4e00-\u9fff]) - (?=[\u4e00-\u9fff])', '\u2014', text)

    # Ellipsis normalization
    text = text.replace('......', '\u2026\u2026')
    text = text.replace('...', '\u2026')

    # HTML entity restoration
    entities = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&nbsp;': ' ', '&#x2019;': '\u2019',
        '&#x201c;': '\u201c', '&#x201d;': '\u201d',
        '&ldquo;': '\u201c', '&rdquo;': '\u201d',
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)

    # Restore code blocks
    for key, val in code_blocks.items():
        text = text.replace(key, val)

    return text
```

### 1.3 Residual Markdown Syntax Processing

Check and handle unconverted raw Markdown syntax:

| Residual Syntax | Handling Method |
|-----------------|-----------------|
| `**text**` | Extract text, set TextRun `bold: true` |
| `*text*` | Extract text, set TextRun `italics: true` |
| `` `code` `` | Extract text, keep as is |
| `[text](link)` | Keep text only |
| `---` separator | Convert to empty paragraph with bottom border, or ignore |
| `> quote` | Remove `> `, render with italic gray paragraph |
| ```code block``` | Render with Courier New font, light gray background, preserve line breaks |

---

## Document Style Specifications

### Page Setup (A4)

```javascript
properties: {
  page: {
    size: { width: 11906, height: 16838 },
    margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
  }
}
```

Content width = 11906 - 1440 - 1800 = **8666 DXA**

### Color Scheme (Black/Gray)

| Element | HEX | Description |
|---------|-----|-------------|
| Headings, body text | `000000` | Pure black |
| Italic notes | `555555` | Dark gray |
| Header/footer text | `888888` | Medium gray |
| Separators, table borders | `AAAAAA` | Light gray |
| Table left column (dark) | `F2F2F2` | Very light gray, black text |
| Table left column (light alternate) | `FAFAFA` | Near white, black text |

### Font

**All text unified in SimSun (宋体)**, use Unicode escape `\u5b8b\u4f53` in scripts to avoid encoding issues.

| Element | size | Point Size |
|---------|------|------------|
| Cover main title | 56 | 28pt |
| H1 | 32 | 16pt |
| H2 | 28 | 14pt |
| H3 | 24 | 12pt |
| Body text | 22 | 11pt |
| Header/footer | 18 | 9pt |

---

## Script Template

### Basic Imports and Utility Functions

```javascript
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, Header, Footer, PageNumber
} = require('docx');
const fs = require('fs');

const CW = 8666;
const bd = { style: BorderStyle.SINGLE, size: 1, color: 'AAAAAA' };
const bAll = { top: bd, bottom: bd, left: bd, right: bd };
const q = (s) => '\u201c' + s + '\u201d';

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, bold: true, size: 32, font: '\u5b8b\u4f53', color: '000000' })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, bold: true, size: 28, font: '\u5b8b\u4f53', color: '000000' })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, bold: true, size: 24, font: '\u5b8b\u4f53', color: '000000' })]
  });
}
function p(text, opts) {
  opts = opts || {};
  return new Paragraph({
    spacing: { before: 60, after: 60, line: 360 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun(Object.assign({ text, size: 22, font: '\u5b8b\u4f53', color: '000000' }, opts.run || {}))]
  });
}
function bl(text, level) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: level || 0 },
    spacing: { before: 40, after: 40, line: 340 },
    children: [new TextRun({ text, size: 22, font: '\u5b8b\u4f53', color: '000000' })]
  });
}
function empty() {
  return new Paragraph({ spacing: { before: 60, after: 60 }, children: [] });
}
function codeBlock(lines) {
  return lines.map(function(line) {
    return new Paragraph({
      spacing: { before: 20, after: 20, line: 280 },
      shading: { fill: 'F5F5F5', type: ShadingType.CLEAR },
      border: { left: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' } },
      indent: { left: 200 },
      children: [new TextRun({ text: line, size: 20, font: 'Courier New', color: '333333' })]
    });
  });
}
```

### numbering Configuration

```javascript
numbering: {
  config: [
    { reference: 'bullets', levels: [
      { level: 0, format: LevelFormat.BULLET, text: '\u25CF', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: LevelFormat.BULLET, text: '\u25CB', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }
    ]}
  ]
}
```

### styles Configuration

```javascript
styles: {
  default: { document: { run: { font: '\u5b8b\u4f53', size: 22, color: '000000' } } },
  paragraphStyles: [
    { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
      run: { size: 32, bold: true, font: '\u5b8b\u4f53', color: '000000' },
      paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
    { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
      run: { size: 28, bold: true, font: '\u5b8b\u4f53', color: '000000' },
      paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
    { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
      run: { size: 24, bold: true, font: '\u5b8b\u4f53', color: '000000' },
      paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
  ]
}
```

### Header/Footer

```javascript
headers: {
  default: new Header({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: 'AAAAAA' } },
      spacing: { before: 0, after: 120 },
      children: [new TextRun({ text: '\u6587\u6863\u6807\u9898', size: 18, font: '\u5b8b\u4f53', color: '888888' })]
    })]
  })
},
footers: {
  default: new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { top: { style: BorderStyle.SINGLE, size: 2, color: 'AAAAAA' } },
      spacing: { before: 120, after: 0 },
      children: [
        new TextRun({ text: '\u7b2c ', size: 18, font: '\u5b8b\u4f53', color: '888888' }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '888888' }),
        new TextRun({ text: ' \u9875', size: 18, font: '\u5b8b\u4f53', color: '888888' })
      ]
    })]
  })
}
```

### Info Table (Very light gray left column, black text, suitable for basic information listing)

```javascript
function infoTable(rows) {
  return new Table({
    width: { size: CW, type: WidthType.DXA },
    columnWidths: [1800, CW - 1800],
    rows: rows.map(function(r, i) {
      var fill = (i % 2 === 0) ? 'F2F2F2' : 'FAFAFA';
      return new TableRow({ children: [
        new TableCell({
          borders: bAll, width: { size: 1800, type: WidthType.DXA },
          shading: { fill: fill, type: ShadingType.CLEAR },
          margins: { top: 100, bottom: 100, left: 150, right: 150 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: r.label, bold: true, size: 22, font: '\u5b8b\u4f53', color: '000000' })] })]
        }),
        new TableCell({
          borders: bAll, width: { size: CW - 1800, type: WidthType.DXA },
          margins: { top: 100, bottom: 100, left: 150, right: 150 },
          children: [new Paragraph({ children: [new TextRun({ text: r.value, size: 22, font: '\u5b8b\u4f53', color: '000000' })] })]
        })
      ]});
    })
  });
}
```

### Feature Table (Very light gray left column, black text, suitable for feature/function description)

```javascript
function featureTable(rows) {
  return new Table({
    width: { size: CW, type: WidthType.DXA },
    columnWidths: [1800, CW - 1800],
    rows: rows.map(function(r, i) {
      var fill = (i % 2 === 0) ? 'F2F2F2' : 'FAFAFA';
      return new TableRow({ children: [
        new TableCell({
          borders: bAll, width: { size: 1800, type: WidthType.DXA },
          shading: { fill: fill, type: ShadingType.CLEAR },
          margins: { top: 100, bottom: 100, left: 150, right: 150 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: r.label, bold: true, size: 22, font: '\u5b8b\u4f53', color: '000000' })] })]
        }),
        new TableCell({
          borders: bAll, width: { size: CW - 1800, type: WidthType.DXA },
          margins: { top: 100, bottom: 100, left: 150, right: 150 },
          children: [new Paragraph({ children: [new TextRun({ text: r.value, size: 22, font: '\u5b8b\u4f53', color: '000000' })] })]
        })
      ]});
    })
  });
}
```

### Generic Table (Multi-column, supports custom headers and rows)

```javascript
function createTable(headers, rows) {
  var colCount = headers.length;
  var colWidth = Math.floor(CW / colCount);
  var columnWidths = headers.map(function() { return colWidth; });
  
  return new Table({
    width: { size: CW, type: WidthType.DXA },
    columnWidths: columnWidths,
    rows: [
      new TableRow({ children: headers.map(function(h) {
        return new TableCell({
          borders: bAll, width: { size: colWidth, type: WidthType.DXA },
          shading: { fill: 'E8E8E8', type: ShadingType.CLEAR },
          margins: { top: 100, bottom: 100, left: 150, right: 150 },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: h, bold: true, size: 22, font: '\u5b8b\u4f53', color: '000000' })] })]
        });
      })}),
      rows.map(function(row, i) {
        var fill = (i % 2 === 0) ? 'F2F2F2' : 'FAFAFA';
        return new TableRow({ children: row.map(function(cell) {
          return new TableCell({
            borders: bAll, width: { size: colWidth, type: WidthType.DXA },
            shading: { fill: fill, type: ShadingType.CLEAR },
            margins: { top: 100, bottom: 100, left: 150, right: 150 },
            children: [new Paragraph({ children: [new TextRun({ text: cell, size: 22, font: '\u5b8b\u4f53', color: '000000' })] })]
          });
        })});
      })
    ].flat()
  });
}
```

### Cover Title Block

```javascript
new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 800, after: 160 },
  children: [new TextRun({ text: '\u4e3b\u6807\u9898', size: 56, bold: true, font: '\u5b8b\u4f53', color: '000000' })]
}),
new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 160 },
  children: [new TextRun({ text: '\u526f\u6807\u9898', size: 56, bold: true, font: '\u5b8b\u4f53', color: '000000' })]
}),
new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 0, after: 600 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '333333' } },
  children: [new TextRun({ text: '\u673a\u6784\u540d\u79f0', size: 24, font: '\u5b8b\u4f53', color: '333333' })]
}),
```

### Output File Naming

Generate filename based on context:
- If converting a file: use original filename with `.docx` extension
- If from text content: derive from H1 heading or ask user
- Sanitize filename: remove `\\ / : * ? " < > |`

```javascript
function sanitizeFilename(name) {
  return name.replace(/[\\/:*?"<>|]/g, '_').trim() || 'document';
}

// Examples:
// Input: "report.md" -> Output: "report.docx"
// Input: "项目总结" -> Output: "项目总结.docx"
// Input: "file:name?" -> Output: "file_name_.docx"
```

### Script Finalization

```javascript
var outputPath = process.argv[2] || './output.docx';
Packer.toBuffer(doc).then(function(buf) {
  fs.writeFileSync(outputPath, buf);
  console.log('Saved to: ' + outputPath);
}).catch(function(e) { console.error(e); process.exit(1); });
```

---

## Key Considerations

### Chinese Quote Nesting (Most Common Error)

```javascript
// Wrong: double quote nesting
p("项目以"三位一体"模式")

// Correct: Unicode escape
p('\u9879\u76ee\u4ee5\u201c\u4e09\u4f4d\u4e00\u4f53\u201d\u6a21\u5f0f')

// Correct: q() concatenation
p('\u9879\u76ee\u4ee5' + q('\u4e09\u4f4d\u4e00\u4f53') + '\u6a21\u5f0f')
```

Common special characters:

| Character | Unicode |
|-----------|---------|
| `"` | `\u201c` |
| `"` | `\u201d` |
| `'` | `\u2018` |
| `'` | `\u2019` |
| `—` | `\u2014` |
| `…` | `\u2026` |
| `宋体` | `\u5b8b\u4f53` |

### Chinese Content Encoding

All Chinese text in scripts uses Unicode escape. Bash command to generate escapes:

```bash
python3 -c "
text = '需要转义的文字'
print(''.join(f'\\\\u{ord(c):04x}' if ord(c) > 127 else c for c in text))
"
```

### Other Common Errors

```javascript
// Wrong: hardcoded bullet character
new Paragraph({ children: [new TextRun('- List item')] })
// Correct: use numbering
new Paragraph({ numbering: { reference: 'bullets', level: 0 }, ... })

// Wrong: table missing cell width, or using PERCENTAGE
new TableCell({ ... })
// Correct: set DXA width for both table and TableCell
new TableCell({ width: { size: 1800, type: WidthType.DXA }, ... })

// Wrong: ShadingType.SOLID turns black
shading: { fill: 'EEEEEE', type: ShadingType.SOLID }
// Correct
shading: { fill: 'EEEEEE', type: ShadingType.CLEAR }
```

---

## Markdown Structure Mapping

| Markdown Element | Word Mapping |
|------------------|--------------|
| `# Heading` | `h1()` |
| `## Heading` | `h2()` |
| `### Heading` | `h3()` |
| Body paragraph | `p()` |
| `- List item` | `bl(text, 0)` |
| `  - Sublist` | `bl(text, 1)` |
| `> Quote` | `p(text, { run: { italics: true, color: '555555' } })` |
| `**text**` | `TextRun({ text, bold: true })` |
| Key-value info block | `infoTable([{ label, value }])` |
| Feature/function comparison | `featureTable([{ label, value }])` |
| Multi-column table | `createTable(headers, rows)` |
| Code block | `codeBlock(lines)` |
| Empty line | `empty()` |

---

## Error Handling

| Error Scenario | Cause | Solution |
|----------------|-------|----------|
| `Error: Cannot find module 'docx'` | docx package not installed | Run `npm install docx` or `npm install -g docx` |
| `TypeError: Cannot read property '...' of undefined` | Incorrect docx API usage | Check import names and object structure against template |
| Garbled Chinese characters in output | Encoding issue | Re-read source file as UTF-8 (see 1.1) |
| `Error: EACCES: permission denied` | Insufficient write permissions | Check output directory permissions or choose different path |
| Table layout broken | Missing width or wrong width type | Ensure all TableCell have `width: { size: N, type: WidthType.DXA }` |
| Bullet points not rendering | Numbering config missing | Include numbering config in Document constructor |

## Execution Command

```bash
# Check dependencies
node -v
npm list docx

# Install if missing
npm install docx

# Run with default output path
node gen_doc.js

# Run with custom output path
node gen_doc.js ./my-document.docx
```
