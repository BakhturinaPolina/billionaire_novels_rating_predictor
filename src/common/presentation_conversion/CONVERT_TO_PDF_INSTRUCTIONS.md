# Instructions for Converting PRESENTATION_SLIDES_DRAFT.md to PDF

An HTML version with embedded images has been created: `PRESENTATION_SLIDES_DRAFT.html`

## Option 1: Using Browser (Recommended - Easiest)

1. Open `PRESENTATION_SLIDES_DRAFT.html` in your web browser
2. Press `Ctrl+P` (or `Cmd+P` on Mac) to print
3. Select "Save as PDF" as the destination
4. Click "Save"

This will create a PDF with all images properly embedded.

## Option 2: Using Python Libraries

If you have a virtual environment or can install packages:

```bash
# Install required packages
pip install markdown weasyprint

# Or use pdfkit (requires wkhtmltopdf)
pip install markdown pdfkit
# sudo apt-get install wkhtmltopdf  # on Ubuntu/Debian

# Run the conversion script
python3 convert_presentation_to_pdf.py PRESENTATION_SLIDES_DRAFT.md PRESENTATION_SLIDES_DRAFT.pdf
```

## Option 3: Using Pandoc (if available)

```bash
# Install pandoc (if not available)
# sudo apt-get install pandoc  # on Ubuntu/Debian

# Convert markdown to PDF (images need to be accessible)
pandoc PRESENTATION_SLIDES_DRAFT.md -o PRESENTATION_SLIDES_DRAFT.pdf --pdf-engine=wkhtmltopdf
```

## Option 4: Using Online Converters

1. Upload `PRESENTATION_SLIDES_DRAFT.html` to an online HTML-to-PDF converter
2. Download the resulting PDF

## Current Status

- ✅ Markdown file updated with examples and visualizations
- ✅ All plot references embedded as image paths
- ✅ HTML version created with embedded images (4.8MB)
- ⏳ PDF conversion requires one of the methods above

The HTML file (`PRESENTATION_SLIDES_DRAFT.html`) contains all images embedded as base64, so it's self-contained and can be opened in any browser.

