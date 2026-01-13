#!/bin/bash
# Convert HTML to PDF using headless Chromium

HTML_FILE="PRESENTATION_SLIDES_DRAFT.html"
PDF_FILE="PRESENTATION_SLIDES_DRAFT.pdf"

# Check if HTML file exists
if [ ! -f "$HTML_FILE" ]; then
    echo "Error: $HTML_FILE not found!"
    exit 1
fi

# Get absolute path
HTML_ABS=$(readlink -f "$HTML_FILE")

# Try chromium-browser first, then chromium, then google-chrome
if command -v chromium-browser &> /dev/null; then
    BROWSER="chromium-browser"
elif command -v chromium &> /dev/null; then
    BROWSER="chromium"
elif command -v google-chrome &> /dev/null; then
    BROWSER="google-chrome"
else
    echo "Error: No Chromium/Chrome browser found!"
    echo "Please use Option 1 from CONVERT_TO_PDF_INSTRUCTIONS.md"
    exit 1
fi

echo "Converting $HTML_FILE to PDF using $BROWSER..."

# Convert HTML to PDF using headless mode
$BROWSER --headless --disable-gpu --print-to-pdf="$PDF_FILE" --print-to-pdf-no-header "file://$HTML_ABS" 2>/dev/null

if [ -f "$PDF_FILE" ]; then
    echo "✓ PDF created successfully: $PDF_FILE"
    ls -lh "$PDF_FILE"
else
    echo "⚠ PDF creation failed. Try opening $HTML_FILE in a browser and printing to PDF."
    echo "Or see CONVERT_TO_PDF_INSTRUCTIONS.md for other options."
fi

