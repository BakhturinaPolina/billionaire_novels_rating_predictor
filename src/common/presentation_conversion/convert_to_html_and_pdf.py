#!/usr/bin/env python3
"""
Convert PRESENTATION_SLIDES_DRAFT.md to both HTML and PDF with embedded images.

This script:
1. Reads the markdown file
2. Converts markdown to HTML
3. Embeds images as base64
4. Saves HTML file
5. Attempts to convert HTML to PDF
"""

import os
import re
import base64
from pathlib import Path

def find_project_root(start_path):
    """Find the project root by looking for common markers."""
    current = Path(start_path).resolve()
    markers = ['README.md', 'requirements.txt', '.git', 'SCIENTIFIC_README.md']
    
    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent
    
    # Fallback: return the directory containing the script
    return Path(__file__).parent.parent.parent.parent

def embed_images_in_html(html_content, base_path):
    """Convert image references to embedded base64 or absolute paths."""
    def process_image_path(img_path, alt_text=""):
        """Process a single image path and return embedded HTML."""
        # Convert relative path to absolute
        if not os.path.isabs(img_path):
            full_path = base_path / img_path
        else:
            full_path = Path(img_path)
        
        # Check if file exists
        if full_path.exists():
            # Try to embed as base64
            try:
                with open(full_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    ext = full_path.suffix.lower()
                    if ext == '.png':
                        mime = 'image/png'
                    elif ext in ['.jpg', '.jpeg']:
                        mime = 'image/jpeg'
                    else:
                        mime = 'image/png'
                    
                    return f'<img src="data:{mime};base64,{img_base64}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
            except Exception as e:
                print(f"Warning: Could not embed {full_path} as base64: {e}")
                # Fall back to file path
                return f'<img src="{full_path.as_uri()}" alt="{alt_text}" style="max-width: 100%; height: auto;" />'
        else:
            print(f"Warning: Image not found: {full_path}")
            return f'<p>[Image not found: {img_path}]</p>'
    
    # Pattern 1: Match markdown images: ![alt](path)
    markdown_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_markdown_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        return process_image_path(img_path, alt_text)
    
    html_content = re.sub(markdown_pattern, replace_markdown_image, html_content)
    
    # Pattern 2: Match HTML img tags (more flexible - handles any attribute order)
    # First extract alt if present
    def extract_alt_and_replace(match):
        full_tag = match.group(0)
        # Extract src
        src_match = re.search(r'src=["\']([^"\']+)["\']', full_tag)
        if not src_match:
            return full_tag
        img_path = src_match.group(1)
        
        # Extract alt if present
        alt_match = re.search(r'alt=["\']([^"\']*)["\']', full_tag)
        alt_text = alt_match.group(1) if alt_match else ""
        
        # Only process if it's not already a data URI
        if not img_path.startswith('data:'):
            return process_image_path(img_path, alt_text)
        return full_tag  # Keep original if already embedded
    
    # Match any img tag
    html_pattern = r'<img[^>]+>'
    html_content = re.sub(html_pattern, extract_alt_and_replace, html_content)
    
    return html_content

def markdown_to_html(md_content, base_path):
    """Convert markdown to HTML."""
    try:
        import markdown
        html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
    except ImportError:
        print("Warning: markdown library not found. Using basic conversion.")
        # Basic fallback
        html = md_content.replace('\n', '<br>\n')
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
    
    # Embed images
    html = embed_images_in_html(html, base_path)
    
    return html

def create_html_document(html_content):
    """Create full HTML document with styling."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Presentation Slides: Modern Romantic Novels — Themes × Popularity</title>
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            color: #333;
        }}
        h1 {{
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            border-bottom: 2px solid #666;
            padding-bottom: 5px;
            margin-top: 40px;
            page-break-before: always;
        }}
        h2:first-of-type {{
            page-break-before: auto;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 3px solid #666;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ccc;
            margin: 40px 0;
        }}
        @media print {{
            h2 {{
                page-break-before: always;
            }}
            img {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

def html_to_pdf(html_content, output_path):
    """Convert HTML to PDF."""
    # Try weasyprint first
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(output_path)
        print(f"✓ PDF created using weasyprint: {output_path}")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: weasyprint failed: {e}")
        pass
    
    # Try pdfkit
    try:
        import pdfkit
        pdfkit.from_string(html_content, output_path)
        print(f"✓ PDF created using pdfkit: {output_path}")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: pdfkit failed: {e}")
        pass
    
    return False

def convert_markdown_to_html_and_pdf(md_file, output_html=None, output_pdf=None):
    """Main function to convert markdown to HTML and PDF."""
    # Use project root as base path for resolving image paths
    base_path = find_project_root(md_file)
    
    # Read markdown
    print(f"Reading markdown file: {md_file}")
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    print("Converting markdown to HTML...")
    html_content = markdown_to_html(md_content, base_path)
    
    # Create full HTML document
    full_html = create_html_document(html_content)
    
    # Determine output paths
    if output_html is None:
        output_html = base_path / 'PRESENTATION_SLIDES_DRAFT.html'
    else:
        output_html = Path(output_html)
    
    if output_pdf is None:
        output_pdf = base_path / 'PRESENTATION_SLIDES_DRAFT.pdf'
    else:
        output_pdf = Path(output_pdf)
    
    # Save HTML
    print(f"Saving HTML to: {output_html}")
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"✓ HTML saved successfully: {output_html}")
    file_size = output_html.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")
    
    # Try to convert to PDF
    print(f"\nAttempting to convert HTML to PDF: {output_pdf}")
    pdf_success = html_to_pdf(full_html, output_pdf)
    
    if not pdf_success:
        print("\n⚠ PDF conversion failed. Available options:")
        print("  1. Install weasyprint: pip install weasyprint")
        print("  2. Install pdfkit: pip install pdfkit (requires wkhtmltopdf)")
        print("  3. Use the HTML file with convert_html_to_pdf_chromium.sh")
        print("  4. Open HTML in browser and print to PDF (Ctrl+P)")
    else:
        pdf_size = output_pdf.stat().st_size / (1024 * 1024)
        print(f"  PDF file size: {pdf_size:.2f} MB")
    
    return output_html, output_pdf if pdf_success else None

if __name__ == '__main__':
    import sys
    
    md_file = Path(__file__).parent / 'PRESENTATION_SLIDES_DRAFT.md'
    if len(sys.argv) > 1:
        md_file = Path(sys.argv[1])
    
    output_html = None
    if len(sys.argv) > 2:
        output_html = sys.argv[2]
    
    output_pdf = None
    if len(sys.argv) > 3:
        output_pdf = sys.argv[3]
    
    if not md_file.exists():
        print(f"Error: Markdown file not found: {md_file}")
        sys.exit(1)
    
    html_path, pdf_path = convert_markdown_to_html_and_pdf(md_file, output_html, output_pdf)
    print(f"\n✓ Conversion complete!")
    print(f"  HTML: {html_path}")
    if pdf_path:
        print(f"  PDF: {pdf_path}")

