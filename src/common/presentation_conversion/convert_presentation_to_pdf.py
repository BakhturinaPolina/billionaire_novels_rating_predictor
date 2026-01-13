#!/usr/bin/env python3
"""
Convert PRESENTATION_SLIDES_DRAFT.md to PDF with embedded images.

This script:
1. Reads the markdown file
2. Converts markdown to HTML
3. Embeds images as base64 or file paths
4. Converts HTML to PDF

Requirements:
- markdown (pip install markdown)
- weasyprint (pip install weasyprint) OR
- pdfkit (pip install pdfkit) with wkhtmltopdf installed
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
    # Pattern to match markdown images: ![alt](path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
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
    
    return re.sub(pattern, replace_image, html_content)

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
    
    # Fallback: save as HTML
    html_path = output_path.with_suffix('.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"⚠ PDF libraries not available. HTML saved to: {html_path}")
    print("   Install weasyprint: pip install weasyprint")
    print("   Or install pdfkit: pip install pdfkit (requires wkhtmltopdf)")
    return False

def create_pdf_from_markdown(md_file, output_pdf=None):
    """Main function to convert markdown to PDF."""
    # Use project root as base path for resolving image paths
    base_path = find_project_root(md_file)
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown_to_html(md_content, base_path)
    
    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            border-bottom: 3px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 2px solid #666;
            padding-bottom: 5px;
            margin-top: 30px;
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
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
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
        }}
        hr {{
            border: none;
            border-top: 2px solid #ccc;
            margin: 40px 0;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
    
    # Determine output path
    if output_pdf is None:
        output_pdf = base_path / 'PRESENTATION_SLIDES_DRAFT.pdf'
    else:
        output_pdf = Path(output_pdf)
    
    # Convert to PDF
    html_to_pdf(full_html, output_pdf)
    
    return output_pdf

if __name__ == '__main__':
    import sys
    
    md_file = Path(__file__).parent / 'PRESENTATION_SLIDES_DRAFT.md'
    if len(sys.argv) > 1:
        md_file = Path(sys.argv[1])
    
    output_pdf = None
    if len(sys.argv) > 2:
        output_pdf = sys.argv[2]
    
    if not md_file.exists():
        print(f"Error: Markdown file not found: {md_file}")
        sys.exit(1)
    
    print(f"Converting {md_file} to PDF...")
    result = create_pdf_from_markdown(md_file, output_pdf)
    print(f"Done! Output: {result}")

