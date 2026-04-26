import os
import markdown
import glob
from xhtml2pdf import pisa

def convert_md_to_pdf(md_file, pdf_file):
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    # Convert specific Markdown tables to HTML
    html_content = markdown.markdown(md_text, extensions=['tables'])
    
    # Add some basic styling
    full_html = f"""
    <html>
    <head>
    <style>
        @page {{
            size: a4 portrait;
            margin: 2cm;
        }}
        body {{
            font-family: Arial, Helvetica, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 10px;
            border: 1px solid #ddd;
            overflow-x: auto;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    with open(pdf_file, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(
            full_html,
            dest=result_file
        )
    
    if pisa_status.err:
        print(f"Error converting {md_file} to PDF.")
    else:
        print(f"Successfully converted {md_file} to {pdf_file}")

# Convert Report
report_md = "report/Assignment3_Report.md"
report_pdf = "report/Assignment3_Report.pdf"
if os.path.exists(report_md):
    convert_md_to_pdf(report_md, report_pdf)

# Convert Logs
log_md = "logs/experiment_logs.md"
log_pdf = "logs/experiment_logs.pdf"
if os.path.exists(log_md):
    convert_md_to_pdf(log_md, log_pdf)
