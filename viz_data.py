import json
import html  # For escaping text in HTML

def visualize_receipt_data(input_file: str = "receipt_dataset.jsonl", output_html: str = "receipt_visualization.html", num_entries: int = None):
    """
    Reads a JSONL file with receipt data and generates an HTML file for easy manual inspection.
    Each entry shows:
    - Left: Raw receipt text (preformatted for readability)
    - Right: Pretty-printed JSON output
    - Below: The instruction (for context)
    
    Args:
    - input_file: Path to the JSONL file.
    - output_html: Path to save the HTML output.
    - num_entries: Optional limit on number of entries to visualize (e.g., 100 for testing).
    """
    with open(input_file, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]
    
    if num_entries:
        entries = entries[:num_entries]
    
    html_content = f"""
    <html>
    <head>
        <title>Receipt Data Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .entry {{ border: 1px solid #ccc; margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
            .header {{ font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .left {{ flex: 1; min-width: 45%; background-color: #f9f9f9; padding: 10px; border-right: 1px solid #ddd; }}
            .right {{ flex: 1; min-width: 45%; padding: 10px; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; }}
            .instruction {{ margin-top: 10px; font-style: italic; color: #555; }}
            .config-btn {{ background-color: #e3f2fd; border: 1px solid #90caf9; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 0.9em; margin-top: 8px; }}
            .config-btn:hover {{ background-color: #bbdefb; }}
            .config-content {{ display: none; background-color: #f5f5f5; border: 1px solid #ddd; padding: 10px; margin-top: 8px; border-radius: 4px; }}
            .config-content.open {{ display: block; }}
        </style>
        <script>
            function toggleConfig(id) {{
                var el = document.getElementById(id);
                el.classList.toggle('open');
            }}
        </script>
    </head>
    <body>
        <h1>Receipt Data Inspection</h1>
        <p>Showing {len(entries)} entries for manual label checking.</p>
    """
    # Then keep adding the loop content with += f"...{variable}..."
        
    for idx, entry in enumerate(entries, 1):
        raw_text = html.escape(entry.get("input", "No input"))
        json_output = json.dumps(json.loads(entry.get("output", "{}")), indent=4, ensure_ascii=False)
        instruction = html.escape(entry.get("instruction", "No instruction"))
        
        html_content += f"""
        <div class="entry">
            <div class="header">Entry {idx}</div>
            <div class="container">
                <div class="left">
                    <h3>Raw Receipt Text:</h3>
                    <pre>{raw_text}</pre>
                </div>
                <div class="right">
                    <h3>Extracted JSON:</h3>
                    <pre><code>{html.escape(json_output)}</code></pre>
                </div>
            </div>
            <button class="config-btn" onclick="toggleConfig('config-{idx}')">Show Instruction / Config</button>
            <div id="config-{idx}" class="config-content">
                <pre>{instruction}</pre>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML visualization saved to {output_html}. Open in your browser to check labels!")

# Example usage: Run this after generating your dataset
if __name__ == "__main__":
    visualize_receipt_data(num_entries=50)  # Limit to first 50 for quick check; remove to do all