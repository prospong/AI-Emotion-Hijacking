import os, re, sys, base64, unicodedata, html, argparse
from pathlib import Path
import nbformat

def nearest_h1_title(nb, idx):
    for back in range(1, 8):
        j = idx - back
        if j >= 0 and nb.cells[j].cell_type == "markdown":
            md = nb.cells[j].source or ""
            m = re.search(r"^\s*#\s+(.+)$", md, flags=re.M)
            if m:
                return m.group(1)
    return ""

def img_data_uri(mime, payload):
    if mime == "image/svg+xml":
        if isinstance(payload, str) and not re.match(r"^[A-Za-z0-9+/=\s]+$", payload):
            svg = payload.replace("</script>", "<\\/script>")
            return "data:image/svg+xml;utf8," + svg
        else:
            if isinstance(payload, list): payload = "".join(payload)
            b = base64.b64decode(payload)
            b64 = base64.b64encode(b).decode("ascii")
            return f"data:{mime};base64,{b64}"
    else:
        if isinstance(payload, list): payload = "".join(payload)
        b = base64.b64decode(payload)
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"

def build(nb_path: Path, out_path: Path, verbose: bool=False):
    if not nb_path.exists():
        print(f"[ERROR] Notebook not found: {nb_path.resolve()}", file=sys.stderr)
        return 2

    if verbose:
        print(f"[INFO] Reading notebook: {nb_path}")

    nb = nbformat.read(str(nb_path), as_version=4)

    cards = []
    fig_count = 0
    code_cells = 0
    image_bundles = 0

    for ci, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code_cells += 1
        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        # 收集此 cell 的文字/表格输出
        textual_blocks = []

        def add_text_block(title, text):
            if not text: return
            esc = html.escape(text)
            textual_blocks.append(
                f"<div class='block'><div class='block-title'>{html.escape(title)}</div><pre>{esc}</pre></div>"
            )

        def add_html_block(title, html_content):
            if not html_content: return
            textual_blocks.append(
                f"<div class='block'><div class='block-title'>{html.escape(title)}</div><div class='htmlwrap'>{html_content}</div></div>"
            )

        # stream: stdout/stderr
        for out in outputs:
            if isinstance(out, dict) and out.get("output_type") == "stream":
                name = out.get("name", "stream")
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                add_text_block(f"stream:{name}", text)

        title_hint = nearest_h1_title(nb, ci)

        for out in outputs:
            if not isinstance(out, dict):
                continue
            data = out.get("data", {})
            if not isinstance(data, dict):
                continue

            # 文本块
            if "text/plain" in data:
                payload = data["text/plain"]
                if isinstance(payload, list): payload = "".join(payload)
                add_text_block("text/plain", str(payload))

            if "text/html" in data:
                payload = data["text/html"]
                if isinstance(payload, list): payload = "".join(payload)
                add_html_block("text/html", payload)

            # 图像
            found_img = False
            for mime in ("image/png", "image/jpeg", "image/svg+xml"):
                if mime in data:
                    try:
                        uri = img_data_uri(mime, data[mime])
                    except Exception as e:
                        add_text_block("error", f"Failed to decode image: {e}")
                        continue
                    fig_count += 1
                    image_bundles += 1
                    found_img = True
                    caption = f"Figure {fig_count:02d}"
                    if title_hint:
                        caption += f" — {html.escape(title_hint)}"
                    card_html = f"""
                    <div class="card">
                      <img src="{uri}" alt="Figure {fig_count:02d}">
                      <div class="cap">{caption}</div>
                      <div class="data">
                        {''.join(textual_blocks) if textual_blocks else "<div class='nodata'>No captured run-time text output for this figure.</div>"}
                      </div>
                    </div>
                    """
                    cards.append(card_html)
            if verbose and found_img:
                print(f"[INFO] Cell {ci} contributed image(s).")

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Emotion Hijacking – Figures + Run Data</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#fafafa}}
h1{{margin-bottom:8px}} .meta{{color:#666;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,0.04);background:#fff}}
.cap{{font-size:13px;color:#333;margin:8px 0 12px}}
.data{{border-top:1px solid #eee;padding-top:10px}}
.block{{margin-bottom:10px}}
.block-title{{font-size:12px;color:#555;margin-bottom:4px}}
pre{{white-space:pre-wrap;word-wrap:break-word;background:#f8fafc;border:1px solid #eef2f7;border-radius:8px;padding:8px;max-height:320px;overflow:auto}}
.htmlwrap{{border:1px solid #eef2f7;border-radius:8px;padding:6px;max-height:360px;overflow:auto;background:#fff}}
.nodata{{font-size:12px;color:#888}}
</style>
</head>
<body>
<h1>AI Emotion Hijacking – Figures + Run Data</h1>
<div class="meta">Source: {html.escape(str(nb_path))} • Code cells: {code_cells} • Output bundles with images: {image_bundles} • Total figures: <b>{fig_count}</b></div>
<div class="grid">
{''.join(cards) if cards else "<p><em>No figures found in the notebook outputs.</em></p>"}
</div>
</body>
</html>"""

    out_path.write_text(html_doc, encoding="utf-8")
    if verbose:
        print(f"[OK] Wrote {out_path} with {fig_count} figure(s).")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", "-n", required=True, help="Path to .ipynb with saved outputs")
    parser.add_argument("--out", "-o", default="AI-Emotion-Hijacking_InlineGallery_WithData.html", help="Output HTML path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    out_path = Path(args.out)
    sys.exit(build(nb_path, out_path, verbose=args.verbose))

if __name__ == "__main__":
    main()
