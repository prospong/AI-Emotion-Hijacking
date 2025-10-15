# make_gallery_with_data_v2_hscroll_column_summary.py
# Based on your v2_hscroll:
# - Keep ONE COLUMN (a single series/flow).
# - In each pane: figure on TOP with a horizontal scrollbar; textual results BELOW.
# - At the very BOTTOM of the column, add a "Conclusion & Comparison" card that
#   aggregates metrics extracted from all panes and writes a brief analysis.

import os, re, sys, base64, html, argparse
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

# -------- metrics extraction (from pane text/html) --------
METRIC_PATTERNS = {
    "hijack_rate": [
        re.compile(r"(hijack(ing)?\s*rate|hijack\s*%)\D+?(\d+(\.\d+)?)\s*%", re.I),
        re.compile(r"\bhijack(ed)?\s*:\s*(\d+(\.\d+)?)\s*%", re.I),
    ],
    "stability": [ re.compile(r"\bstability\b\D+?(\d+(\.\d+)?)", re.I) ],
    "entropy":   [ re.compile(r"\bentropy\b\D+?(\d+(\.\d+)?)", re.I) ],
    "mutual_information": [
        re.compile(r"\bMI\b\D+?(\d+(\.\d+)?)", re.I),
        re.compile(r"\bmutual\s*information\b\D+?(\d+(\.\d+)?)", re.I),
    ],
    "switch_rate":[ re.compile(r"\bswitch(ing)?\s*rate\b\D+?(\d+(\.\d+)?)\s*%", re.I) ],
    "win_rate":  [ re.compile(r"\bwin\s*rate\b\D+?(\d+(\.\d+)?)\s*%", re.I) ],
}

def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "")

def extract_metrics_from_blocks(block_html_list):
    text = "\n".join(strip_tags(x) for x in block_html_list)
    out = {k: [] for k in METRIC_PATTERNS}
    for key, regs in METRIC_PATTERNS.items():
        for rg in regs:
            for m in rg.finditer(text):
                nums = [g for g in (m.groups() or []) if g and re.match(r"^\d+(\.\d+)?$", g)]
                if nums:
                    try:
                        out[key].append(float(nums[-1]))
                    except:
                        pass
    return out

def summarize_metrics(agg):
    def avg(v): return sum(v)/len(v) if v else None
    labels = {
        "hijack_rate": "Hijack Rate (%)",
        "stability": "Stability",
        "entropy": "Entropy",
        "mutual_information": "Mutual Information",
        "switch_rate": "Switch Rate (%)",
        "win_rate": "Win Rate (%)",
    }
    # table rows
    rows=[]
    for k, lab in labels.items():
        v = avg(agg.get(k, []))
        if v is not None:
            rows.append((lab, f"{v:.3f}"))

    # narrative bullets
    bullets=[]
    if agg.get("hijack_rate"):
        hr = avg(agg["hijack_rate"])
        if hr is not None:
            if hr >= 50:
                bullets.append("High hijack rate overall; prioritize defenses and input controls.")
            elif hr >= 20:
                bullets.append("Moderate hijack rate; consider tuning gating thresholds and noise budget.")
            else:
                bullets.append("Low hijack rate under the tested conditions.")
    if agg.get("stability"):
        st = avg(agg["stability"])
        if st is not None:
            if st >= 0.95:
                bullets.append("Stability is strong, suggesting robust gating–memory coupling.")
            elif st >= 0.85:
                bullets.append("Stability is acceptable but shows room for improvement.")
            else:
                bullets.append("Low stability; revisit thresholds and coupling strategies.")
    if agg.get("mutual_information"):
        bullets.append("Mutual Information indicates coupling strength of internal states to outputs.")
    if agg.get("switch_rate"):
        bullets.append("Switch Rate reflects fast–slow path competition dynamics.")

    table_html = ("<table class='summary-table'><thead><tr><th>Metric</th><th>Avg</th></tr></thead><tbody>"
                  + "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in rows)
                  + "</tbody></table>") if rows else "<div class='no-metrics'>No metrics extracted.</div>"
    bullets_html = "<ul class='summary-bullets'>" + "".join(f"<li>{html.escape(x)}</li>" for x in bullets) + "</ul>" if bullets else ""

    return f"""
    <div class="summary-card">
      <div class="summary-title">Conclusion & Comparison</div>
      {table_html}
      {bullets_html}
    </div>
    """

def build(nb_path: Path, out_path: Path, verbose: bool=False):
    if not nb_path.exists():
        print(f"[ERROR] Notebook not found: {nb_path.resolve()}", file=sys.stderr)
        return 2

    if verbose:
        print(f"[INFO] Reading notebook: {nb_path}")

    nb = nbformat.read(str(nb_path), as_version=4)
    panes = []
    fig_count = 0
    code_cells = 0
    image_bundles = 0

    # aggregate metrics across all panes
    agg_metrics = {k: [] for k in METRIC_PATTERNS.keys()}

    for ci, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code_cells += 1
        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        # Collect textual outputs for this cell (streams + text/plain + text/html)
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

        # stream outputs (stdout/stderr)
        for out in outputs:
            if isinstance(out, dict) and out.get("output_type") == "stream":
                name = out.get("name", "stream")
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                add_text_block(f"stream:{name}", text)

        title_hint = nearest_h1_title(nb, ci)

        # per-output bundle: include its text + (possibly) an image
        for out in outputs:
            if not isinstance(out, dict):
                continue
            data = out.get("data", {})
            if not isinstance(data, dict):
                continue

            if "text/plain" in data:
                payload = data["text/plain"]
                if isinstance(payload, list): payload = "".join(payload)
                add_text_block("text/plain", str(payload))

            if "text/html" in data:
                payload = data["text/html"]
                if isinstance(payload, list): payload = "".join(payload)
                add_html_block("text/html", payload)

            # extract metrics from this pane's textual blocks and add to global agg
            pane_metrics = extract_metrics_from_blocks(textual_blocks)
            for k, vals in pane_metrics.items():
                agg_metrics[k].extend(vals)

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
                    caption = f"Figure {fig_count:02d}" + (f" — {html.escape(title_hint)}" if title_hint else "")

                    # LAYOUT CHANGE:
                    # Figure on TOP within a horizontally-scrollable container (.figure-scroll),
                    # text BELOW (not on the right).
                    pane_html = f"""
                    <div class="pane">
                      <div class="figure-scroll">
                        <img src="{uri}" alt="{caption}">
                      </div>
                      <div class="cap">{caption}</div>
                      <div class="pane-data">
                        {''.join(textual_blocks) if textual_blocks else "<div class='nodata'>No captured run-time text output for this figure.</div>"}
                      </div>
                    </div>
                    """
                    panes.append(pane_html)

            if verbose and found_img:
                print(f"[INFO] Cell {ci} contributed image(s).")

    # Build the bottom "Conclusion & Comparison" card
    conclusion_card = summarize_metrics(agg_metrics)

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Emotion Hijacking – Gallery (h-scroll panes + column conclusion)</title>
<style>
:root {{
  --pane-bg: #fff;
  --pane-border: #e5e7eb;
  --soft: rgba(0,0,0,0.04);
}}
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#fafafa}}
h1{{margin-bottom:8px}} .meta{{color:#666;margin-bottom:20px}}

/* ONE COLUMN grid (readable and consistent with your requirement) */
.grid{{display:grid;grid-template-columns:1fr;gap:14px}}

/* pane: image top (h-scroll), text below */
.pane{{
  border:1px solid var(--pane-border);
  border-radius:12px;
  background:var(--pane-bg);
  box-shadow:0 1px 2px var(--soft);
  padding: 10px;
}}
.figure-scroll{{
  overflow-x:auto;  /* horizontal scrollbar only inside figure area */
  overflow-y:hidden;
  padding-bottom:6px;
}}
.figure-scroll img{{display:block;height:auto; /* width unconstrained -> scroll to view fully */}}
.cap{{font-size:12px;color:#333;margin:6px 0 8px}}

.pane-data{{}}
.block{{margin-bottom:8px}}
.block-title{{font-size:12px;color:#555;margin-bottom:4px}}
pre{{white-space:pre-wrap;word-wrap:break-word;background:#f8fafc;border:1px solid #eef2f7;border-radius:8px;padding:8px;max-height:240px;overflow:auto}}
.htmlwrap{{border:1px solid #eef2f7;border-radius:8px;padding:6px;background:#fff;max-height:260px;overflow:auto}}
.nodata{{font-size:12px;color:#888}}

/* bottom summary card */
.summary-card{{
  border:1px solid var(--pane-border);
  border-radius:12px;
  background:#fcfcff;
  box-shadow:0 1px 2px var(--soft);
  padding:12px;
}}
.summary-title{{font-weight:700;margin-bottom:8px}}
.summary-table{{width:100%;border-collapse:collapse;margin:6px 0 8px 0;font-size:13px}}
.summary-table th,.summary-table td{{border:1px solid #e6e6e6;padding:6px 8px;text-align:left}}
.summary-bullets{{margin:0;padding-left:18px}}
.no-metrics{{color:#777}}
</style>
</head>
<body>
<h1>Emotion Hijacking – Gallery (h-scroll panes + column conclusion)</h1>
<div class="meta">Source: {html.escape(str(nb_path))} • Code cells: {code_cells} • Output bundles w/ images: {image_bundles} • Total figures: <b>{fig_count}</b></div>

<div class="grid">
  {''.join(panes) if panes else "<p><em>No figures found in the notebook outputs.</em></p>"}
  {conclusion_card}
</div>

</body>
</html>"""

    out_path.write_text(html_doc, encoding="utf-8")
    if verbose:
        print(f"[OK] Wrote {out_path} with {fig_count} figure(s).")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", "-n", required=True, help="Path to .ipynb with saved outputs (English version)")
    parser.add_argument("--out", "-o", default="EmotionHijack_Gallery_hscroll_with_summary.html", help="Output HTML path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    out_path = Path(args.out)
    sys.exit(build(nb_path, out_path, verbose=args.verbose))

if __name__ == "__main__":
    main()
