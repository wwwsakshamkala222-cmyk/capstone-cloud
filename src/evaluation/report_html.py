"""REPORT.md -> self-contained HTML page (figures embedded as data URIs). Minimal markdown subset converter."""
import re, base64, html
from pathlib import Path
V3 = Path("/dev/shm/saksham/ids/v3"); md = (V3 / "results/REPORT.md").read_text().splitlines()
OUT = V3 / "results/ids_v3_report.html"

def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = s.replace("←selected", '<span class="pick">selected</span>')
    for word, cls in [("NOT MET", "bad"), ("PENDING", "wait")]:
        s = s.replace(f"<strong>{word}</strong>", f'<span class="pill {cls}">{word}</span>')
    s = re.sub(r"<strong>MET</strong>", '<span class="pill good">MET</span>', s)
    return s

def figure(path):
    p = V3 / "figures" / Path(path).name
    if not p.exists(): return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{p.stem}"><figcaption>{p.name}</figcaption></figure>'

out, i, title = [], 0, "CICIoT2023 IDS v3"
while i < len(md):
    line = md[i]
    if line.startswith("# "):
        title = line[2:].split(" — ")[0]; out.append(f"<h1>{inline(line[2:])}</h1>")
    elif line.startswith("## "):
        out.append(f"<h2>{inline(line[3:])}</h2>")
    elif line.startswith("|"):
        rows = []
        while i < len(md) and md[i].startswith("|"):
            if not re.match(r"^\|[-| ]+\|$", md[i]): rows.append([c.strip() for c in md[i].strip("|").split("|")])
            i += 1
        head, body = rows[0], rows[1:]
        t = "<div class='tw'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
        for r in body:
            cls = " class='sel'" if any("←selected" in c for c in r) else ""
            t += f"<tr{cls}>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
        out.append(t + "</tbody></table></div>"); continue
    elif line.startswith("- "):
        items = []
        while i < len(md) and md[i].startswith("- "): items.append(f"<li>{inline(md[i][2:])}</li>"); i += 1
        out.append("<ul>" + "".join(items) + "</ul>"); continue
    elif line.startswith("Figure: "):
        out.append(figure(re.search(r"`([^`]+)`", line).group(1)))
    elif line.strip():
        out.append(f"<p>{inline(line)}</p>")
    i += 1

CSS = """
:root{--bg:#f6f7f9;--panel:#ffffff;--fg:#1b2230;--muted:#5b6578;--line:#dde2ea;--accent:#2f5d9e;--good:#1f7a4d;--goodbg:#e3f3ea;--bad:#a8322d;--badbg:#f8e4e2;--wait:#8a6400;--waitbg:#fbf0d4;--sel:#eaf0fa;
--display:"IBM Plex Sans",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11151c;--panel:#181e27;--fg:#e4e8ef;--muted:#9aa4b5;--line:#2a3240;--accent:#8db3ea;--good:#6fd19e;--goodbg:#173527;--bad:#f0948e;--badbg:#3d1d1b;--wait:#e9c46a;--waitbg:#3a3015;--sel:#1d2a3d;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11151c;--panel:#181e27;--fg:#e4e8ef;--muted:#9aa4b5;--line:#2a3240;--accent:#8db3ea;--good:#6fd19e;--goodbg:#173527;--bad:#f0948e;--badbg:#3d1d1b;--wait:#e9c46a;--waitbg:#3a3015;--sel:#1d2a3d;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--body)}
main{max-width:980px;margin:0 auto;padding-block:32px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font:600 1.7rem/1.25 var(--display);text-wrap:balance;margin:0 0 12px}
h2{font:600 1.15rem/1.3 var(--display);margin:36px 0 10px;padding-top:14px;border-top:1px solid var(--line);color:var(--accent)}
p,li{max-width:75ch;color:var(--fg)} p{margin:8px 0} ul{padding-left:20px}
code{font:0.85em var(--mono);background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:0 4px;overflow-wrap:anywhere}
.tw{overflow-x:auto;margin:10px 0 16px;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;letter-spacing:.03em;text-transform:uppercase;background:var(--bg)}
tr:last-child td{border-bottom:0} tr.sel td{background:var(--sel)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.03em;white-space:nowrap}
.good{color:var(--good);background:var(--goodbg)} .bad{color:var(--bad);background:var(--badbg)} .wait{color:var(--wait);background:var(--waitbg)}
.pick{font-size:11px;font-weight:600;color:var(--accent);margin-left:6px;text-transform:uppercase;letter-spacing:.04em}
figure{margin:14px 0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:10px}
figure img{display:block;width:100%;height:auto} figcaption{font:12px var(--mono);color:#5b6578;margin-top:6px}
"""
page = f"""<title>{html.escape(title)} Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<main>
{chr(10).join(out)}
</main>
"""
OUT.write_text(page); print("wrote", OUT, len(page) // 1024, "KB")
