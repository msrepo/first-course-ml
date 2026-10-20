"""Build docs/_includes/nav.html (the sidebar) from the chapter groups in docs/index.md.

Run after editing docs/index.md:  python3 scripts/gen_nav.py
"""
import html
import pathlib
import re

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
LINK = re.compile(r"^- \[(.+?)\]\((.+?)\)\s*$")
EX = re.compile(r"^(Exercises? [\d.]+(?: and [\d.]+)?),")


def short(text):
    body = re.sub(r"^(Notes|Interactive):\s*", "", text)
    m = EX.match(body)
    return m.group(1) if m else body[0].upper() + body[1:]


out = ['<a class="nav-home" href="index.html">A First Course in Machine Learning</a>']
open_details = False
open_ul = False
for line in (DOCS / "index.md").read_text().splitlines():
    if line.startswith("## "):
        if open_ul:
            out.append("</ul>")
            open_ul = False
        if open_details:
            out.append("</details>")
        label = line[3:].strip()
        m = re.match(r"Chapter (\d+): (.*)", label)
        out.append(f'<details><summary>Ch. {m.group(1)}: {html.escape(m.group(2))}</summary>' if m else f"<details><summary>{html.escape(label)}</summary>")
        open_details = True
    elif line.startswith("### ") and open_details:
        if open_ul:
            out.append("</ul>")
            open_ul = False
        out.append(f'<div class="nav-sub">{html.escape(line[4:].strip())}</div>')
    else:
        m = LINK.match(line)
        if m and open_details:
            if not open_ul:
                out.append("<ul>")
                open_ul = True
            out.append(f'<li><a href="{m.group(2)}">{html.escape(short(m.group(1)))}</a></li>')
if open_ul:
    out.append("</ul>")
if open_details:
    out.append("</details>")
(DOCS / "_includes" / "nav.html").write_text("\n".join(out) + "\n")
print("wrote nav with", sum(1 for l in out if "<li>" in l), "links")
