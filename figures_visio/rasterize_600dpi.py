import re, subprocess, os, sys

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

figs = [
    ("e5_admm_split.svg", 3498, 1800),   # 5.83in x 3.0in @ 600dpi
    ("e6_protocol.svg", 3498, 2160),     # 5.83in x 3.6in @ 600dpi
]

for src, W, H in figs:
    s = open(src, encoding="utf-8").read()
    s = re.sub(r'width="[^"]*"', 'width="%d"' % W, s, count=1)
    s = re.sub(r'height="[^"]*"', 'height="%d"' % H, s, count=1)
    tmp = src.replace(".svg", "_hi.svg")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(s)
    out = src.replace(".svg", "_600dpi.png")
    url = "file:///" + os.path.abspath(tmp).replace("\\", "/")
    cmd = [
        EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--window-size=%d,%d" % (W, H),
        "--screenshot=" + os.path.abspath(out),
        url,
    ]
    r = subprocess.run(cmd, timeout=180, capture_output=True)
    size = os.path.getsize(out) if os.path.exists(out) else None
    print(f"{src}: rc={r.returncode} -> {out} ({size} bytes)")
