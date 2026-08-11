#!/usr/bin/env python3
"""Headless-Chromium renderer for the FQ operator heatmap and the SVG figures.

Drives heatmap.html through Playwright: loads a real stats .jsonl through the
page's OFFLINE file picker, flips the controls, and screenshots the result.
Also rasterises the standalone SVG figures.

Run via ./render.sh, which supplies the LD_LIBRARY_PATH / FONTCONFIG_FILE that
this box needs (no root, no system fonts -- see RENDER-REVIEW.md).
"""
import argparse
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
M5 = HERE.parent
AXES = M5 / "results" / "axes"
OUT = HERE / "renders"

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--force-color-profile=srgb",
    "--font-render-hinting=none",
]

# Wide enough that the control bar lays out as designed rather than wrapping.
VIEWPORT = {"width": 1600, "height": 1400}


def log(msg):
    print(f"[render] {msg}", flush=True)


# --------------------------------------------------------------- page driving
def load_dump(page, path, label):
    """Push a stats .jsonl through the page's offline file picker."""
    page.set_input_files("#file", str(path))
    # indexFile() streams the blob in 8 MB chunks; #record enables when done.
    page.wait_for_selector("#record:not([disabled])", timeout=180_000)
    page.wait_for_function("() => document.getElementById('scan').classList.contains('hidden')",
                           timeout=180_000)
    page.wait_for_timeout(400)
    recs = page.eval_on_selector("#record", "el => el.options.length")
    log(f"  loaded {label}: {recs} records, showing last")
    return recs


def canvas_stats(page):
    """Pull real pixel statistics off the figure canvas -- proof it drew."""
    return page.evaluate("""() => {
      const cv = document.getElementById('fig');
      const ctx = cv.getContext('2d');
      const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
      const seen = new Map();
      let n = 0;
      for (let i = 0; i < d.length; i += 4) {
        const k = (d[i] << 16) | (d[i+1] << 8) | d[i+2];
        seen.set(k, (seen.get(k) || 0) + 1);
        n++;
      }
      const top = [...seen.entries()].sort((a,b) => b[1]-a[1]).slice(0, 8)
        .map(([k, c]) => ({
          hex: '#' + k.toString(16).padStart(6, '0').toUpperCase(),
          pct: +(100*c/n).toFixed(2),
        }));
      return {w: cv.width, h: cv.height, distinct: seen.size, top};
    }""")


def shoot(page, name, selector=None, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    if selector:
        page.locator(selector).screenshot(path=str(p))
    else:
        page.screenshot(path=str(p), full_page=True)
    kb = p.stat().st_size / 1024
    log(f"  -> {p.name} ({kb:.0f} KB) {note}")
    return p


def new_page(ctx_kwargs, browser):
    ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2, **ctx_kwargs)
    page = ctx.new_page()
    page.on("pageerror", lambda e: log(f"  !! JS ERROR: {e}"))
    page.on("console", lambda m: log(f"  !! console.{m.type}: {m.text}")
            if m.type in ("error", "warning") else None)
    page.goto((HERE / "heatmap.html").as_uri())
    page.wait_for_selector("#fig", timeout=30_000)
    return ctx, page


# ------------------------------------------------------------------ scenarios
def render_heatmap(browser, scheme):
    """All heatmap shots for one colour scheme ('light' or 'dark')."""
    tag = scheme
    ctx_kwargs = {"color_scheme": scheme}
    report = {}

    axis1 = AXES / "stats-axis1_general.jsonl"
    axis2 = AXES / "stats-axis2_legal.jsonl"

    # ---- 1. default view (count metric, native order, tier strip, marginals)
    ctx, page = new_page(ctx_kwargs, browser)
    log(f"[{tag}] default view")
    load_dump(page, axis1, "axis1_general")
    report["default"] = canvas_stats(page)
    report["stale_class"] = page.evaluate(
        "() => document.getElementById('figwrap').classList.contains('stale')")
    shoot(page, f"{tag}-01-default-fullpage")
    shoot(page, f"{tag}-02-default-figure", "#fig")

    # DEFECT ISOLATION: the live poll fails (no serve), poll()'s catch adds
    # .stale unconditionally, and .figwrap.stale #fig is opacity:.55 -- so the
    # offline figure is drawn 45% faded. Drop the class and re-shoot to show
    # what the same data looks like undimmed.
    page.evaluate("() => document.getElementById('figwrap').classList.remove('stale')")
    page.wait_for_timeout(300)
    shoot(page, f"{tag}-02b-default-figure-UNDIMMED", "#fig",
          note="(same data, .stale removed)")
    page.evaluate("() => document.getElementById('figwrap').classList.add('stale')")
    page.wait_for_timeout(200)

    # ---- 2. metric toggle: mass
    log(f"[{tag}] metric = mass")
    page.click("#btn-mass")
    page.wait_for_timeout(500)
    report["mass"] = canvas_stats(page)
    report["mass_note"] = page.eval_on_selector("#mass-note", "el => el.textContent")
    shoot(page, f"{tag}-03-metric-mass", "#fig")

    page.click("#metric-seg button[data-metric='count']")
    page.wait_for_timeout(400)

    # ---- 3. dead-cell flag off vs on (the OPEN defect)
    log(f"[{tag}] dead-cell flag")
    report["dead_off"] = canvas_stats(page)
    page.check("#p-dead")
    page.wait_for_timeout(500)
    report["dead_on"] = canvas_stats(page)
    shoot(page, f"{tag}-04-dead-flagged", "#fig")
    page.uncheck("#p-dead")
    page.wait_for_timeout(300)

    # ---- 4. mismatch panel
    log(f"[{tag}] mismatch panel")
    page.check("#p-mismatch")
    page.wait_for_timeout(600)
    shoot(page, f"{tag}-05-mismatch", "#fig")
    page.uncheck("#p-mismatch")
    page.wait_for_timeout(300)

    # ---- 5. dead-cell zoom: crop the ramp legend so deltaE is judgeable
    dead_crop = page.evaluate("""() => {
      const cv = document.getElementById('fig');
      const r = cv.getBoundingClientRect();
      return {x: r.x, y: r.y, width: Math.min(r.width, 900), height: 150};
    }""")
    shoot_clip(page, f"{tag}-06-legend-closeup", dead_crop)

    ctx.close()

    # ---- 6. COMPARE mode: axis1 (A) vs axis2 (B)
    ctx, page = new_page(ctx_kwargs, browser)
    log(f"[{tag}] compare axis1 vs axis2")
    load_dump(page, axis1, "axis1_general")
    page.click("#slots button[data-mark='A']")
    page.wait_for_timeout(400)
    load_dump(page, axis2, "axis2_legal")
    page.click("#slots button[data-mark='B']")
    page.wait_for_timeout(400)
    page.check("#p-compare")
    page.wait_for_timeout(400)
    page.select_option("#cmpa", "A")
    page.select_option("#cmpb", "B")
    page.wait_for_timeout(800)
    report["compare"] = canvas_stats(page)
    report["slots"] = page.eval_on_selector_all("#slots .slot .d", "els => els.map(e => e.textContent)")
    shoot(page, f"{tag}-07-compare-axis1-vs-axis2-fullpage")
    shoot(page, f"{tag}-08-compare-axis1-vs-axis2", "#fig")

    # compare metric variants (values come from the #cmpmetric <option> list)
    for cm in ("symlog", "logratio"):
        page.select_option("#cmpmetric", cm)
        page.wait_for_timeout(700)
        shoot(page, f"{tag}-09-compare-{cm}", "#fig")
    page.select_option("#cmpmetric", "delta")
    page.wait_for_timeout(500)

    # legend strip only -- where the tick/caption collision lives
    lg = page.evaluate("""() => {
      const cv = document.getElementById('fig');
      const r = cv.getBoundingClientRect();
      return {x: r.x, y: r.y + r.height - 260, width: Math.min(r.width, 1000), height: 200};
    }""")
    shoot_clip(page, f"{tag}-10-legend-strip", lg)
    ctx.close()
    return report


def shoot_clip(page, name, clip):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    page.screenshot(path=str(p), clip=clip)
    log(f"  -> {p.name} ({p.stat().st_size/1024:.0f} KB)")
    return p


def render_svgs(browser, targets, width=1600):
    """Rasterise standalone SVG figures at a PR-friendly width."""
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = browser.new_context(viewport={"width": width, "height": 1000},
                              device_scale_factor=1)
    page = ctx.new_page()
    for svg in targets:
        if not svg.exists():
            log(f"  (skip, missing: {svg})")
            continue
        # Navigating straight to a .svg yields an SVG document with no <body>,
        # so nothing can be styled or measured. Inline it into a real HTML page.
        markup = svg.read_text()
        page.set_content(
            "<style>html,body{margin:0;padding:0;background:#fff}"
            f"svg{{width:{width}px;height:auto;display:block}}</style>{markup}",
            wait_until="load",
        )
        page.wait_for_timeout(400)
        dims = page.evaluate("""() => {
          const s = document.querySelector('svg');
          const vb = (s.getAttribute('viewBox') || '').trim().split(/[\\s,]+/).map(Number);
          const r = s.getBoundingClientRect();
          return {vw: vb[2] || r.width, vh: vb[3] || r.height,
                  rw: Math.round(r.width), rh: Math.round(r.height)};
        }""")
        page.set_viewport_size({"width": width, "height": min(max(dims["rh"], 200), 30000)})
        page.wait_for_timeout(300)
        out = OUT / f"svg-{svg.stem}-{width}px.png"
        page.locator("svg").screenshot(path=str(out))
        log(f"  -> {out.name} ({out.stat().st_size/1024:.0f} KB) "
            f"[viewBox {dims['vw']}x{dims['vh']} -> raster {dims['rw']}x{dims['rh']}]")
    ctx.close()


def make_dead_probe(dest):
    """Every real dump has 0 never-routed cells, so the dead-cell render path
    is untestable on real data. Derive a fixture from a real record by zeroing
    a block of known geometry, then look at whether you can see the hole."""
    src = AXES / "stats-axis1_general.jsonl"
    rec = json.loads(src.read_text().splitlines()[-1])
    for key in ("count", "mass"):
        if key not in rec:
            continue
        g = rec[key]
        for l in range(10, 30):          # 20 layers x 30 experts solid block
            for e in range(40, 70):
                g[l][e] = 0.0
        for l in range(len(g)):          # a diagonal of single dead cells
            g[l][(l * 7 + 150) % len(g[0])] = 0.0
    dest.write_text(json.dumps(rec) + "\n")
    log(f"  dead probe fixture: {dest} (block L10-29 x E40-69 + diagonal)")
    return dest


def render_dead_probe(browser, scheme, fixture):
    ctx, page = new_page({"color_scheme": scheme}, browser)
    log(f"[{scheme}] dead-cell probe")
    load_dump(page, fixture, "dead-probe")
    stats_off = canvas_stats(page)
    shoot(page, f"{scheme}-11-deadprobe-flag-off", "#fig")
    page.check("#p-dead")
    page.wait_for_timeout(600)
    stats_on = canvas_stats(page)
    shoot(page, f"{scheme}-12-deadprobe-flag-on", "#fig")
    ctx.close()
    return {"dead_probe_off": stats_off, "dead_probe_on": stats_on}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["heatmap", "svg", "deadprobe", "all"], default="all")
    ap.add_argument("--scheme", choices=["light", "dark", "both"], default="both")
    ap.add_argument("--svg-width", type=int, default=1600)
    args = ap.parse_args()

    report = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=LAUNCH_ARGS)
        log(f"chromium {browser.version}")
        schemes = ["light", "dark"] if args.scheme == "both" else [args.scheme]
        if args.only in ("heatmap", "all"):
            for s in schemes:
                report[s] = render_heatmap(browser, s)
        if args.only in ("deadprobe", "all"):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                fx = make_dead_probe(pathlib.Path(td) / "stats-DEAD-PROBE.jsonl")
                for s in schemes:
                    report.setdefault(s, {}).update(render_dead_probe(browser, s, fx))
        if args.only in ("svg", "all"):
            log("SVG figures")
            targets = sorted(set(
                list((M5 / "results").rglob("*.svg")) + list(HERE.glob("*.svg"))
            ))
            log(f"  found {len(targets)} SVG(s): {[t.name for t in targets]}")
            render_svgs(browser, targets, args.svg_width)
        browser.close()

    if report:
        rp = OUT / "canvas-stats.json"
        rp.write_text(json.dumps(report, indent=2))
        log(f"pixel statistics -> {rp}")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
