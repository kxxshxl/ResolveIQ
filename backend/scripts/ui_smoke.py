"""Browser smoke test of the whole support console (and the screenshot generator for the docs).

    python scripts/ui_smoke.py --ui http://localhost:5174 [--shots ../docs/screenshots] [--browser msedge]

Needs a running API with a seeded database and the UI (npm run dev, or the built frontend behind a proxy), plus `pip install playwright`. With --browser msedge or chrome it
drives the browser that is already installed; otherwise run `playwright install chromium` first. Every view is opened and its key elements and interactions are asserted
(step -> evidence highlighting, replay diff, retrieval-lab relevance, clustered groups, quality report, health). Any uncaught page error or console error fails the run.
Nothing is written to the application except what the app itself records when you resolve a complaint and replay a case.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

COMPLAINT = "My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me."


def flow(page: Page, ui: str, shots: Path | None, browser) -> None:
    def shot(name: str, full: bool = True):
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shots / f"{name}.png"), full_page=full)

    # ---- shell and navigation
    page.goto(f"{ui}/#/")
    expect(page.get_by_role("navigation", name="Sections")).to_be_visible()
    for name in ("Resolve", "Case replay", "Retrieval lab", "Discovery", "Drift monitoring", "Feedback & quality", "Evaluation", "System health"):
        expect(page.get_by_role("link", name=name)).to_be_visible()
    page.keyboard.press("?")
    expect(page.get_by_role("dialog", name="Keyboard shortcuts")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog")).to_have_count(0)

    # ---- resolve: result, step -> evidence highlighting, graph, trace
    page.locator("#complaint").fill(COMPLAINT)
    page.get_by_role("button", name="Resolve", exact=True).click()
    expect(page.get_by_text("Grounded resolution")).to_be_visible()
    expect(page.get_by_role("heading", name="Recommended resolution")).to_be_visible()
    steps = page.locator("ol.steps button.step")
    assert steps.count() >= 2, "the resolution should have steps"
    steps.first.click()
    expect(steps.first).to_have_attribute("aria-pressed", "true")
    assert page.locator(".src.hl").count() >= 1, "selecting a step must highlight a supporting source"
    assert page.locator("svg.graph .g-node.sel").count() == 1
    page.keyboard.press("2")
    expect(steps.nth(1)).to_have_attribute("aria-pressed", "true")
    page.get_by_role("tab", name="Trace").click()
    expect(page.get_by_text("retrieve", exact=True).first).to_be_visible()
    page.get_by_role("tab", name="Confidence").click()
    expect(page.get_by_text("Evidence strength")).to_be_visible()
    page.get_by_role("tab", name="Sources").click()
    shot("console-resolve")
    if shots:
        page.get_by_role("heading", name="Resolution lineage").locator("xpath=ancestor::section[1]").screenshot(path=str(shots / "console-evidence-graph.png"))
    page.get_by_role("button", name="👎 Not helpful").click()
    expect(page.get_by_role("button", name="Send feedback")).to_be_visible()
    page.get_by_role("button", name="Cancel").click()

    # ---- case replay
    page.get_by_role("button", name="Open as case").click()
    expect(page.get_by_role("heading", name="Replay", exact=True)).to_be_visible()
    expect(page.get_by_role("tab", name="Stage trace")).to_be_visible()
    page.get_by_label("Call the model").uncheck()      # a replay without the model finishes in milliseconds
    page.get_by_role("button", name="Replay case").click()
    expect(page.get_by_text("Reproduced exactly").or_(page.get_by_text("The replay differs from the original"))).to_be_visible()
    page.get_by_role("button", name="Compare retrieval strategies").click()
    expect(page.locator(".lab-col").first).to_be_visible()
    shot("console-case-replay")

    # ---- retrieval lab with ground truth
    page.goto(f"{ui}/#/lab")
    page.get_by_label("Labelled example (ground truth known)").select_option(index=1)
    page.get_by_role("button", name="Compare").click()
    expect(page.locator(".lab-col")).to_have_count(5)
    assert page.locator(".res.relevant").count() >= 1 and page.get_by_text("ground truth known").count() >= 1
    expect(page.get_by_text("Same complaint, different words")).to_be_visible()
    shot("console-retrieval-lab")

    # ---- discovery, drift, recurring clusters
    page.goto(f"{ui}/#/discovery")
    expect(page.get_by_role("heading", name="Emerging-class discovery")).to_be_visible()
    shot("console-discovery")
    page.goto(f"{ui}/#/drift")
    expect(page.get_by_role("heading", name="Drift monitoring", level=1)).to_be_visible()
    expect(page.get_by_text("Current vs baseline distribution")).to_be_visible()
    shot("console-drift")
    page.get_by_role("tab", name="Recurring complaint clusters").click()
    expect(page.get_by_text("Support-side recurring complaint clusters.")).to_be_visible()
    expect(page.locator("article.proposal").first).to_be_visible()
    shot("console-recurring")

    # ---- feedback and quality
    page.goto(f"{ui}/#/quality")
    expect(page.get_by_role("heading", name="Feedback & quality")).to_be_visible()
    expect(page.get_by_role("heading", name="Improvement report")).to_be_visible()
    shot("console-quality")

    # ---- evaluation and health
    page.goto(f"{ui}/#/evaluation")
    expect(page.get_by_role("heading", name="Retrieval quality")).to_be_visible()
    shot("console-evaluation")
    page.goto(f"{ui}/#/health")
    expect(page.get_by_text("All systems ready")).to_be_visible()
    expect(page.get_by_role("heading", name="Database", exact=True)).to_be_visible()
    shot("console-health")

    # ---- keyboard navigation and a phone-sized layout
    page.goto(f"{ui}/#/")
    page.keyboard.press("g")
    page.keyboard.press("c")
    expect(page.get_by_role("heading", name="Case replay")).to_be_visible()
    small = browser.new_context(viewport={"width": 390, "height": 844}, color_scheme="light").new_page()
    small.goto(f"{ui}/#/")
    expect(small.get_by_role("button", name="Open navigation")).to_be_visible()
    assert small.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2"), "no horizontal page scroll on a phone"
    if shots:
        small.screenshot(path=str(shots / "console-mobile.png"), full_page=False)


def run(ui: str, shots: Path | None, browser_name: str, verbose: bool = False) -> int:
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({"channel": browser_name} if browser_name in ("msedge", "chrome") else {}))
        page = browser.new_context(viewport={"width": 1360, "height": 900}, color_scheme="light").new_page()
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("console", lambda m: problems.append(f"console {m.type}: {m.text[:300]}") if m.type == "error" else None)
        page.on("response", lambda r: problems.append(f"HTTP {r.status} {r.request.method} {r.url}") if r.status >= 400 else None)
        if verbose:
            page.on("request", lambda r: print("REQ ", r.method, r.url.split("/api/")[-1]))
            page.on("response", lambda r: print("RESP", r.status, r.url.split("/api/")[-1]))
        page.on("requestfailed", lambda r: problems.append(f"request failed {r.method} {r.url} {r.failure}"))
        page.set_default_timeout(90_000)
        try:
            flow(page, ui, shots, browser)
        except Exception:
            shot_path = Path(tempfile.gettempdir()) / "resolveiq_ui_smoke_failure.png"
            page.screenshot(path=str(shot_path))
            print("FAILED at", page.url, "- screenshot:", shot_path)
            print("visible text:", page.locator("main").inner_text()[-1800:].replace("\n", " | "))
            print("console / page errors so far:", *problems, sep="\n  ")
            raise
        finally:
            browser.close()
    if problems:
        print("PROBLEMS:\n" + "\n".join(problems))
        return 1
    print("UI SMOKE TEST PASSED: every view rendered, interactions behaved, no console errors")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default="http://localhost:5174")
    ap.add_argument("--shots", default=None)
    ap.add_argument("--browser", default="msedge")
    ap.add_argument("--verbose", action="store_true", help="log every POST the page makes")
    a = ap.parse_args()
    sys.exit(run(a.ui, Path(a.shots) if a.shots else None, a.browser, a.verbose))
