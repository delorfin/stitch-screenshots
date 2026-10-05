#!/usr/bin/env python3
"""
selftest.py — synthetic round-trip tests for stitch.py.

Builds a known "master" image with rich, non-repeating content, slices it into
overlapping tiles (with optional perpendicular jitter and fake scrollbar chrome),
then checks that:
  * detect_overlap / detect_shift recover the injected values,
  * --ignore-perp neutralises a changing edge stripe,
  * the full CLI reconstructs the master (vertical AND horizontal).

Run:  python3 selftest.py
Exit code 0 = all passed.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

import stitch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def make_master(h, w, seed=7):
    """Realistic screenshot-like content: aperiodic 'text' rows of varying-width
    blocks on a flat background, a few colour UI blocks, and mild noise. Gives a
    smooth, wide, non-repeating correlation basin (like real UI), not the 1px
    spikes + aliasing that pure noise / perfectly periodic lines would produce.
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 250, np.uint8)
    y = 8
    while y < h - 12:                      # irregular row spacing -> aperiodic
        x = int(rng.integers(8, 40))
        for _ in range(int(rng.integers(3, 8))):
            wlen = int(rng.integers(20, 90))
            hgt = int(rng.integers(5, 9))
            shade = int(rng.integers(30, 110))
            cv2.rectangle(img, (x, y), (min(w - 4, x + wlen), y + hgt), (shade, shade, shade), -1)
            x += wlen + int(rng.integers(8, 22))
            if x > w - 30:
                break
        y += int(rng.integers(14, 30))
    for _ in range(max(1, h // 120)):      # colour UI blocks
        y0 = int(rng.integers(0, h - 60)); x0 = int(rng.integers(0, w - 120))
        col = tuple(int(c) for c in rng.integers(60, 220, 3))
        cv2.rectangle(img, (x0, y0), (x0 + int(rng.integers(40, 110)),
                                      y0 + int(rng.integers(20, 55))), col, -1)
    noise = rng.integers(0, 6, (h, w, 3), dtype=np.uint8)
    return (img.astype(int) + noise).clip(0, 255).astype(np.uint8)


# ---------- unit-level recovery tests ----------

def test_overlap_and_shift():
    print("Unit: overlap + perpendicular-shift recovery")
    M = make_master(1600, 700)
    TH, OV, W, x0, y0 = 400, 130, 600, 40, 200

    # clean pair, no shift
    prev = M[y0:y0 + TH, x0:x0 + W].copy()
    curr = M[y0 + TH - OV:y0 + TH - OV + TH, x0:x0 + W].copy()
    a = stitch.detect_alignment(prev, curr, min_overlap=50, max_overlap=800, max_shift=40)
    check("overlap recovered (no shift)", abs(a["overlap"] - OV) <= 3, f"got {a['overlap']}, want {OV}")
    check("zero shift on aligned pair", abs(a["shift"]) <= 2, f"got {a['shift']}")

    # pair with injected horizontal jitter dx
    for dx in (6, -9, 15):
        prev = M[y0:y0 + TH, x0:x0 + W].copy()
        curr = M[y0 + TH - OV:y0 + TH - OV + TH, x0 + dx:x0 + dx + W].copy()
        a = stitch.detect_alignment(prev, curr, min_overlap=50, max_overlap=800, max_shift=40)
        check(f"shift recovered (dx={dx})", abs(a["shift"] - dx) <= 2, f"got {a['shift']}, want {dx}")
        check(f"overlap recovered with shift (dx={dx})", abs(a["overlap"] - OV) <= 3,
              f"got {a['overlap']}, want {OV}")

    # tall frames take the 4x-downscaled coarse path (scroll captures)
    T = make_master(3000, 700, seed=19)
    for ov in (90, 500, 850):
        prev, curr = T[0:900], T[900 - ov:1800 - ov]
        a = stitch.detect_alignment(prev, curr, min_overlap=50, max_overlap=100000, max_shift=0)
        check(f"overlap recovered on 900-row frames (ov={ov})", a["overlap"] == ov,
              f"got {a['overlap']}, want {ov}")


def test_ignore_perp_chrome():
    print("Unit: --ignore-perp excludes a differing edge stripe from the match")
    M = make_master(1600, 700)
    TH, OV, W, x0, y0 = 400, 130, 600, 40, 200
    STRIPE = 30
    prev = M[y0:y0 + TH, x0:x0 + W].copy()
    curr = M[y0 + TH - OV:y0 + TH - OV + TH, x0:x0 + W].copy()
    # A moving scrollbar thumb differs between shots -> the right stripe never
    # matches and injects large error into the overlap MSE.
    prev[:, W - STRIPE:] = 30
    curr[:, W - STRIPE:] = 220

    bad = stitch.detect_alignment(prev, curr, min_overlap=50, max_overlap=800, max_shift=0, ignore_perp=0)
    good = stitch.detect_alignment(prev, curr, min_overlap=50, max_overlap=800, max_shift=0,
                                   ignore_perp=STRIPE + 4)
    check("ignore-perp keeps overlap correct through chrome",
          abs(good["overlap"] - OV) <= 3, f"got {good['overlap']}, want {OV}")
    check("ignore-perp removes chrome error from the match",
          good["mse_best"] < bad["mse_best"] * 0.5,
          f"matched MSE {bad['mse_best']:.1f} -> {good['mse_best']:.1f}")


# ---------- end-to-end CLI reconstruction ----------

def slice_vertical(M, TH, OV):
    step = TH - OV
    tiles, y = [], 0
    while y + TH <= M.shape[0]:
        tiles.append(M[y:y + TH, :].copy())
        y += step
    return tiles, y - step + TH  # last covered row


def run_cli(folder, out, extra):
    cmd = [sys.executable, os.path.join(HERE, "stitch.py"), folder, "-o", out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
    return r.returncode == 0


def run_cli_files(files, out, extra):
    cmd = [sys.executable, os.path.join(HERE, "stitch.py")] + files + ["-o", out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
    return r.returncode == 0


def test_e2e(direction):
    print(f"E2E: full reconstruction ({direction})")
    tmp = tempfile.mkdtemp(prefix="stitch_test_")
    try:
        TH, OV = 380, 140
        if direction == "vertical":
            M = make_master(1700, 560, seed=11)
            tiles, covered = slice_vertical(M, TH, OV)
            ref = M[:covered, :]
        else:  # horizontal: slice a vertical master, then transpose tiles + reference
            Mv = make_master(1700, 560, seed=11)
            tiles_v, covered = slice_vertical(Mv, TH, OV)
            tiles = [stitch.to_work_frame(t, "horizontal") for t in tiles_v]
            ref = stitch.to_work_frame(Mv[:covered, :], "horizontal")

        for i, t in enumerate(tiles):
            cv2.imwrite(os.path.join(tmp, f"tile_{i:03d}.png"), t)
        out = os.path.join(tmp, "out.png")
        ok = run_cli(tmp, out, ["--direction", direction, "--no-multiscale"])
        check(f"{direction}: CLI ran", ok)
        if not ok:
            return
        res = cv2.imread(out)
        check(f"{direction}: dimensions match master",
              res.shape == ref.shape, f"got {res.shape}, want {ref.shape}")
        if res.shape == ref.shape:
            err = stitch.mse(res, ref)
            check(f"{direction}: pixels match master", err < 1.0, f"MSE={err:.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_flags():
    print("E2E: flag/edge-case smoke tests")
    tmp = tempfile.mkdtemp(prefix="stitch_flags_")
    try:
        TH, OV = 360, 130
        M = make_master(1500, 540, seed=23)
        tiles, _ = slice_vertical(M, TH, OV)

        # jitter: shift each tile horizontally by a known wandering offset
        rng = np.random.default_rng(3)
        Mw = make_master(1500, 640, seed=23)          # wider master to crop windows from
        jtiles, off = [], 60
        for k in range(len(tiles)):
            off += int(rng.integers(-8, 9))
            off = max(0, min(100, off))
            y = k * (TH - OV)
            jtiles.append(Mw[y:y + TH, off:off + 540].copy())

        def write(ts):
            for f in os.listdir(tmp):
                if f.endswith(".png"):
                    os.remove(os.path.join(tmp, f))
            for i, t in enumerate(ts):
                cv2.imwrite(os.path.join(tmp, f"t_{i:03d}.png"), t)

        cases = [
            ("plain", tiles, []),
            ("no-overlap concat", tiles, ["--no-overlap"]),
            ("guard=10", tiles, ["--guard", "10"]),
            ("prefer later", tiles, ["--prefer", "later"]),
            ("edges", tiles, ["--edges"]),
            ("exhaustive (--no-multiscale)", tiles, ["--no-multiscale"]),
            ("jitter recovery", jtiles, ["--max-shift", "40"]),
            ("debug previews", tiles, ["--debug"]),
        ]
        for name, ts, extra in cases:
            write(ts)
            out = os.path.join(tmp, "out.png")
            ok = run_cli(tmp, out, extra)
            res = cv2.imread(out) if ok else None
            check(f"flag: {name}", ok and res is not None and res.shape[0] > TH,
                  None if res is None else f"{res.shape}")

        # mismatched widths: pad up to the MAX width (no cropping / no content loss).
        write([t[:, :540] for t in tiles[:2]] + [tiles[2][:, :500]])
        out = os.path.join(tmp, "out.png")
        ok = run_cli(tmp, out, ["--no-shift"])
        res = cv2.imread(out) if ok else None
        check("flag: mismatched widths -> padded to max width (no crop)",
              ok and res is not None and res.shape[1] == 540, None if res is None else f"{res.shape}")

        check("flag: debug seam dir created",
              os.path.isdir(os.path.join(tmp, "debug_seams")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_inputs_and_naming():
    print("Inputs: file lists, folders, auto-naming, collisions")
    tmp = tempfile.mkdtemp(prefix="stitch_io_")
    try:
        shots = os.path.join(tmp, "shots")
        os.makedirs(shots)
        M = make_master(900, 480, seed=5)
        tiles, _ = slice_vertical(M, 320, 120)
        paths = [os.path.join(shots, f"t_{i:02d}.png") for i in range(len(tiles))]
        for p, t in zip(paths, tiles):
            cv2.imwrite(p, t)

        # explicit files preserve given order; folder expands sorted -> same set
        check("resolve: explicit files keep order",
              stitch.resolve_inputs(paths) == paths)
        check("resolve: folder expands to sorted images",
              stitch.resolve_inputs([shots]) == sorted(paths))
        check("resolve: non-image / missing skipped",
              stitch.resolve_inputs(paths[:2] + [os.path.join(tmp, "nope.txt")]) == paths[:2])

        # auto-name: folder -> '<folder>-stitched.png' beside the folder
        name = stitch.auto_output_name([shots], stitch.resolve_inputs([shots]))
        check("auto-name: folder -> <folder>-stitched.png",
              name == os.path.join(tmp, "shots-stitched.png"), name)

        # collision -> append -2, -3
        open(name, "w").close()
        name2 = stitch.auto_output_name([shots], stitch.resolve_inputs([shots]))
        check("auto-name: collision appends -2",
              name2 == os.path.join(tmp, "shots-stitched-2.png"), name2)
        open(name2, "w").close()
        name3 = stitch.auto_output_name([shots], stitch.resolve_inputs([shots]))
        check("auto-name: second collision appends -3",
              name3 == os.path.join(tmp, "shots-stitched-3.png"), name3)

        # end-to-end: no -o, output auto-created next to the folder
        out_auto = os.path.join(tmp, "shots2-stitched.png")
        shots2 = os.path.join(tmp, "shots2")
        os.makedirs(shots2)
        for i, t in enumerate(tiles):
            cv2.imwrite(os.path.join(shots2, f"t_{i:02d}.png"), t)
        r = subprocess.run([sys.executable, os.path.join(HERE, "stitch.py"), shots2],
                           capture_output=True, text=True)
        check("e2e: auto-named output is created",
              r.returncode == 0 and os.path.exists(out_auto),
              out_auto if not os.path.exists(out_auto) else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_chrome(w, h, seed):
    """A distinctive, non-uniform band (so it's detectable as sticky chrome)."""
    rng = np.random.default_rng(seed)
    band = np.full((h, w, 3), tuple(int(c) for c in rng.integers(40, 200, 3)), np.uint8)
    for _ in range(6):
        x = int(rng.integers(0, w - 60)); y = int(rng.integers(0, max(1, h - 8)))
        cv2.rectangle(band, (x, y), (x + 50, y + 6), (255, 255, 255), -1)
    return band


def test_sticky():
    print("Sticky: fixed header/footer de-duplication")
    HH, FH, VH, OV, W = 60, 40, 300, 110, 480
    S = make_master(1400, W, seed=42)                 # scrolling content
    header = make_chrome(W, HH, seed=1)
    footer = make_chrome(W, FH, seed=2)
    step = VH - OV
    frames, y = [], 0
    while y + VH <= S.shape[0]:
        frames.append(np.vstack([header, S[y:y + VH], footer]))
        y += step
    covered = y - step + VH

    th, bh = stitch.detect_sticky_bands(frames)
    check("sticky: header height detected", abs(th - HH) <= 3, f"got {th}, want {HH}")
    check("sticky: footer height detected", abs(bh - FH) <= 3, f"got {bh}, want {FH}")

    # a shared *blank* margin is not chrome — must not be flagged (avoids a
    # misleading --sticky hint on images that merely share white space)
    margin = np.full((50, W, 3), 250, np.uint8)
    f1 = np.vstack([margin, make_master(600, W, seed=71)])
    f2 = np.vstack([margin, make_master(600, W, seed=72)])
    tb, _ = stitch.detect_sticky_bands([f1, f2])
    check("sticky: blank shared margin is NOT flagged as chrome", tb == 0, f"got {tb}")

    tmp = tempfile.mkdtemp(prefix="stitch_sticky_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"f_{i:02d}.png"), f)
        out = os.path.join(tmp, "out.png")
        ok = run_cli(tmp, out, ["--sticky"])
        res = cv2.imread(out)
        ref = np.vstack([header, S[:covered], footer])     # chrome once, full scroll
        check("sticky: reconstructs page with chrome once",
              ok and res is not None and res.shape == ref.shape and stitch.mse(res, ref) < 1.0,
              None if res is None else f"{res.shape} vs {ref.shape}, MSE={stitch.mse(res, ref):.2f}")
        # without --sticky the repeated chrome inflates the height
        ok2 = run_cli(tmp, os.path.join(tmp, "nostick.png"), [])
        res2 = cv2.imread(os.path.join(tmp, "nostick.png"))
        check("sticky: omitting --sticky duplicates chrome (taller)",
              ok2 and res2 is not None and res2.shape[0] > ref.shape[0],
              None if res2 is None else f"no-sticky H={res2.shape[0]} vs correct {ref.shape[0]}")

        # over-large manual band must not collapse a frame to nothing (guard)
        frame_h = frames[0].shape[0]
        ok3 = run_cli(tmp, os.path.join(tmp, "big.png"), ["--sticky-top", str(frame_h + 50)])
        res3 = cv2.imread(os.path.join(tmp, "big.png"))
        check("sticky: oversized band is skipped, content preserved",
              ok3 and res3 is not None and res3.shape[0] >= ref.shape[0],
              None if res3 is None else f"{res3.shape}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_window_frames(with_scrollbar, fade=0, thumb=(0, 0, 255), th=30, count=None):
    """Whole-window frames: header / [sidebar | scrolling content] / footer, plus
    a tiny spinner in the header that changes every frame. `with_scrollbar`
    draws an overlay thumb (macOS style) over the content's right edge, moving
    with scroll progress. `fade` dims that many content rows under the header
    and above the footer, as scroll views do under pinned bars."""
    HH, FH, SW, VH, OV, W, SB, TH = 50, 30, 120, 320, 120, 420, 14, th
    S = make_master(1500, W, seed=55)
    header = make_chrome(SW + W, HH, seed=3)
    footer = make_chrome(SW + W, FH, seed=4)
    sidebar = make_chrome(SW, VH, seed=5)
    step = VH - OV
    n = count or (S.shape[0] - VH) // step + 1
    frames = []
    for k in range(n):
        y = k * step
        h = header.copy()
        h[10:22, 10:22] = (40 * k) % 256          # spinner
        view = S[y:y + VH].copy()
        if fade:
            view[:fade] = (view[:fade] * 0.5).astype(np.uint8)
            view[-fade:] = (view[-fade:] * 0.5).astype(np.uint8)
        if with_scrollbar:
            t = int((VH - TH) * k / max(1, n - 1))
            view[t:t + TH, W - SB + 3:W - 3] = thumb
        frames.append(np.vstack([h, np.hstack([sidebar, view]), footer]))
    covered = (n - 1) * step + VH
    return dict(frames=frames, S=S, header=header, footer=footer, sidebar=sidebar,
                HH=HH, FH=FH, SW=SW, VH=VH, W=W, SB=SB, TH=TH, covered=covered)


def test_window():
    print("Window: whole-window frames -> scrolling region stitched, fixed UI kept once")
    d = make_window_frames(with_scrollbar=False)
    frames, S, HH, SW, VH, W = d["frames"], d["S"], d["HH"], d["SW"], d["VH"], d["W"]

    box = stitch.detect_motion_box(frames)
    ok_box = box is not None
    if ok_box:
        y0, y1, x0, x1 = box
        inside = HH <= y0 < y1 <= HH + VH and SW <= x0 < x1 <= SW + W
        coverage = (y1 - y0) * (x1 - x0) / (VH * W)
        check("window: box excludes header, footer, sidebar and spinner", inside, f"box={box}")
        check("window: box covers most of the scrolling area", coverage > 0.85,
              f"coverage={coverage:.2f}")
    else:
        check("window: box detected", False)
    check("window: identical frames -> no box", stitch.detect_motion_box([frames[0]] * 3) is None)
    check("window: mismatched sizes -> no box",
          stitch.detect_motion_box([frames[0], frames[1][:-5]]) is None)

    tmp = tempfile.mkdtemp(prefix="stitch_window_")
    outdir = tempfile.mkdtemp(prefix="stitch_window_out_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"f_{i:02d}.png"), f)
        if ok_box:
            ry0, ry1 = y0 - HH, y1 - HH               # box rows within the viewport
            body = S[ry0:d["covered"] - (VH - ry1), x0 - SW:x1 - SW]

            out = os.path.join(outdir, "content.png")
            ok = run_cli(tmp, out, ["--content-only", "--no-shift"])
            res = cv2.imread(out)
            check("window: --content-only reconstructs only the scrolled content",
                  ok and res is not None and res.shape == body.shape and stitch.mse(res, body) < 1.0,
                  None if res is None else f"{res.shape} vs {body.shape}"
                  + (f", MSE={stitch.mse(res, body):.2f}" if res.shape == body.shape else ""))

            out = os.path.join(outdir, "window.png")
            ok = run_cli(tmp, out, ["--window", "--no-shift"])
            res = cv2.imread(out)
            ref, _ = stitch.compose_window(frames[0], frames[-1], body, box)
            check("window: fixed UI kept once around the content",
                  ok and res is not None and res.shape == ref.shape and stitch.mse(res, ref) < 1.0,
                  None if res is None else f"{res.shape} vs {ref.shape}")
            check("window: header from first frame, footer from last",
                  res is not None and res.shape == ref.shape
                  and stitch.mse(res[:HH], frames[0][:HH]) < 1.0
                  and stitch.mse(res[-d["FH"]:], d["footer"]) < 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)


def test_window_fixed_cols():
    print("Window: a scrollbar declared with --fixed-cols is not stitched into the content")
    d = make_window_frames(with_scrollbar=True)
    frames, TH, SW, W, SB = d["frames"], d["TH"], d["SW"], d["W"], d["SB"]
    bar = f"{SW + W - SB}:{SW + W}"                   # thumb columns, as the OS reports them

    def red_rows(img):
        red = (img[:, :, 2] > 200) & (img[:, :, 1] < 60) & (img[:, :, 0] < 60)
        return int(red.any(axis=1).sum())

    tmp = tempfile.mkdtemp(prefix="stitch_fixed_")
    outdir = tempfile.mkdtemp(prefix="stitch_fixed_out_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"f_{i:02d}.png"), f)
        run = lambda name, extra: (run_cli(tmp, os.path.join(outdir, name), extra),
                                   cv2.imread(os.path.join(outdir, name)))
        ok, res = run("plain.png", ["--window", "--no-shift"])
        check("fixed-cols: without it, an overlay thumb repeats (why the flag exists)",
              ok and res is not None and red_rows(res) > TH,
              None if res is None else f"{red_rows(res)} red rows")
        ok, res = run("window.png", ["--window", "--no-shift", "--fixed-cols", bar])
        check("fixed-cols: --window shows the track without any thumb (nothing left to scroll)",
              ok and res is not None and red_rows(res) == 0,
              None if res is None else f"{red_rows(res)} red rows")
        check("fixed-cols: window keeps its full width",
              res is not None and res.shape[1] == frames[0].shape[1])
        ok, res = run("content.png", ["--content-only", "--no-shift", "--fixed-cols", bar])
        check("fixed-cols: --content-only has no thumb",
              ok and res is not None and red_rows(res) == 0,
              None if res is None else f"{red_rows(res)} red rows")
        # a short page: the thumb fills most of the track in both frames
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        short = make_window_frames(with_scrollbar=True, th=200, count=2)
        for i, f in enumerate(short["frames"]):
            cv2.imwrite(os.path.join(tmp, f"f_{i:02d}.png"), f)
        ok, res = run("short.png", ["--window", "--no-shift", "--fixed-cols", bar])
        check("fixed-cols: tall thumb on a 2-frame page -> track colour, no thumb stripe",
              ok and res is not None and red_rows(res) == 0,
              None if res is None else f"{red_rows(res)} red rows")
        r = subprocess.run([sys.executable, os.path.join(HERE, "stitch.py"), tmp, "-o",
                            os.path.join(outdir, "bad.png"), "--window", "--fixed-cols", "12"],
                           capture_output=True, text=True)
        check("fixed-cols: malformed value -> clear error, non-zero exit",
              r.returncode != 0 and "expected A:B" in r.stderr, r.stderr.strip()[-80:])
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"f_{i:02d}.png"), f)

        ok, res = run("outside.png", ["--window", "--no-shift", "--fixed-cols", "0:40"])
        ok2, ref = run("plain2.png", ["--window", "--no-shift"])
        check("fixed-cols: a range outside the scrolling region changes nothing",
              ok and ok2 and res is not None and ref is not None and res.shape == ref.shape
              and stitch.mse(res, ref) < 0.01)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)


def test_pinned_bottom():
    print("Window: side panes - pinned bottom controls move down, lists stay whole")
    bg = np.full((400, 60, 3), 230, np.uint8)
    side = bg.copy(); side[20:200:20, 5:50] = 40; side[372:388, 20:40] = 40   # items + a button at the foot
    lst = bg.copy(); lst[5:395:12, 5:55] = 40                                 # a list down to the bottom
    check("pinned: button below empty space is found", stitch.pinned_bottom(side) == 372,
          f"got {stitch.pinned_bottom(side)}")
    check("pinned: a list running to the bottom is left whole", stitch.pinned_bottom(lst) == 400,
          f"got {stitch.pinned_bottom(lst)}")
    check("pinned: an empty pane has nothing to move", stitch.pinned_bottom(bg) == 400)


def test_window_alpha():
    print("Window: transparent rounded corners stay transparent, never smear down the sides")
    d = make_window_frames(with_scrollbar=False)
    R = 12
    frames = []
    for f in d["frames"]:
        f = f[:-d["FH"]]                                  # no bottom bar: content reaches the window's bottom
        a = np.full(f.shape[:2], 255, np.uint8)
        yy, xx = np.mgrid[0:R, 0:R]
        outside = (yy - R) ** 2 + (xx - R) ** 2 > R * R   # rounded-corner mask
        for ys, xs in ((slice(0, R), slice(0, R)), (slice(0, R), slice(-R, None)),
                       (slice(-R, None), slice(0, R)), (slice(-R, None), slice(-R, None))):
            m = outside[::1 if ys.start == 0 else -1, ::1 if xs.start == 0 else -1]
            a[ys, xs][m] = 0
        bgra = np.dstack([f, a])
        bgra[a == 0] = 0                                  # screencapture leaves black under alpha 0
        frames.append(bgra)
    tmp = tempfile.mkdtemp(prefix="stitch_alpha_")
    outdir = tempfile.mkdtemp(prefix="stitch_alpha_out_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"f_{i:02d}.png"), f)
        for prefer in ("earlier", "middle"):
            out = os.path.join(outdir, f"w-{prefer}.png")
            ok = run_cli(tmp, out, ["--window", "--no-shift", "--prefer", prefer])
            res = cv2.imread(out, cv2.IMREAD_UNCHANGED)
            good = ok and res is not None and res.ndim == 3 and res.shape[2] == 4
            check(f"alpha ({prefer}): output keeps an alpha channel", good,
                  None if res is None else f"shape {res.shape}")
            if good:
                h = res.shape[0]
                corners = [res[0, 0, 3], res[0, -1, 3], res[-1, 0, 3], res[-1, -1, 3]]
                check(f"alpha ({prefer}): all four corners transparent",
                      all(c == 0 for c in corners), f"{corners}")
                check(f"alpha ({prefer}): interior fully opaque", (res[R:h - R, R:-R, 3] == 255).all())
                side = res[R:h - R, :3, :3]                 # left edge of the sidebar column
                check(f"alpha ({prefer}): no black smear down the side below the first frame",
                      int((side.max(axis=2) < 20).sum()) == 0,
                      f"{int((side.max(axis=2) < 20).sum())} near-black px")
        out2 = os.path.join(outdir, "c.png")
        ok = run_cli(tmp, out2, ["--content-only", "--no-shift"])
        res2 = cv2.imread(out2, cv2.IMREAD_UNCHANGED)
        check("alpha: --content-only output stays opaque BGR",
              ok and res2 is not None and res2.ndim == 3 and res2.shape[2] == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)


def test_feather():
    print("Feather: actually blends differing overlap content (not a no-op)")
    tmp = tempfile.mkdtemp(prefix="stitch_feather_")
    try:
        W, TH, OV, D = 240, 300, 100, 30
        M = make_master(700, W, seed=7)
        prev = M[0:TH, :].copy()
        # curr overlaps prev's bottom OV rows, but is uniformly D DARKER
        # (subtract, not add — the master bg is ~250 and +D would clip at 255,
        # hiding the signal). This simulates brightness drift so blending shows.
        curr = (M[TH - OV:2 * TH - OV, :].astype(int) - D).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tmp, "a.png"), prev)
        cv2.imwrite(os.path.join(tmp, "b.png"), curr)
        a, b = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")

        hard = os.path.join(tmp, "hard.png")
        soft = os.path.join(tmp, "soft.png")
        ok1 = run_cli_files([a, b], hard, [])
        ok2 = run_cli_files([a, b], soft, ["--feather", "90"])
        rh, rs = cv2.imread(hard), cv2.imread(soft)
        # both should detect overlap=OV -> height 2*TH-OV
        H = 2 * TH - OV
        ok_dims = rh is not None and rs is not None and rh.shape[0] == H and rs.shape[0] == H
        check("feather: overlap detected, dims match", ok1 and ok2 and ok_dims,
              None if rs is None else f"hard={rh.shape} soft={rs.shape} want H={H}")
        if ok_dims:
            zone_top = slice(TH - OV + 5, TH - OV + 15)   # near top of overlap
            zone_bot = slice(TH - 15, TH - 5)             # near bottom of overlap
            # hard cut (prefer=earlier) keeps pure prev across the whole overlap
            hard_jump = float(rh[zone_bot].mean()) - float(rh[zone_top].mean())
            # feather pulls curr (D darker) in toward the bottom of the overlap
            soft_pull = float(rs[zone_bot].mean()) - float(rh[zone_bot].mean())
            check("feather: hard cut leaves overlap unblended (no drift pulled in)",
                  abs(hard_jump) < 12, f"hard zone delta={hard_jump:.1f}")
            check("feather: blend pulls later frame (-%d) in at the seam" % D,
                  soft_pull < -D * 0.5, f"pulled {soft_pull:.1f} (of -{D})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dry_run():
    print("Dry-run: reports without writing")
    tmp = tempfile.mkdtemp(prefix="stitch_dry_")
    try:
        M = make_master(1200, 480, seed=8)
        tiles, _ = slice_vertical(M, 340, 120)
        for i, t in enumerate(tiles):
            cv2.imwrite(os.path.join(tmp, f"t_{i:02d}.png"), t)
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "stitch.py"), tmp, "--dry-run"],
            capture_output=True, text=True)
        wrote_nothing = not os.path.exists(os.path.join(tmp, "out.png")) and \
            not any(f.endswith("-stitched.png") for f in os.listdir(tmp))
        check("dry-run: exits 0, reports overlap, writes no image",
              r.returncode == 0 and "overlap:" in r.stderr + r.stdout and wrote_nothing)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_strict():
    print("Strict: low-quality seam forces non-zero exit")
    tmp = tempfile.mkdtemp(prefix="stitch_strict_")
    try:
        M = make_master(1000, 480, seed=9)
        t0 = M[0:340, :].copy()
        t1 = M[220:560, :].copy()                 # overlaps t0
        rng = np.random.default_rng(0)
        t2 = rng.integers(0, 255, (340, 480, 3), dtype=np.uint8)  # unrelated -> low quality
        for i, t in enumerate([t0, t1, t2]):
            cv2.imwrite(os.path.join(tmp, f"t_{i}.png"), t)

        out = os.path.join(tmp, "out.png")
        ok_loose = run_cli(tmp, out, [])
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "stitch.py"), tmp, "-o", out, "--strict"],
            capture_output=True, text=True)
        check("strict: succeeds (exit 0) without --strict", ok_loose)
        check("strict: exits non-zero with --strict on low-quality seam",
              r.returncode != 0, f"exit={r.returncode}")
        check("strict: still names the bad pair", "low-quality seam" in r.stdout + r.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_overlap_honesty():
    print("No-overlap: unrelated frames concatenated whole, flagged honestly")
    tmp = tempfile.mkdtemp(prefix="stitch_noov_")
    try:
        # two uncorrelated images (same width) — no alignment can improve MSE
        rng = np.random.default_rng(101)
        a = rng.integers(0, 255, (700, 480, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (700, 480, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp, "a.png"), a)
        cv2.imwrite(os.path.join(tmp, "b.png"), b)
        out = os.path.join(tmp, "out.png")
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "stitch.py"),
             os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png"), "-o", out],
            capture_output=True, text=True)
        res = cv2.imread(out)
        msg = r.stdout + r.stderr
        # nothing trimmed: full height of both frames is preserved
        check("no-overlap: concatenated with no truncation",
              res is not None and res.shape[0] == a.shape[0] + b.shape[0],
              None if res is None else f"{res.shape[0]} vs {a.shape[0] + b.shape[0]}")
        check("no-overlap: reported honestly (not silently 'high')",
              "no overlap detected" in msg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_adversarial():
    print("Adversarial: degenerate inputs never silently drop or abort")
    tmp = tempfile.mkdtemp(prefix="stitch_adv_")
    try:
        M = make_master(1200, 480, seed=33)

        def cli(args):
            return subprocess.run([sys.executable, os.path.join(HERE, "stitch.py")] + args,
                                  capture_output=True, text=True)

        # tiny images (< min_overlap): must concatenate, not drop the 2nd frame
        cv2.imwrite(os.path.join(tmp, "t1.png"), M[:20, :30])
        cv2.imwrite(os.path.join(tmp, "t2.png"), M[10:30, :30])
        out = os.path.join(tmp, "tiny.png")
        cli([os.path.join(tmp, "t1.png"), os.path.join(tmp, "t2.png"), "-o", out])
        res = cv2.imread(out)
        check("adversarial: tiny frames concatenated (not dropped)",
              res is not None and res.shape[0] == 40, None if res is None else f"{res.shape}")

        # single image -> graceful non-zero exit, no output
        only = os.path.join(tmp, "only.png"); cv2.imwrite(only, M[:200])
        r = cli([only, "-o", os.path.join(tmp, "x.png")])
        check("adversarial: single image exits non-zero", r.returncode != 0, f"exit={r.returncode}")

        # corrupt file among 2 good ones -> skip it, stitch the good pair
        d = os.path.join(tmp, "mixed"); os.makedirs(d)
        cv2.imwrite(os.path.join(d, "1.png"), M[0:360])
        cv2.imwrite(os.path.join(d, "2.png"), M[240:600])
        open(os.path.join(d, "3bad.png"), "w").write("garbage")
        out2 = os.path.join(tmp, "mixed.png")
        r = cli([d, "-o", out2])
        res2 = cv2.imread(out2)
        check("adversarial: corrupt file skipped, good pair still stitched",
              r.returncode == 0 and res2 is not None and "skipping unreadable" in r.stdout + r.stderr,
              None if res2 is None else f"{res2.shape}")

        # only corrupt files -> graceful non-zero exit
        d2 = os.path.join(tmp, "allbad"); os.makedirs(d2)
        cv2.imwrite(os.path.join(d2, "ok.png"), M[:200])
        open(os.path.join(d2, "bad.png"), "w").write("garbage")
        r = cli([d2, "-o", os.path.join(d2, "o.png")])
        check("adversarial: <2 readable images exits non-zero", r.returncode != 0, f"exit={r.returncode}")

        # grayscale + RGBA inputs load fine and reconstruct the master content.
        # Both inputs carry the SAME gray content (one as 1-channel, one as RGBA
        # with gray color channels) so the all-gray reference matches.
        d3 = os.path.join(tmp, "chan"); os.makedirs(d3)
        TH, OV = 380, 120
        g = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(d3, "a.png"), g[:TH])                                   # grayscale PNG
        cv2.imwrite(os.path.join(d3, "b.png"),
                    cv2.cvtColor(g[TH - OV:2 * TH - OV], cv2.COLOR_GRAY2BGRA))           # RGBA, gray content
        out3 = os.path.join(tmp, "chan.png")
        r = cli([os.path.join(d3, "a.png"), os.path.join(d3, "b.png"), "-o", out3])
        res3 = cv2.imread(out3)
        ref3 = cv2.cvtColor(g[:2 * TH - OV], cv2.COLOR_GRAY2BGR)
        ok3 = r.returncode == 0 and res3 is not None and res3.shape == ref3.shape
        check("adversarial: grayscale + RGBA reconstruct master content",
              ok3 and stitch.mse(res3, ref3) < 1.0,
              None if res3 is None else (f"{res3.shape}, MSE={stitch.mse(res3, ref3):.2f}"
                                         if ok3 else f"{res3.shape} vs {ref3.shape}"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prefer_later():
    print("Prefer: 'later' keeps the later frame's pixels in the overlap")
    tmp = tempfile.mkdtemp(prefix="stitch_prefer_")
    try:
        W, TH, OV, D = 240, 300, 100, 30
        M = make_master(700, W, seed=15)
        prev = M[0:TH, :].copy()
        curr = (M[TH - OV:2 * TH - OV, :].astype(int) - D).clip(0, 255).astype(np.uint8)  # darker
        cv2.imwrite(os.path.join(tmp, "a.png"), prev)
        cv2.imwrite(os.path.join(tmp, "b.png"), curr)
        a, b = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
        early = os.path.join(tmp, "early.png"); later = os.path.join(tmp, "later.png")
        run_cli_files([a, b], early, ["--prefer", "earlier"])
        run_cli_files([a, b], later, ["--prefer", "later"])
        re_, rl = cv2.imread(early), cv2.imread(later)
        zone = slice(TH - OV + 5, TH - 5)             # inside the overlap region
        if re_ is not None and rl is not None and re_.shape == rl.shape:
            delta = float(rl[zone].mean()) - float(re_[zone].mean())
            # earlier keeps prev (normal), later keeps curr (D darker) -> ~D darker
            check("prefer later: overlap sourced from later frame (-%d)" % D,
                  delta < -D * 0.6, f"later-earlier overlap delta={delta:.1f} (want ~-{D})")
        else:
            check("prefer later: outputs comparable", False,
                  f"{None if re_ is None else re_.shape} vs {None if rl is None else rl.shape}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prefer_middle():
    print("Prefer: 'middle' keeps faded frame edges out of the seams")
    tmp = tempfile.mkdtemp(prefix="stitch_middle_")
    try:
        TH, OV, FADE = 360, 140, 25
        M = make_master(1500, 480, seed=61)
        tiles, covered = slice_vertical(M, TH, OV)
        last = len(tiles) - 1
        for i, t in enumerate(tiles):
            # scroll views often fade content under pinned bars: dim the edges
            # that sit inside an overlap (not the page's own top/bottom)
            if i > 0:
                t[:FADE] = (t[:FADE] * 0.5).astype(np.uint8)
            if i < last:
                t[-FADE:] = (t[-FADE:] * 0.5).astype(np.uint8)
            cv2.imwrite(os.path.join(tmp, f"t_{i:02d}.png"), t)
        ref = M[:covered]
        outdir = tempfile.mkdtemp(prefix="stitch_middle_out_")   # outside the input folder
        mid, early = os.path.join(outdir, "mid.png"), os.path.join(outdir, "early.png")
        ok1 = run_cli(tmp, mid, ["--prefer", "middle", "--no-shift"])
        ok2 = run_cli(tmp, early, ["--prefer", "earlier", "--no-shift"])
        rm, re_ = cv2.imread(mid), cv2.imread(early)
        check("prefer middle: reconstructs master with no fade bands",
              ok1 and rm is not None and rm.shape == ref.shape and stitch.mse(rm, ref) < 1.0,
              None if rm is None else f"{rm.shape} vs {ref.shape}"
              + (f", MSE={stitch.mse(rm, ref):.2f}" if rm.shape == ref.shape else ""))
        check("prefer earlier: keeps fade bands (control)",
              ok2 and re_ is not None and re_.shape == ref.shape and stitch.mse(re_, ref) > 5.0,
              None if re_ is None else f"MSE={stitch.mse(re_, ref):.2f}" if re_.shape == ref.shape
              else f"{re_.shape} vs {ref.shape}")
        shutil.rmtree(outdir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_seam_row():
    print("Seam: --prefer middle cuts where the frames agree, avoiding animated content")
    M = make_master(800, 300, seed=77)
    ov = M[100:500].copy()                                  # a 400-row overlap
    check("seam: identical overlap -> exact middle", stitch.seam_row(ov, ov.copy()) == 200,
          f"got {stitch.seam_row(ov, ov.copy())}")
    changed = ov.copy()
    changed[120:300] = 255 - changed[120:300]               # an animation changed this block
    r = stitch.seam_row(ov, changed)
    check("seam: cut avoids the changed block", not 120 <= r < 300, f"cut at {r}")
    check("seam: cut stays clear of the overlap's ends", 24 <= r <= 376, f"cut at {r}")


def test_quality_calibration():
    print("Quality: calibration thresholds (high<300<medium<1500<low MSE)")
    W, OV = 200, 120
    M = make_master(600, W, seed=21)
    prev = M[0:240, :].copy()
    for D, want in ((12, "high"), (30, "medium"), (48, "low")):  # MSE ~ D^2: 144 / 900 / 2304
        ov = (prev[-OV:].astype(int) - D).clip(0, 255).astype(np.uint8)  # darken (avoid 255 clip)
        curr = np.vstack([ov, M[100:100 + (240 - OV), :]])
        q = stitch.assess_stitch_quality(prev, curr, 0, OV)
        check(f"quality: D={D} (MSE~{D*D}) -> {want}", q["confidence"] == want,
              f"got {q['confidence']} (mse {q.get('mse', 0):.0f})")


def test_ignore_perp_cli():
    print("Ignore-perp: flag works end-to-end through the CLI")
    tmp = tempfile.mkdtemp(prefix="stitch_ipcli_")
    try:
        W, TH, OV, STRIPE = 300, 320, 120, 30
        M = make_master(700, W, seed=27)
        prev = M[0:TH, :].copy()
        curr = M[TH - OV:2 * TH - OV, :].copy()
        prev[:, W - STRIPE:] = 30          # differing right-edge chrome
        curr[:, W - STRIPE:] = 220
        cv2.imwrite(os.path.join(tmp, "a.png"), prev)
        cv2.imwrite(os.path.join(tmp, "b.png"), curr)
        out = os.path.join(tmp, "o.png")
        ok = run_cli_files([os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")], out,
                           ["--ignore-perp", str(STRIPE + 4)])
        res = cv2.imread(out)
        # overlap should be found and trimmed -> height 2*TH-OV, and the content
        # left of the chrome stripe should match the master
        H = 2 * TH - OV
        ref = M[:H, :W - STRIPE]
        good = (res is not None and res.shape[0] == H
                and stitch.mse(res[:, :W - STRIPE], ref) < 1.0)
        check("ignore-perp CLI: overlap found through chrome, content correct",
              ok and good, None if res is None else f"{res.shape} want H={H}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_loose_file_naming():
    print("Auto-name: loose files in cwd -> 'stitched.png' (collision-numbered)")
    tmp = tempfile.mkdtemp(prefix="stitch_loose_")
    try:
        a = os.path.join(tmp, "x.png"); b = os.path.join(tmp, "y.png")
        cv2.imwrite(a, make_master(300, 200, seed=1)[:150])
        cv2.imwrite(b, make_master(300, 200, seed=2)[:150])
        # loose files whose common dir == that dir -> '<dir>-stitched.png' beside it,
        # but when the common dir resolves to cwd it should be bare 'stitched.png'.
        files = stitch.resolve_inputs([a, b])
        # simulate cwd == tmp by checking the cwd branch directly
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            name = stitch.auto_output_name(["x.png", "y.png"], ["x.png", "y.png"])
            check("auto-name: loose files in cwd -> stitched.png",
                  os.path.basename(name) == "stitched.png" and os.path.dirname(name) in ("", "."),
                  name)
            open("stitched.png", "w").close()
            name2 = stitch.auto_output_name(["x.png", "y.png"], ["x.png", "y.png"])
            check("auto-name: cwd collision -> stitched-2.png",
                  os.path.basename(name2) == "stitched-2.png", name2)
        finally:
            os.chdir(cwd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_overlap_and_shift()
    test_ignore_perp_chrome()
    test_e2e("vertical")
    test_e2e("horizontal")
    test_cli_flags()
    test_inputs_and_naming()
    test_sticky()
    test_window()
    test_window_fixed_cols()
    test_pinned_bottom()
    test_window_alpha()
    test_feather()
    test_dry_run()
    test_strict()
    test_no_overlap_honesty()
    test_adversarial()
    test_prefer_later()
    test_prefer_middle()
    test_seam_row()
    test_quality_calibration()
    test_ignore_perp_cli()
    test_loose_file_naming()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
