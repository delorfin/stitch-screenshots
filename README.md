# stitch.py — Image / Screenshot Stitcher

Stitch a sequence of overlapping images into one continuous image. Built for
viewport screenshots (long web pages, documents, scroll captures) but works for
any overlapping sequence. Handles vertical **or** horizontal sequences, detects
both the overlap and any perpendicular drift between frames, and can strip
window chrome (e.g. a macOS scrollbar) so it doesn't corrupt the match.

## Requirements

- Python 3.6+
- OpenCV (`cv2`) and NumPy

```bash
pip install -r requirements.txt   # or: pip install opencv-python numpy
```

## Quick start

```bash
# Whole folder, top-to-bottom -> ./screenshots-stitched.png
python3 stitch.py ./screenshots

# Specific files, in the order you list them
python3 stitch.py a.png b.png c.png

# Shell glob + an explicit output name
python3 stitch.py shots/*.png -o result.png

# Left-to-right; text/UI content; show what it's doing
python3 stitch.py ./screenshots -H -e -v

# Strip a macOS scrollbar before matching
python3 stitch.py ./screenshots --crop-right 16 --ignore-perp 24
```

Common flags have short forms: `-H` (horizontal), `-e` (edges), `-d` (debug),
`-v` (verbose).

### Inputs and output naming

- **Inputs** can be a single **folder** (expanded to its images, sorted by name),
  an explicit **list of files** (kept in the order given — handy for controlling
  sequence), or a **shell glob** like `shots/*.png`. Default is the current folder.
- **Output** is optional. With `-o NAME` you name it; otherwise it's auto-named
  `<folder>-stitched.png` next to the source folder (or `stitched.png` in the
  current folder for loose files). An existing auto-name is **never overwritten** —
  it gets `-2`, `-3`, … appended. (An explicit `-o` path *is* overwritten.)
- Any input that lives at the output path is skipped, so re-running is safe.

## How it works

The overlap between two frames and any perpendicular drift can't be measured
independently — the seam strips only line up once *both* are right. So for each
adjacent pair the script searches the **(overlap, shift) plane jointly**,
coarse-to-fine, minimising mean-squared error over the actual overlap strip:

1. **Normalize** — all frames are cropped to a common perpendicular size so seams
   are comparable. Horizontal stitching is handled by transposing on load and
   transposing the result back, so there is a single, well-tested code path.
2. **Joint search** — a strided coarse pass locates the basin, then a
   single-pixel refinement pinpoints the overlap and shift.
3. **Acceptance tests** — a non-zero shift or a larger-than-minimum overlap is
   only used if it beats the safe default by a relative-MSE margin, so noise
   doesn't trigger spurious shifts or over-trimming.
4. **Quality score** — each seam is scored high / medium / low from the residual
   MSE over the overlap region.

## Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `inputs` | `.` | Image files (in order), a glob, or a single folder |
| `-o`, `--output` | auto | Output path; auto-named `<folder>-stitched.png` (auto-numbered) if omitted |
| `--direction` | `vertical` | `vertical` (top→bottom) or `horizontal` (left→right) |
| `-H`, `--horizontal` | — | shorthand for `--direction horizontal` |
| `--min-overlap` | `50` | Minimum overlap to search (px) |
| `--max-overlap` | `800` | Maximum overlap to search (px) |
| `--max-shift` | `30` | Max perpendicular drift to search (px). Alias: `--max-x-shift` |
| `--shift-accept-ratio` | `0.008` | Relative MSE improvement needed to accept a drift |
| `--overlap-accept-ratio` | `0.01` | Relative MSE improvement needed to accept an overlap |
| `--guard` | `0` | Extra px trimmed at each seam |
| `--prefer` | `earlier` | Which frame wins in the overlap (`earlier` / `later`) |
| `--feather N` | `0` | Blend an N-px transition across each seam (hides faint seam lines) |
| `--sticky` | off | Auto-detect & de-duplicate fixed headers/footers repeated in every shot |
| `--sticky-top N` / `--sticky-bottom N` | auto | Force the sticky band heights instead of auto-detecting |
| `--strict` | off | Exit non-zero if any seam is low-quality |
| `--dry-run` | off | Report per-seam overlap/shift/quality and write nothing |
| `--crop-right` | `0` | Permanently crop px from the right edge of every frame |
| `--crop-bottom` | `0` | Permanently crop px from the bottom edge of every frame |
| `--ignore-perp` | `0` | Ignore px on the trailing perpendicular edge during matching (scrollbar/thumbnail). Aliases: `--ignore-right`, `--ignore-bottom` |
| `--no-shift` | off | Disable perpendicular-drift detection. Alias: `--no-x-shift` |
| `--no-overlap` | off | Skip overlap detection (plain concatenation). Alias: `--no-y-overlap` |
| `-e`, `--edges` | off | Match on Sobel edges (better for text/UI) |
| `--multiscale` | on | Fast coarse-to-fine search |
| `--no-multiscale` | — | Exhaustive single-pixel search (slower, most precise) |
| `-d`, `--debug` | off | Save per-seam previews to `debug_seams/` |
| `-v`, `--verbose` | off | Detailed per-seam logging |

### Coordinate note for `--crop-*` / `--ignore-perp`

`--crop-right` / `--crop-bottom` are applied in real screen coordinates before
anything else, so they mean the same thing in both directions. `--ignore-perp`
masks the *perpendicular* trailing edge during matching — the right edge for
vertical stitching, the bottom edge for horizontal — which is where fixed window
chrome (scrollbars, thumbnails) tends to sit.

## Reading the log (`--verbose`)

```
[1] shot_01.png <-> shot_02.png
    shift: raw=  0 accepted=  0  overlap: raw=130 accepted=130  (MSE 8869.7 -> 0.0)
    quality: high (score 1.000)
```

- **raw** — best value the search found; **accepted** — value actually used
  (falls back to the safe default if the improvement was below the ratio).
- **MSE** — match error at the minimum overlap → at the chosen alignment (lower
  is better; `0.0` means a pixel-perfect seam).

## Stitching real web-page screenshots

Long-page captures usually have a **fixed nav bar / footer** that appears in every
shot. Without help, naive stitching reproduces that chrome at every seam.

The tool **detects** repeated chrome automatically and prints a hint when it sees
some — but never strips it unless you ask, so it can't silently delete content:

```
ℹ Looks like a fixed header ~65px, footer ~180px repeats across all frames.
  Re-run with --sticky to de-duplicate it.
```

Turn on de-duplication to keep the chrome only once:

```bash
python3 stitch.py ./page-shots --sticky            # auto-detect the repeated bands
python3 stitch.py ./page-shots --sticky-top 64     # or pin the heights yourself
```

(Blank/whitespace margins shared between shots are *not* treated as chrome.)

Other useful passes:

```bash
python3 stitch.py ./shots --dry-run            # check overlaps/quality before committing
python3 stitch.py ./shots --feather 24         # soften seams if frames don't match exactly
python3 stitch.py ./shots --strict             # fail the run if any seam is low-quality (CI)
```

A low-quality seam is reported at the end of every run (with the offending file
pair); `--strict` turns that into a non-zero exit code.

## Tuning

- **Shifts not detected** — lower `--shift-accept-ratio`, raise `--max-shift`,
  add `--edges`.
- **Spurious shifts** — raise `--shift-accept-ratio`, lower `--max-shift`.
- **Over-/under-trimmed seams** — adjust `--min-overlap` / `--max-overlap`, or
  lower `--overlap-accept-ratio`.
- **Scrollbar/chrome bleeding into the match** — `--ignore-perp N` (and/or
  `--crop-right N` to remove it from the output entirely).
- **Slow on large images** — keep `--multiscale` (default), tighten
  `--max-overlap` / `--max-shift`.

## Tests

`selftest.py` builds a known master image, slices it into overlapping tiles
(with injected drift, fake scrollbar chrome, and fixed headers/footers) and
verifies the script recovers the alignment and reconstructs the master
pixel-for-pixel — covering both directions plus the input/naming,
sticky-chrome, feather, dry-run, and strict paths:

```bash
python3 selftest.py
```

> Note: these tests are synthetic (pixel-exact, but generated). Validating on
> real screenshots still needs a real capture folder — point the tool at one and
> check the result by eye.

## Behaviour on awkward inputs

The tool is designed to fail safe — it never silently drops or crops content:

- **No overlap found** → frames are concatenated whole and the seam is flagged
  low-quality with a warning (it won't invent a seam).
- **Different sizes** → all frames are padded to the largest width; nothing is
  cropped.
- **A frame smaller than the search window / fully contained** → kept untrimmed
  rather than dropped.
- **An unreadable/corrupt file** → skipped with a warning; the rest still stitch
  (it aborts only if fewer than 2 readable images remain, with a non-zero exit).
- **Grayscale / RGBA inputs** → converted to BGR automatically.

Known limitations:
- EXIF orientation is **not** auto-applied (it reads raw pixels — fine for
  screenshots, but rotate phone photos first).
- Extreme width mismatches (e.g. 4:1) match poorly and are flagged low-quality
  rather than forced.
- A genuine overlap that lands *exactly* at `--min-overlap` can be read as
  "no overlap" (the improvement is measured against that minimum). If a real
  seam is being concatenated, lower `--min-overlap` below the actual overlap.
