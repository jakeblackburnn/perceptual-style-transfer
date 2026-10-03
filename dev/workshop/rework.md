# Workshop: rework
2026-10-03 · status: closed (changes are on branch `rework`, not merged into `main`)
**Material:** `dev/brainstorm/rework/` (index plus four notes) and the tracked code at `dfbac70`;
`demo/` in the checkout. Goal from `dev/journal.md`: CLI-first, optional localhost UI, better
code, new models. Already decided there: the public site comes down.

## Tensions
| # | tension | depends on | status | outcome |
|---|---|---|---|---|
| 1 | A model file does not say what it is; three places look for it differently | – | decided | `model.json` beside each final `.pth`; one root, `artifacts/models` |
| 2 | CLI surface: three commands or thirteen, and how it is installed | 1 | decided | four commands; `pyproject.toml` console script |
| 3 | Local server: new tracked package or the untracked `demo/` slimmed | 2 | decided | new tracked `style_transfer_web/`; same features as today |
| 4 | Loss: apply the ignored layer weights only, or also move to the paper's reductions + TV | – | decided | match the paper now; style weights rescaled as untested starting points |
| 5 | `perceptual_loss` shape: keep the function with a global VGG, or a `PerceptualLoss` object | 4 | parked | waits on diffusion work resuming |
| 6 | Models: default set, and train first or build a benchmark first | 4 | decided | four defaults; first run is the long Kanagawa reference |

Decided on the spot (no real alternative; go to the change list):
- a. Style layer weights are read and never applied (`style_transfer/loss.py:29`): apply them,
  with a test. How far the loss changes beyond that is tension 4.
- b. `kanagawa_custom_layers` cannot resolve its preset (`config/styles/kanagawa.py:7`,
  `train.py:101`): move the preset into `LAYER_PRESETS`; add a test that every registered
  experiment resolves.
- c. Output size is rounded up to a multiple of four (65×97 → 68×100): pad and crop back inside
  `stylize_image` (`inference.py:36`), with odd-size tests; convert to RGB there too.
- d. Delete `style_transfer/apply_style.py` and `style_transfer/build.py`; the interactive
  `run_experiment.py` goes once the CLI covers it (tension 2).
- e. `README.md:3,19` links to the public site: remove.
- f. Visualizer normalizes twice (`utils/visualize.py:315`, `vgg.py:28`): drop the transform's
  normalization.

## Threads
### 1. Model identity and lookup
Evidence: final file is a bare state dict (`train.py:174`); size comes from the current config
(`inference.py:111`); `train.py:105` writes `models/`, `run_experiment.py:41` reads `models/`,
`inference.py:59` prefers `artifacts/models/`; the demo has its own three-entry table
(`demo/core/config.py:53`).
**Decided:** training writes `model.json` beside the final `.pth` (model size, resolved layer
preset, style weight, style and content source, steps, commit). One models root,
`artifacts/models`, which training writes to and the CLI and server read from; `--models-dir`
overrides it; top-level `models/` stops being used. A one-off script writes `model.json` for the
23 archived models from the current config. Inference reads size from `model.json`, not from the
experiment config.
Why: the registry option's `fetch` has nothing to download yet, and metadata inside the `.pth`
adds a third checkpoint format for the loader. Not chosen: catalog, checksums, a home-directory
root.

### 2. CLI surface and install
**Decided:** four commands: `style-transfer list` (models and experiments), `stylize` (one
image with `-o`, or a directory with `--output-dir`), `train --experiment`, `serve --port`.
Installed by `pip install -e .` from a `pyproject.toml` console script; `python -m
style_transfer` does the same. Optional groups `[ui]`, `[viz]`, `[research]`, `[dev]`;
`requirements.txt` goes away. `run_experiment.py`'s interactive menus are deleted. The
visualizer and ONNX export stay as `python -m` modules, not subcommands.
Why: the thirteen-command proposal is mostly registry commands that thread 1 dropped. Not
chosen: utilities as subcommands, keeping the menus, `python -m` only.

### 3. Local server
**Decided:** a new tracked `style_transfer_web/` beside the library: one FastAPI app bound to
`127.0.0.1`, serving the existing static page (copied from `demo/web_ui/public`, with relative
API URLs) and stylizing in one request that returns the image. No Node server, no upload store,
no CORS, no Docker. Features stay as today: pick a model, upload, view, download.
`style_transfer/` never imports it; `style-transfer serve` starts it. `demo/` is retired after
the takedown.
Why: a fresh clone gets the UI, and the three-request upload/transfer/download flow has a
stale-upload bug that a single request removes. Not chosen: slimming `demo/` in place, a
max-side control in the page, side-by-side compare.

### 4. Loss fix
**Decided:** match Johnson et al. now. The style term sums squared Gram differences over the
C×C entries (today it averages them), layer weights are applied, and a total-variation term is
added. Against my recommendation, which was to apply the weights only.
Consequence, measured on the Wave against four frog photos at 256 px with the `standard`
preset: the style term grows about 85,000× (0.00042 → 35.8), and the per-layer share moves from
conv3_1 (57%) toward conv3_1 and conv4_1 together (92%), with conv1_1 falling from 10% to 0.5%.
**Decided (weights):** every curriculum style weight is divided by that factor as an untested
starting point (4e4 → 0.5, 8e4 → 1.0, 2e5 → 2.5); total variation starts at 1e-6, a value
recalled from the paper's reference code and not verified. The old reduction is not kept as an
option. The 23 archived models are marked `"loss": "legacy-mean"` in their `model.json`; they
are not comparable with new runs.

### 5. `perceptual_loss` shape
**Parked.** The function already takes any generator's output and `diffuser/` uses it. The
`PerceptualLoss` object (own VGG, component terms, cached style features) waits until diffusion
work resumes. The function gains `tv_weight` only.

### 6. Models
**Decided:** the default set is `mini_kanagawa`, `mini_colors1`, `mini_colors2` and
`high_starry_night`; only these four get a `model.json` now, so only they appear in `list` and
the UI. When training resumes, the first run is a long Kanagawa reference (small model, about
20,000 steps, VOC content, new loss), judged by eye on the frog set; it tunes the new style
weight. Then Starry Night and `lines`. Full list: `dev/agents/backlog.md`. Not chosen: building
the benchmark first, retraining the defaults first.

## Change list (applied on worker branches; nothing merged without the owner)
Slice A, training side (threads 1, 4, 6; spot items a, b):
- `style_transfer/loss.py`: sum reduction, layer weights applied, `tv_weight`.
- `style_transfer/config/layer_presets.py`, `config/styles/kanagawa.py`: Kanagawa preset moved
  into `LAYER_PRESETS`; legacy converter removed if unused.
- `style_transfer/config/curricula.py`: rescaled style weights, `tv_weight`.
- `style_transfer/train.py`, `utils/metrics.py`: output under `artifacts/models/<name>/`,
  `model.json` written on success, final path returned.
- `style_transfer/config/styles/*`: add `kanagawa_long`.
- tests: hand-computed loss, weights have an effect, every experiment resolves, `model.json`.
Slice B, inference, CLI, packaging (threads 1, 2; spot items c, d, e):
- `style_transfer/inference.py`: pad/crop, RGB, `--max-side`, load by `model.json`.
- new `style_transfer/cli.py`, `style_transfer/__main__.py`, `pyproject.toml`.
- delete `style_transfer/apply_style.py`, `style_transfer/build.py`, `run_experiment.py`,
  `requirements.txt`; `run_diffusion_experiment.py` updated for the lookup change.
- `README.md` rewritten; public links removed.
- one-off `scripts/write_model_json.py` for the four default models.
Slice C, local server (thread 3):
- new `style_transfer_web/` (`app.py`, `static/`), tests.
Not in a slice: spot item f (visualizer double normalization), left as a follow-up.

## Changes applied
On branch `rework` (`fc5b4fa`, four commits on `d52e6f0`; 33 files, +1752/−648). 108 tests pass.
- Slice A `971efdc`: loss, presets, rescaled curricula, `kanagawa_long`, `train_model` writing
  `model.json` under the models root. Also rescaled the `diffuser/` default style weight
  (1e5 → 1.2).
- Slice B `223a308`: `stylize_image` pad/crop/RGB/`max_side`, `list_models`,
  `load_named_model`, `style_transfer/cli.py`, `pyproject.toml`, `scripts/write_model_json.py`,
  README; deleted `apply_style.py`, `build.py`, `run_experiment.py`, `requirements.txt`.
- Slice C `35a3a56`: `style_transfer_web/` (85-line FastAPI app, static page, 10 tests).
- Integration `fc5b4fa`: `find_checkpoint` removed from the library; the CLI `serve` test fixed
  (it started a real server once the web package existed) and a second one added.
- Outside git: `model.json` written for the four default models in `artifacts/models/`.
Checked by running, from the main checkout against the branch's code: `list`; `stylize` on a
directory and a file (271×186 and 601×481 outputs match their inputs); `train --experiment
kanagawa_dry_run` at 32 px into a temp models dir (520 steps, `model.json` written, loadable);
`serve` bound to `127.0.0.1` only, page and `/api/models` served, two models applied to one
upload, 404 and 400 on bad requests.
Not checked: `pip install -e .` and the `style-transfer` console script (run as
`python -m style_transfer` instead); the page in a real browser; any training under the new loss
beyond the 32 px smoke.

## Follow-ups (outside the material, not applied)
- `CLAUDE.md` (the owner's, gitignored): the `demo/` paragraph and goal 6 describe a deployed,
  ignored demo; update once tension 3 is settled.
- Takedown of Firebase Hosting and the Cloud Run service: commands given to the owner.
- Optional Codex second pass on `models-to-train.md` after 20:34.
- `style_transfer/utils/visualize.py:315` normalizes before a VGG that normalizes again; fix
  when the visualizer is next used, and discard its activation cache.
- `README.md` example images were made with legacy-loss models.
- `feature_blender` uses the raw-feature branch, whose reduction did not change, but the
  curricula style weights were all divided by 85,000; any raw-feature run needs its own weights.
  No registered experiment uses it.
- `.gitignore` is untracked (it ignores itself), so worktrees and fresh clones have none and
  `__pycache__/` shows as untracked there. Track a root `.gitignore`.
- Training still writes `metrics_config.json`, which now overlaps `model.json`.
- `diffuser/` style-weight default was rescaled the same way and is equally untested.
- The page lost its "created by J. Blackburn" link (it pointed at a hosted site).
- `artifacts/models/` holds 4.2 GB of per-epoch checkpoints; the owner decides whether to prune.
