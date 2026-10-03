# Brainstorm: rework (2026-10-03)

Input for `/workshop rework`. Options and recommendations only; nothing here is decided.

Source: four angles, written against `dfbac70`. Three notes are from Codex (`gpt-6-astra`,
effort `max`); each was complete on disk when the account hit its usage limit, so none carries
its closing summary. `models-to-train.md` is by Claude. A Codex second pass on that angle is
possible after the limit resets (20:34).

| Note | Angle | Author |
|---|---|---|
| [code-review.md](code-review.md) | 8 ranked findings, target layout, refactor order | Codex |
| [cli-and-local-server.md](cli-and-local-server.md) | command surface, model registry, what of `demo/` survives, 8 slices | Codex |
| [models-to-train.md](models-to-train.md) | inventory of 23 trained models, default set, 8-model shortlist | Claude |
| [future-experiments.md](future-experiments.md) | 12 experiments (E1–E12), ranked twice | Codex |

## Checked against the code

Claims I re-ran or re-read before relying on them:
- Layer weights are ignored: the style loss is identical under `standard`, `standard_weighted`
  and `standard_x_shallow` (same value to 16 digits). `style_transfer/loss.py:29`.
- `kanagawa_custom_layers` cannot resolve its preset; the other 26 experiments do.
- VGG feature `'0'` is post-ReLU (minimum 0.0), because the next layer's in-place ReLU
  overwrites the saved tensor. `style_transfer/feature_extractors/vgg.py:32-35`.
- A 65×97 input gives a 68×100 output.
- Checkpoint lookup differs by entry point: `train.py:105` writes `models/`,
  `run_experiment.py:41` reads `models/`, `inference.py:59` prefers `artifacts/models/`.
- The visualizer normalizes twice (`utils/visualize.py:315`, `vgg.py:28`).
- The demo binds `0.0.0.0` (`demo/core/config.py:18`); the web UI points at the public API for
  any host other than `localhost` (`demo/web_ui/public/script.js:2`).
- 21 tests pass.

Not checked: the browser repro of the stale-upload bug, the 4×4 and RGBA/grayscale failures, and
the paper citations in `future-experiments.md` (Codex says some are from memory).

## Where the notes agree

- Training/inference value ranges are already consistent, and inference is already shared by
  every caller. The duplication left is in commands, checkpoint lookup and device selection.
- One argparse command with `stylize`, `train`, list; delete `apply_style.py`, `build.py` and
  the interactive `run_experiment.py`.
- A trained model must carry its own architecture; inference must stop reading model size from
  the current experiment config (`inference.py:111`).
- Pad to a multiple of four and crop back inside `stylize_image`, with tests for odd sizes.
- Fix the ignored layer weights before any further layer-weight training.
- Add `pyproject.toml` with optional dependency groups; core inference should not pull in Dash,
  ONNX, notebooks or diffusers.
- Keep `diffuser/` in the repo as parked research, outside the supported CLI.
- Keep the Dash visualizer separate and optional; fix its double normalization.

## Where they differ (workshop material)

1. **Model metadata.** A `model.json` manifest beside each checkpoint with a registry module
   and a `~/.style-transfer/models` root (`cli-and-local-server.md`), or metadata inside the
   checkpoint file and lookup staying in the repo (`code-review.md` finding 4).
2. **Size of the CLI.** Thirteen commands including `models fetch`, `models import`,
   `export-onnx`, `visualize` (`cli-and-local-server.md`), or three (`code-review.md`). No
   download source for weights exists, so `fetch` has nothing to fetch yet.
3. **Where the local server lives.** A new tracked `style_transfer_web/` package, or a slimmed
   `demo/` run as `python -m demo` and still untracked.
4. **How to fix the loss.** Apply the weights and keep the current reductions as the "legacy"
   objective, or move to the paper's reductions and add total variation now. The second changes
   every style weight and makes old and new models incomparable.
5. **`perceptual_loss` shape.** A `PerceptualLoss` object owning its VGG (removes the
   process-global `_vgg_model`), with `criterion(generated, content, style=...)`. `CLAUDE.md`
   goal 5 asks for `perceptual_loss(image_a, image_b)`; a two-argument form cannot express
   separate content and style targets.
6. **Train now or measure first.** `models-to-train.md` says one long Kanagawa run answers the
   biggest open question for minutes of compute. `future-experiments.md` puts a fixed benchmark
   (E1) ahead of any training, so results can be compared.

## Recommendation

Do the rework in this order, which is also the order the tensions should be settled:

1. Loss and config correctness: apply layer weights, register the Kanagawa preset, resolve and
   validate every experiment in a test. Small, and it invalidates past comparisons until done.
2. Inference contract: pad/crop, RGB and EXIF handling, self-describing model files.
3. One CLI with a small surface (`list`, `stylize`, `train`, `serve`), packaging, README.
4. Local server: one FastAPI process on `127.0.0.1` serving the existing static UI with a
   single request per stylize; no Node, no upload store.
5. Then train, starting with the long Kanagawa reference run.

The larger registry (`fetch`, catalog, checksums) and the `PerceptualLoss` object are worth
having but neither blocks the "easy to use from the CLI and a local server" goal; I would park
both until the first three steps are in.
