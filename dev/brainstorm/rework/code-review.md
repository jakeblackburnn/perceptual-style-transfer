1. **P1 — Several named loss experiments do not implement their advertised objective. Size: M.**

   **Observed:** `style_layer_weights` is read but never multiplied into either the Gram or raw-feature contribution (`style_transfer/loss.py:29`, `style_transfer/loss.py:49`, `style_transfer/loss.py:57`). Consequently, `standard_weighted` and `standard_x_shallow` have exactly the same objective as `standard`, given identical tensors; their only preset differences are the unused weights (`style_transfer/config/layer_presets.py:6`, `style_transfer/config/layer_presets.py:14`, `style_transfer/config/layer_presets.py:22`). A read-only probe changed all five weights to zero: the style-only loss remained **256.4427795410156**. This invalidates comparisons attributed to those weights at this revision; it does not prove which code trained the archived models.

   **Observed:** `kanagawa_custom_layers` selects `kanagawa_optimized`, defined only in the style-local dictionary (`style_transfer/config/styles/kanagawa.py:7`, `style_transfer/config/styles/kanagawa.py:155`). Training calls `initialize_vgg` without that dictionary (`style_transfer/train.py:101`), although the resolver requires it for local names (`style_transfer/config/layer_presets.py:78`). Resolving all 27 registered experiments produced one failure: this experiment raises `KeyError` before training. The preset exists; its wiring is missing.

   **Numerical comparison:** Johnson et al. define content distance as squared feature error divided by CHW, normalize Gram matrices by CHW, sum squared Gram-entry errors for style, and add total variation regularization. Their experiments use VGG16 and specified ReLU layers. [Paper, equations 2–5 and sections 3.3/4.1](https://arxiv.org/pdf/1603.08155).

   The content MSE and per-image Gram normalization here match those definitions, with an additional average over batch (`style_transfer/loss.py:6`, `style_transfer/loss.py:37`). The style reduction averages over C×C instead of summing, adding a layer-dependent factor of 1/C² (`style_transfer/loss.py:56`). One global style weight cannot undo that relative layer scaling. The returned total contains no TV term (`style_transfer/loss.py:59`). VGG19 and different feature layers are explicit project variants, not inherently bugs (`style_transfer/feature_extractors/vgg.py:18`, `style_transfer/config/layer_presets.py:8`). Describe this as Johnson-inspired, not an exact reproduction.

   **Observed extension:** Multi-style batches average every generated/style pairing (`style_transfer/loss.py:54`). Algebraically, their gradient attracts each generated Gram toward the mean target Gram; it does not give the generator independently selectable styles. The raw-feature branch instead compares spatial activations and requires compatible feature-map dimensions (`style_transfer/loss.py:43`). It should not be described as Gram style loss.

   **Recommendation:** Apply and validate layer weights in both branches; resolve every registered preset before constructing VGG. Keep the current Gram-mean reduction explicitly identified as the legacy objective while fixing the ignored weights. Compare a separately named Frobenius-sum objective and optional TV in later experiments; do not silently reuse historical weight values or experiment names after changing the objective. Report content, style and TV separately.

   **Alternative:** Switch all new training immediately to the paper's reductions. This is simpler mathematically but changes layer balance and requires retuning; it is more than a behavior-preserving refactor. TV strength for this implementation remains unverified and should be an experiment, with zero preserving existing behavior.

2. **P1 — The image contract preserves value range, but not arbitrary dimensions. Size: M.**

   **Observed, already correct:** Training converts to RGB, resizes, and applies `ToTensor`; inference applies `ToTensor`; the generator maps tanh to [0,1]; ImageNet normalization lives inside VGG (`style_transfer/dataset.py:61`, `style_transfer/dataset.py:76`, `style_transfer/inference.py:17`, `style_transfer/models.py:98`, `style_transfer/feature_extractors/vgg.py:24`). There is no current inference ×255 bug in that path. Keep this convention.

   **Observed:** Two stride-2 convolutions followed by two ×2 upsamplers produce `4 * ceil(H/4)` by `4 * ceil(W/4)`, with no output crop (`style_transfer/models.py:53`, `style_transfer/models.py:58`, `style_transfer/models.py:71`, `style_transfer/models.py:77`, `style_transfer/inference.py:40`). Probes returned 64×68 for a tensor with H×W=63×65. Loading the archived `mini_kanagawa` through the real loader also turned a PIL image of W×H=65×63 into 68×64. A 4×4 input failed in instance normalization; tiny images need an explicit policy, too (`style_transfer/models.py:59`).

   **Observed:** The public PIL helper does not convert modes, although the CLI does (`style_transfer/inference.py:38`, `style_transfer/inference.py:129`). Direct grayscale and RGBA calls both failed with channel-count errors. Training stretches every image to a square; inference keeps native geometry (`style_transfer/dataset.py:62`, `style_transfer/inference.py:17`). The parity test uses a square already at the requested training resolution, so it verifies range conversion, not resizing policy (`tests/test_preprocessing_parity.py:15`, `tests/test_preprocessing_parity.py:28`).

   **Recommendation:** Make `stylize_image` own RGB conversion, EXIF orientation, dimension validation, padding to a multiple of four, and cropping back to the original dimensions. Keep the checkpoint architecture and parameter names unchanged. Define a minimum-size policy explicitly. Require aligned training resolutions until the training path has a deliberate crop policy. Preserve native-resolution inference; document the current square training resize and separately compare aspect-preserving crops if desired.

   **Alternative:** Crop the existing oversized output without padding input. It changes less, but boundary handling is less explicit. Resizing every input to a fixed square would undermine the owner's arbitrary-dimension goal and is not recommended.

3. **P2 — Finish inference consolidation at the command and checkpoint boundaries. Size: M.**

   **Observed:** Numerical inference is already centralized: `apply_style.py` and `run_experiment.py` import the library loader and PIL helper (`style_transfer/apply_style.py:6`, `run_experiment.py:9`). `build.py` only selects a hardcoded training experiment and calls `train_model`; it is not another inference implementation (`style_transfer/build.py:8`, `style_transfer/build.py:23`). The export utility is a remaining independent model loader with a different accepted checkpoint format (`style_transfer/utils/convert_to_onnx.py:19`, `style_transfer/utils/convert_to_onnx.py:23`).

   **Observed:** The CLI searches `artifacts/models` before `models`, while the interactive runner and old apply script search only `models` (`style_transfer/inference.py:59`, `run_experiment.py:41`, `style_transfer/apply_style.py:59`). With both copies present, callers can select different weights under the same experiment name. The diffusion runner also uses the archive-first lookup immediately after training writes under `models` (`run_diffusion_experiment.py:31`, `run_diffusion_experiment.py:170`, `diffuser/train.py:88`). A training call should return its exact output path, not rediscover it by name.

   **Observed:** Both runners require interactive input, call macOS `open`, and finish with `os._exit(0)` (`run_experiment.py:140`, `run_experiment.py:86`, `run_experiment.py:258`, `run_diffusion_experiment.py:91`, `run_diffusion_experiment.py:71`, `run_diffusion_experiment.py:206`). The apply script additionally ignores CUDA in device selection and uses an older content-root convention (`style_transfer/apply_style.py:17`, `style_transfer/apply_style.py:20`). All three inference file loops force stem-based JPEG names, so `photo.png` and `photo.jpg` collide (`style_transfer/inference.py:131`, `run_experiment.py:78`, `style_transfer/apply_style.py:83`).

   **Recommendation:** One argparse CLI with `list`, `stylize`, and `train`; `python -m style_transfer` and an installed console command should dispatch to the same parser. Keep `inference.py` as the numerical API. Add one small file helper for decode → `stylize_image` → save, and let a directory loop reuse it while loading the model once. Make output paths/formats explicit, detect collisions, and return ordinary nonzero exit statuses for failures.

   **Delete after migration:** `style_transfer/apply_style.py`, `style_transfer/build.py`, and the interactive implementation of `run_experiment.py`; optionally retain a short forwarding shim for one transition. Route ONNX loading through the same loader if export survives. Remove the unused model factory and unused alternate final-save function after a caller check (`style_transfer/models.py:102`, `style_transfer/train.py:11`, `style_transfer/utils/metrics.py:89`); their tracked references were checked during this review. Preserve checkpoint-comparison behavior only if the owner still uses it, as an explicit CLI option rather than a second inference script.

   **Local UI boundary:** The ignored demo already delegates model loading and forward inference to the library (`/Users/jackblackburn/code/main/audio_video/style-transfer/py/demo/core/model_manager.py:143`, `/Users/jackblackburn/code/main/audio_video/style-transfer/py/demo/core/inference.py:80`). Preserve that direction. A separate `python -m demo` may serve localhost; the core CLI/package must not import `demo`. Website removal is already decided and is outside this notes-only review.

4. **P2 — Resolve configuration once and save enough information to reproduce a run. Size: L.**

   **Observed:** Model-size definitions remain inline in the constructor, while curricula expose learning rate, resolution, epochs, style weight and batch sizes (`style_transfer/models.py:9`, `style_transfer/config/curricula.py:19`). Content weight is fixed by the loss-call default; worker count and persistence are hardcoded (`style_transfer/train.py:36`, `style_transfer/loss.py:14`, `style_transfer/train.py:143`). Adam's initial rate is another literal, although each stage overwrites it (`style_transfer/train.py:112`, `style_transfer/train.py:55`). Make genuinely varied choices configurable; fixed architectural details need not all become knobs.

   **Observed:** Small style files are direct experiment dictionaries; Kanagawa adds a dataset registry, local preset registry and factory, yet all are still explicitly imported—there is no auto-discovery (`style_transfer/config/styles/colors2.py:11`, `style_transfer/config/styles/kanagawa.py:24`, `style_transfer/config/styles/kanagawa.py:52`, `style_transfer/config/styles/__init__.py:2`). The local preset indirection is already broken in finding 1. Shared curriculum lists are passed through by reference (`style_transfer/config/styles/colors2.py:17`, `style_transfer/config/styles/kanagawa.py:59`); future in-place overrides could affect sibling experiments. That is a risk inferred from aliasing, not a demonstrated mutation in today's trainer.

   **Observed:** Fractions are sampled without a supplied seed and can floor to zero after the nonempty check; invalid fractions merely print and retain the entire dataset (`style_transfer/dataset.py:25`, `style_transfer/dataset.py:29`). A probe using the five tracked example images and fraction 0.01 produced a dataset of length zero. Each curriculum stage constructs new datasets, hence new random subsets (`style_transfer/train.py:116`, `style_transfer/train.py:121`). A missing stage learning rate becomes `None`, not a validated error (`style_transfer/train.py:55`).

   **Observed:** Final weights are a bare state dict, while intermediate checkpoints contain epoch, stage, weights and optimizer only (`style_transfer/train.py:173`, `style_transfer/utils/metrics.py:79`). Metadata stores stages and dataset counts/names, omitting the resolved architecture, VGG preset/weights, preprocessing, random seed and selected paths; single-style metadata is empty (`style_transfer/utils/metrics.py:42`, `style_transfer/dataset.py:101`). Runs reuse names and overwrite CSV/checkpoint/final paths (`style_transfer/train.py:105`, `style_transfer/utils/metrics.py:69`, `style_transfer/utils/metrics.py:85`). The archived mini-Kanagawa metadata inspected here also has an empty style object (`/Users/jackblackburn/code/main/audio_video/style-transfer/py/artifacts/models/mini_kanagawa/metrics_config.json:19`).

   **Observed:** Every stage's two persistent loaders are retained until training ends, and cleanup deletes private iterator attributes only on the success path (`style_transfer/train.py:143`, `style_transfer/train.py:151`, `style_transfer/train.py:156`, `style_transfer/train.py:177`). This allows workers to accumulate with the curriculum; no worker-count benchmark was run. The runners' forced exit masks lifecycle problems rather than exposing their cause (`run_experiment.py:258`).

   **Recommendation:** Keep explicit Python configs and one explicit registry. Move size presets into `config/model_presets.py`; use one `resolve_experiment()` to copy dictionaries, fill defaults and validate names, weights, fractions, learning rates, nonempty stages and paths. Flatten the redundant `curriculum: {stages: ...}` wrapper to `stages`. Put Kanagawa's custom loss preset in the same registry as every other preset. Remove the legacy converter and its unused extractor-selection promise instead of building a discovery/plugin framework (`style_transfer/config/layer_presets.py:100`). Dataclasses are an alternative if they simplify validation; a YAML/Hydra migration has no demonstrated benefit here.

   **Recommendation:** Save a resolved run JSON before training, including seed, stable sampled file list, style source, architecture, loss version/layers/weights and preprocessing. Select the content subset once per run unless resampling is explicitly requested. Give each run a chosen output directory and reject accidental reuse. New checkpoint envelopes should include architecture and format version; preserve loading of old bare and `model_state_dict` checkpoints without rewriting archived weights. Return the final path from training. Start with configurable `num_workers=0`, then use scoped loaders with reliable cleanup when workers are enabled. This removes private-attribute deletion and forced process exit.

   **Alternative:** Add only JSON beside existing `.pth` files. This is smaller, but a copied checkpoint would still lose its architecture identity. Full resumable training, a model registry service and automatic checkpoint migration are not needed for the first rework.

5. **P2 — Perceptual loss is generator-independent already; remove its implicit VGG state and misleading feature inspection. Size: M.**

   **Observed:** `perceptual_loss` already accepts generated/content/style tensors and never runs a generator (`style_transfer/loss.py:11`). Diffusion genuinely uses it today: it masks generated/content elements by timestep, leaves the style batch independent, and calls the same function (`diffuser/loss.py:30`, `diffuser/loss.py:34`). The empty-mask branch remains attached to the generated graph (`diffuser/loss.py:32`). This is useful reuse, not a future-only possibility.

   **Observed:** The dependency is a process-global extractor. Calling before initialization raises, and later initialization replaces the state for every caller (`style_transfer/loss.py:24`, `style_transfer/feature_extractors/vgg.py:42`, `style_transfer/feature_extractors/vgg.py:54`, `style_transfer/feature_extractors/vgg.py:58`). All three image groups run through VGG on every batch, including the unchanged single-style target (`style_transfer/loss.py:33`, `style_transfer/train.py:123`). VGG parameters are correctly frozen, while the generated input still has a gradient path (`style_transfer/feature_extractors/vgg.py:19`, `style_transfer/loss.py:33`). Do not put the generated-image feature pass under `no_grad`.

   **Observed:** Feature entries are stored by reference as the full VGG sequence runs (`style_transfer/feature_extractors/vgg.py:32`). With the installed torchvision, the following in-place ReLU changes the saved convolution output. The probe found raw conv0 minimum −9.8827 but returned feature `'0'` minimum 0, exactly equal to `relu(conv0)`. The preset's convolution labels therefore conceal post-ReLU semantics (`style_transfer/config/layer_presets.py:8`). The activation hooks also detach without cloning, preserving that aliasing (`style_transfer/utils/activation_extractor.py:10`). This is a labeling/contract trap; blindly cloning would change the training objective.

   **Observed:** The visualization transform applies ImageNet normalization before invoking VGG, which normalizes again (`style_transfer/utils/visualize.py:312`, `style_transfer/utils/visualize.py:292`, `style_transfer/feature_extractors/vgg.py:25`). Its activation cache uses only parent-directory and image-stem names, without preprocessing/extractor identity (`style_transfer/utils/visualize.py:327`, `style_transfer/utils/visualize.py:333`). Removing the extra normalization requires invalidating those cached displays. VGG also evaluates through its final pool even when the requested features are earlier (`style_transfer/feature_extractors/vgg.py:32`); the existing tests accommodate this with 32px inputs (`tests/test_diffuser_loss.py:11`).

   **Recommendation:** A small `PerceptualLoss` module should own an explicit frozen VGG instance and resolved loss config. This single object replaces `initialize_vgg`, `get_vgg_model`, the global variable and manual global cleanup; it is not a generator framework. Return component terms plus total. Normalize once using registered buffers, cache detached style features/Grams per training stage, and stop extraction after the last requested activation. Make layer names explicit ReLU names to preserve today's effective semantics; treat actual pre-ReLU features as a new experiment.

   **Two-image API option, recommended:** `criterion(a, b)` means content-feature distance; `criterion(generated, content, style=style_target)` adds the explicit style objective. A two-argument function cannot infer separate content and style targets. The alternative is to use `b` for both targets, but that hides a different question. Diffusion should pass its denoised output and both explicit targets to the same object. Use injected tiny feature extractors for numerical unit tests, plus one real-VGG integration test.

   **Scope option:** Keep the Dash visualizer only if it informs model decisions. If kept, use the same input contract and explicit activation selection, remove hooks after use, and invalidate the old cache. Otherwise park it as a research utility and remove Dash/Plotly from the default installation.

6. **P2 — Park diffusion as research; its tests do not establish a useful stylization objective. Size: S to park, L+ to redesign.**

   **Observed strengths:** It uses a content-conditioned UNet, explicit [0,1]↔[−1,1] conversions, library schedulers, gradient clipping and EMA (`diffuser/models.py:29`, `diffuser/models.py:40`, `diffuser/schedule.py:16`, `diffuser/schedule.py:24`, `diffuser/train.py:61`). Its loss integration is a valuable second consumer of the perceptual component (finding 5).

   **Observed objective:** Noise is added to the original content, the same clean content is supplied as conditioning, and the target is that added noise; style enters only through an auxiliary perceptual penalty (`diffuser/train.py:42`, `diffuser/train.py:46`, `diffuser/train.py:48`, `diffuser/train.py:57`). **Inference:** Given clean content `c`, the ideal denoising answer is analytically `(x_t - sqrt(alpha_bar)*c) / sqrt(1-alpha_bar)`. Thus the noise objective rewards reconstruction of the original content, while the style objective asks for changes. This explains a tension worth testing; it does not prove that every weighting produces bad pictures. A tensor-only probe recovered the noise from that formula to about 2.4e-7 maximum error.

   **Observed concrete gaps:** `prediction_type` is configurable, but training always targets epsilon and `predict_x0` always uses the epsilon formula (`diffuser/config/schema.py:33`, `diffuser/train.py:49`, `diffuser/schedule.py:54`). Reject non-epsilon configurations unless implemented. Divisibility is checked only for configured training size; inference accepts unchecked native image dimensions (`diffuser/models.py:21`, `diffuser/inference.py:51`). A 15×17 input failed a skip-concatenation shape check; 16×18 worked with the small model. Final saving writes only EMA weights after all epochs; there are no epoch checkpoints or EMA/optimizer resume state (`diffuser/train.py:123`, `diffuser/train.py:139`). The notebook's claim that it checkpoints each epoch is stale (`notebooks/diffusion_perceptual_loss_experiment.ipynb:84`).

   **Recommendation:** Park it in this repository, outside the supported CLI/model catalogue, with an optional research dependency group and an explicit full research test command. Keeping its source and tests costs less than a premature separate package, and exercises the shared loss boundary. Leave the existing diffusion runner clearly marked experimental rather than building a second supported CLI. Limit immediate changes to preserving compatibility with the new loss API and fixing misleading contracts.

   **Alternatives:** Split only when diffusion has an independently maintained workflow and released interface; remove it only if the owner no longer wants the experiment. Before promoting it, compare the feed-forward baseline, identity output and a diffusion variant on fixed held-out content and seeds. A candidate future experiment is denoising stylized teacher targets while conditioning on content, which removes the direct identity-target tension; a frozen pretrained denoiser with perceptual guidance is a larger, different experiment. Neither is a recommendation to start training during this rework.

7. **P2 — The 21 green tests mainly protect shapes and glue, leaving the highest-risk research semantics unchecked. Size: M across the refactor.**

   **Observed inventory:** The complete suite passed: **21 tests in 1.86 s**, using the supplied environment. These are the protections actually asserted:

   | Tests | Count | Protection and limitation |
   | --- | ---: | --- |
   | `tests/test_loss.py:6`, `tests/test_loss.py:20` | 2 | One known Gram result and symmetry; never calls `perceptual_loss`. |
   | `tests/test_models.py:6`, `tests/test_models.py:15`, `tests/test_models.py:26` | 3 | All size presets at 64×64, unit-range output, invalid preset; no odd/rectangular/tiny inputs. |
   | `tests/test_preprocessing_parity.py:24`, `tests/test_preprocessing_parity.py:37` | 2 | Already-sized RGB preprocessing parity and white-image range. |
   | `tests/test_diffuser_model.py:8`, `tests/test_diffuser_model.py:23` | 2 | Small UNet shape and constructor divisibility rejection. |
   | `tests/test_diffuser_schedule.py:7`, `tests/test_diffuser_schedule.py:25`, `tests/test_diffuser_schedule.py:35` | 3 | Perfect-epsilon reconstruction, monotone schedule and cosine default. |
   | `tests/test_diffuser_preprocessing_parity.py:16`, `tests/test_diffuser_preprocessing_parity.py:21`, `tests/test_diffuser_preprocessing_parity.py:27` | 3 | Range conversion, round trip and clamp. |
   | `tests/test_diffuser_loss.py:16`, `tests/test_diffuser_loss.py:29`, `tests/test_diffuser_loss.py:41` | 3 | Empty/all/mixed timestep gating; nonempty results are compared to the same loss implementation. |
   | `tests/test_diffuser_sampling.py:29`, `tests/test_diffuser_sampling.py:49` | 2 | One seeded tiny sampling run stays within chosen bounds; final range is also forced by the conversion clamp (`diffuser/schedule.py:21`). |
   | `tests/test_diffuser_train_smoke.py:49` | 1 | Three tiny training steps have finite scalar losses; no convergence, parameter-change or nonzero perceptual-gradient assertion (`tests/test_diffuser_train_smoke.py:72`). |

   **Observed limitations:** Real VGG initializes at collection time in two modules, making the suite depend on pretrained weights being available and on shared global state (`tests/test_diffuser_loss.py:7`, `tests/test_diffuser_train_smoke.py:31`). The smoke test can pass when no perceptual timesteps qualify because zero is finite (`diffuser/loss.py:32`, `tests/test_diffuser_train_smoke.py:77`). The sampling test bounds `x_t`, while the scheduler is configured to clip predicted clean samples; the test's arbitrary ±1.5 bound should not be treated as a general DDIM guarantee (`tests/test_diffuser_sampling.py:43`, `diffuser/schedule.py:39`).

   **Recommendation, in priority order:** Test hand-computed weighted content/Gram reductions and generated-image gradients; resolve all registered experiments; verify exact odd/rectangular output sizes and RGB/EXIF behavior; round-trip both existing checkpoint formats and the proposed envelope; test a complete CLI image-to-file operation from outside the repository; reject zero-sized datasets/invalid stages; finally test training output metadata and ordinary shutdown. Pair each regression with its fix rather than committing a red suite. Use one small real-VGG integration test to verify feature names and normalization; deterministic fake features should carry most objective tests.

   **Alternative:** Add more random end-to-end smoke tests. They are easy to write but would still not catch ignored weights or double normalization. If diffusion remains parked, preserve its 14 tests in the full environment rather than silently deleting coverage or accepting import failures in a minimal installation.

8. **P2 — Installation is an experiment environment, not a defined CLI product. Size: S–M.**

   **Observed:** `requirements.txt` lists unversioned core, visualization, ONNX, testing, notebook and diffusion dependencies together (`requirements.txt:1`, `requirements.txt:5`, `requirements.txt:8`, `requirements.txt:11`). The README installs only torch, torchvision, Pillow, Dash and Plotly, so it does not install the dependencies needed by the diffusion tests or export utility (`README.md:29`, `tests/test_diffuser_model.py:5`, `style_transfer/utils/convert_to_onnx.py:47`). The tracked-file inventory contains no `pyproject.toml`; an absent file has no source line to cite. Pytest puts the repository root on the import path, so the green result does not verify an installable package (`pytest.ini:2`).

   **Recommendation:** Add a minimal `pyproject.toml` with explicit package discovery, supported Python range and one console entry point. Keep the flat source layout. Separate `dev`, `research`, `viz`, `export` and optional local `ui` dependencies from core inference. Use one checked-in lock/constraints snapshot for the tested environment and compatible dependency bounds in package metadata; make README installation use that source of truth. The inspected environment was Python 3.14.3, torch 2.13.0, torchvision 0.28.0, Pillow 12.3.0, diffusers 0.39.0 and pytest 9.1.1. This is a working local baseline, not evidence of cross-platform support or a proposed minimum version.

   **Utility option:** Park ONNX export unless a current consumer needs it. Its independent loader, fixed preset/path and schema-only verification add a separate maintenance surface (`style_transfer/utils/convert_to_onnx.py:9`, `style_transfer/utils/convert_to_onnx.py:76`, `style_transfer/utils/convert_to_onnx.py:64`). If retained, require shared loading and an optional numerical ONNX-runtime comparison at multiple dimensions. No export was executed during this review, so compatibility with the installed exporter is unverified.

   **Alternative:** Only pin `requirements.txt` and fix the README. This is a useful interim patch, but leaves installed commands, optional dependencies and outside-repository imports undefined. The README's public-site links should be replaced as part of the already-decided local transition (`README.md:3`, `README.md:19`).

**Proposed target tree — option for the workshop, not an approved migration**

```text
pyproject.toml                 # package metadata, CLI, optional dependencies
requirements.lock             # or one equivalent chosen lock/constraints format
README.md
LICENSE
pytest.ini
style_transfer/
  __init__.py
  __main__.py                  # delegates to cli.main
  cli.py                      # list / stylize / train; no demo imports
  inference.py                # load, PIL inference, one image-file adapter
  checkpoints.py              # versioned save/load; retains old formats
  models.py
  architectures/residual_block.py
  dataset.py
  train.py
  loss.py                     # explicit PerceptualLoss and component math
  feature_extractors/vgg.py
  config/
    __init__.py               # explicit experiment registry
    resolve.py                # copying/defaults/validation, no framework
    model_presets.py
    layer_presets.py
    curricula.py
    styles/*.py               # one consistent experiment shape
  utils/metrics.py            # run metadata + component CSV; no weight loading
  utils/visualize.py          # optional, only if retained
  utils/activation_extractor.py # optional, if hooks still earn their place
  utils/convert_to_onnx.py    # optional, only if retained
demo/                         # selected local-only wrapper/static UI, if adopted
diffuser/                     # parked research consumer; optional dependencies
run_diffusion_experiment.py   # parked runner, not a supported product command
notebooks/                    # research-only; fix overstated claims
tests/
  test_*.py                   # numerical, config, checkpoint, CLI contracts
  research/                   # relocated diffusion tests, explicitly runnable
artifacts/examples/           # curated examples, not ad hoc run outputs
dev/brainstorm/rework/
```

`checkpoints.py` earns its extra file by replacing save/load format decisions in inference, metrics, training and export. `cli.py` replaces the two ad hoc feed-forward launchers and interactive runner; `resolve.py` replaces the legacy preset adapter and implicit defaults. Keep the current residual block rather than reorganizing working model internals for appearance. Parking diffusion in place avoids import churn and a new package release boundary.

Untracked runtime data would live under an explicit artifact root: images, models/run IDs and generated outputs. The notebook currently saves quick-test images under `outputs/quick_test` (`notebooks/diffusion_perceptual_loss_experiment.ipynb:147`); the tracked inventory includes such generated JPEGs. Recommendation: retain selected examples in `artifacts/examples`, remove incidental run outputs from tracking after deciding what is worth preserving, and add ignore rules. The existing archived weights remain usable through compatibility loading; no bulk migration is required.

**Ordered refactor option — each numbered step should finish green**

1. **Establish the unchanged baseline and package boundary.** Add minimal packaging and align installation docs; make optional research tests an explicit suite while continuing to run all 21 in the full environment. Verify installed import/CLI scaffolding from a temporary directory without loading VGG or creating a model. This removes ambiguity about where paths/imports resolve before changing numerics.
2. **Make loss ownership explicit without changing its values.** Inject the extractor, replace the global state, and adapt both trainers. Add tiny-feature tests for existing reductions and one real-VGG feature/normalization integration check. Preserve the current effective ReLU activations. Existing diffusion gate assertions must still pass.
3. **Fix loss/config correctness together.** Apply layer weights, resolve the custom Kanagawa preset, validate all registered experiments, and add independent expected-value tests in the same commit. Record the corrected objective identity. Keep Gram reduction unchanged in this step; a paper-sum/TV variant requires a separate workshop choice and tests.
4. **Fix the image boundary.** Add odd, rectangular, mode/orientation and tiny-input regressions with padding/cropping and validation. Assert byte-to-tensor range parity separately from spatial policy. Load the archived mini-Kanagawa model read-only to verify compatibility; no training is needed.
5. **Unify checkpoint and command paths.** Add the envelope plus both legacy loaders, explicit model/output locations, collision handling and the complete CLI file test. Have training return a checkpoint path. Remove the old apply/build scripts and interactive runner after their supported actions pass through the new CLI. The local UI remains a downstream caller.
6. **Make experiments repeatable and cleanup ordinary.** Resolve/copy config once; validate data before VGG allocation; seed and save the content subset and run JSON; emit component metrics. Scope workers and remove private iterator deletion/forced exit together. Add a tiny injected-loss training smoke check for parameter updates, artifacts and normal shutdown; do not launch a real training job as refactor validation.
7. **Quarantine optional tools and finish the local wrapper.** Move diffusion tests to the explicit research suite, maintain loss-adapter compatibility, correct the visualizer/cache or park it, and retain export only with a consumer. If the optional local UI is adopted, track its selected source, bind only to localhost, and prove the core installs/imports without it. Run core and research suites separately and together. Public takedown execution belongs to the owner's operational plan, not this worker's notes.

The ordering separates compatibility changes from new training objectives. Rough sizes above include focused tests and documentation: S is about half a day, M roughly one to two days, L several days; they exclude model training, visual model selection and public infrastructure removal.

**Evidence and limits**

All relative source citations refer to the private clone at `dfbac70`; absolute citations refer to the read-only real checkout. Every tracked Python module, both runners, all nine test files, the notebook source and installation files were inspected. The real `CLAUDE.md` goals were read, including the explicit independence requirement and long-term shared-loss aim (`/Users/jackblackburn/code/main/audio_video/style-transfer/py/CLAUDE.md:7`, `/Users/jackblackburn/code/main/audio_video/style-transfer/py/CLAUDE.md:16`).

Verification command:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jackblackburn/code/main/audio_video/style-transfer/py/.venv/bin/python -m pytest -p no:cacheprovider
```

Result: 21 passed in 1.86 s. Additional in-memory probes checked loss-weight effects, all preset names, dimensions, image modes, VGG activation aliasing/normalization, fraction rounding, diffusion dimensions and the analytical denoising target. One archived checkpoint was loaded and forwarded without saving an image. No training run was launched beyond the existing automated smoke test. Visual quality, speed/memory benchmarks, GPU behavior, export compatibility, deployment state and the training provenance of the full checkpoint archive remain unverified. All changes proposed here require the owner's later workshop decision.
