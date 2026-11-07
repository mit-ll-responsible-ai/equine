# Changelog

## Unreleased

### Added
- **`EquineProtonet(pred_cov_type=, ood_cov_type=)`**: the covariance type
  of the class distances at prediction and of the distance to the support
  mean in the relative-Mahalanobis OOD score, which 0.1.8 fixed at DIAGONAL
  (`None` keeps DIAGONAL). Saved only when not the default.
- **`update_support(calib_x=, calib_y=)`**: an explicit calibration set for
  the outlier scores, so the whole support is kept and the split is
  reproducible. Keyword-only; both or neither.
- **`SpectralResidualMLP`, SNGP's embedding** (registered as
  `equine.spectral_residual_mlp`). A linear projection and `num_blocks`
  residual blocks `h + coefficient * relu(W h + b)` whose weights are
  spectrally normalized, then a spectrally normalized projection. Once torch's
  power-iteration estimate of each weight's largest singular value has
  converged, each block changes the distance between two hidden states by a
  factor between `1 - coefficient` and `1 + coefficient` (default 0.95). The
  input and output projections have no such lower bound, and `EquineGP`'s
  variance is not guaranteed to grow with an input's distance from the training
  data. Its arguments are validated like `MLP`'s (`MAX_HIDDEN_LAYERS` caps
  `num_blocks`, `MAX_PARAMETERS` the size), and it saves as a recipe like any
  registered architecture. A model using it cannot be pickled: torch serializes
  parametrized modules only through `state_dict()`, which `save()` uses;
  `copy.deepcopy` works.

### Changed
- **Python 3.11–3.15.** EQUINE requires Python 3.11 or later and supports
  Python 3.15 (EQUINE 0.1.8 supported 3.9 and later; 3.9 and 3.10 are end of
  life). On Python 3.15, install torch from PyTorch's index until torch
  publishes 3.15 wheels on PyPI
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). On
  macOS with Python 3.15, use torch >= 2.15: torch 2.14 there crashes when
  torch is used from several threads at once (for example two threads calling
  `predict`), and EQUINE warns at import when it finds that combination. Until
  torch 2.15 is released, install it there from
  `https://download.pytorch.org/whl/test/cpu`.
- **Dependency floors.** numpy `>= 2.0, < 3`, scipy `>= 1.13`, torchmetrics
  `>= 1.2.1` (the previous `>= 0.6` allowed releases that cannot import
  EQUINE's metrics, #201, #205), tqdm `>= 4.66.3`, beartype `>= 0.22.1`
  (`>= 0.23` on Python 3.15) and icontract `>= 2.6.2`. safetensors `>= 0.5.0` is a
  new dependency (model files are safetensors files, see below). The `tests` extra needs
  pytest `>= 7.3.2` and pytest-cov `>= 4.0`.
- The `tests` extra holds what the test suite needs, now including
  `pytest-xdist`; `pre-commit` and `codespell` move to a new `dev` extra
  (`pip install -e '.[dev]'`). The package metadata uses the PEP 639 license
  fields (`license = "MIT"`, `license-files`) instead of the license table
  and classifier that setuptools deprecates; building needs setuptools >= 77
  (#203).
- **Inference entry points compute in eval mode and restore the caller's
  mode** (#209, #179). `predict` and `update_support` on both classes call
  `eval()` on the model for the duration of the call and then restore the
  mode every submodule had before it: each submodule whose class overrides
  `train()` gets `train()` with its previous mode, which undoes what its
  `train(False)` did (loralib layers with `merge_weights=True` unmerge their
  LoRA weights again and keep receiving gradients), and then every
  submodule's `training` flag is set back directly. A prediction is the same
  in either mode and a hand-written fine-tune loop can call `predict` between
  epochs; `EquineGP.predict` after `model.train()` no longer accumulates into
  the Laplace precision matrix (which used to raise `Did not reset precision
  matrix at start of epoch`), and a training-mode forward marks the cached
  Laplace covariance stale, so a covariance cached by a `predict` before the
  epoch's last batch (allowed once more than `num_data - batch_size` of the
  epoch's rows have been seen, that is, when the last batch is smaller than
  `batch_size`) is recomputed by the next eval-mode forward rather than
  reused. Because the flags are set back individually, an embedding frozen
  for fine-tuning (`model.train(); model.embedding_model.eval()`, keeping
  BatchNorm statistics and dropout fixed) is still frozen after the call, as
  is any other mix the caller set.
  `EquineGP.train_model` toggles training and eval mode on the
  wrapper rather than only on the inner module, and both classes leave the
  model in eval mode afterwards. `Protonet.update_support` computes the global
  moments of the support embeddings whatever mode the caller left the model in
  (via `train()`/`eval()`), so `update_support` works on a freshly constructed
  (untrained) `EquineProtonet`; the covariance choice per mode is unchanged
  and trained models produce the same numbers.
- **Inference embeds each input once and builds no autograd graph** (#173,
  #182, #212, #249). `predict` on both classes runs under `torch.no_grad()` and
  performs a single embedding pass (previously two); its outputs are ordinary
  tensors without a `grad_fn`, so input attribution through `predict` no
  longer works (use `forward`). `predict` no longer calls the `EquineGP` /
  `EquineProtonet` object or its inner module (`model.model`) through
  `nn.Module.__call__`, so forward hooks registered on those do not observe
  it; hooks on the embedding model still fire, once per call (previously
  twice). `EquineGP.forward` likewise no longer runs hooks registered on the
  inner `_Laplace` module (`model.model`), only on the wrapper and the
  embedding model. `EquineProtonet.update_support` and the OOD-calibration
  step of `EquineProtonet.train_model` also no longer call `model.model`
  through `__call__`, so hooks on it do not see those passes (temperature
  calibration still calls it), and hooks on the embedding model fire once per
  calibration pass there (previously twice). That calibration pass also runs
  under `torch.no_grad()` now. `EquineGP.update_support`
  and `EquineGP.load` embed the support set once per class; loaded GP models
  expose `support_embeddings` (#214). The public
  `EquineGP.compute_prototypes()` still re-embeds the support and returns the
  prototypes without storing anything on the model.
- **Device handling** (#170, #177, #206, #216, #188). Both model classes now
  keep every tensor they own on their `device`: the `temperature` buffer, the
  support set, the GP's Laplace `seen_data` counter and the covariances it
  allocates. Inputs cross to the model in one place: the arguments of
  `predict`, `forward` and `compute_embeddings`, the support stored by
  `update_support`, and every batch of `train_model` and
  `EquineGP.calibrate_model` are moved to the model device and, when floating
  point, cast to the model's parameter dtype (the embedding model's; for an
  `EquineGP` whose embedding has no parameters, such as `nn.Identity`, the
  Laplace head's). A float64 input to a float32 model used to raise; integer
  inputs, e.g. embedding indices, keep their dtype. `EquineProtonet.train_model`
  samples its episodes and support where the dataset is and moves each batch
  at that boundary. `EquineGP.device`
  is a `str` like `EquineProtonet.device`; `EquineGP.device_type` is
  deprecated (see below). `model.to(...)`, like `cpu()`, `cuda()` and the
  dtype casts, also moves the support, its embeddings, the prototypes and the
  covariances. Afterwards `device` (and the inner `Protonet`'s) names the
  device they are on: it is unchanged when it already names that device, and
  otherwise becomes that device's name (`"mps:0"` after `model.to("mps")` from
  the CPU, `"cpu"` after `cpu()` or `float()` on a model unpickled onto the CPU
  from a CUDA host).
  `EquineGP.load` and `load_equine_model`
  accept `device=` (positional second argument) like `EquineProtonet.load`;
  an explicit device is checked before the file is read, and `meta`, an
  unavailable device or an index at or above the device count (`cuda:7` on a
  one-GPU machine, `mps:1`) raises `ValueError`; so does loading a file
  whose tensors have a dtype the device lacks (float64 on MPS), naming the
  dtype. A file records its model's `device`, where a load without `device=` builds
  the model; pass `device="cpu"` to load one saved from an accelerator
  on a machine without one.
- **Model files no longer embed executable code** (#191). `save()`
  stores the embedding model as a recipe (the name of a registered architecture
  plus its constructor arguments) and a `state_dict`. Register your embedding
  class once:

  ```python
  @equine.embedding_architecture("myproject.encoder")
  class Encoder(torch.nn.Module):
      def __init__(self, in_features: int, out_features: int): ...
  ```

  Constructor arguments must be plain values (numbers, strings, booleans,
  `None`, and lists, tuples and dicts of those, with `str` or `int` dict
  keys). The shipped `equine.MLP` is already registered. The example
  notebooks now register their embedding architectures.
- **Model files are safetensors files** (#168, #287). `save()` writes the
  tensors as a safetensors file, and the settings, the embedding recipe and the
  train summary as JSON in its header (`equine_format_version` 4; EQUINE 0.1.8
  wrote none). Values JSON does not hold (tuples, dict keys that are not
  strings, NaN and infinities, `OrderedDict` and its module versions,
  `torch.Size`) are tagged, so they load back as they were saved. Loading reads
  no pickle and runs no code from the file. Files written by earlier releases
  are refused. Rewrite each one once, passing the embedding architecture it was
  built from (only for files you trust: the upgrade unpickles the file without
  restrictions and loads its TorchScript):

  ```python
  # from Python
  from equine.upgrade import upgrade_model_file
  upgrade_model_file("old.eq", Encoder(6, 3))
  ```

  ```shell
  # or from the command line
  python -m equine.upgrade old.eq --import myproject.models \
      --recipe '{"builder": "myproject.encoder", "kwargs": {"in_features": 6, "out_features": 3}}'
  ```

  The upgrade copies the TorchScript weights into the architecture (including
  tensors the old module kept as plain attributes), writes a data-only file,
  loads it back and checks that its embedding reproduces the old one on the
  file's support rows and on any check inputs before replacing anything.
  `EquineGP` files usually store no support rows (`vis_support=False` is the
  default), and the upgrade refuses a file it cannot check, writing nothing.
  Pass inputs to compare on, one per row: `check_inputs=X` from Python, or
  `--check-inputs FILE` with a tensor saved by `torch.save`. To upgrade
  without the check, pass `allow_unchecked=True` (`--no-check`). The upgrade
  reads the file with `torch.load(weights_only=True)` first, and unpickles a
  file from an earlier release only after the recipe, the architecture, the
  check inputs and the output path have been validated. The upgrade also
  rewrites a format 2 or 3 file, which development builds wrote, with no
  `--recipe`: `python -m equine.upgrade FILE`, adding `--import MODULE` when
  its recipe names an architecture that EQUINE does not ship. Reading the old
  layout is deprecated: `equine.upgrade` is removed in a later release.
- `load_equine_model` accepts `embedding_model` (keyword-only), like
  `EquineProtonet.load` and `EquineGP.load`.
- Requires `torch >= 2.14.1`, which has MPS kernels for the Cholesky
  inversion and the entropy that `EquineGP` uses, so `EquineGP` trains and
  predicts on MPS (with EQUINE 0.1.8 and torch 2.6 or 2.9, `EquineGP.predict`
  raised `NotImplementedError` there). On macOS this means an Apple silicon
  Mac with macOS 14 or later: torch 2.12 and later publish macOS wheels only
  for those, so Intel Macs are not supported.
- `save()` refuses a `train_summary` entry or embedding extra state that is not a
  plain value, a tensor, a `torch.Size` or an `OrderedDict`, and a tensor in either
  that is quantized or carries Python attributes. Dict keys must be `str` or `int`;
  in `train_summary` they may also be `float`, `bool` or `None`. It also refuses a
  tensor anywhere in the model that is not dense or whose dtype a model file does
  not store (only float64, float32, float16, bfloat16, int64, int32, int16, int8,
  uint8 and bool are stored), settings, a recipe, feature and label names and a
  summary that need more than 4 MiB of JSON (a recipe whose lists contain each
  other many levels deep can reach it), and a model file header longer than
  12 MiB.
- Loading refuses a header other than the ones `save` writes: a header longer
  than 12 MiB, other metadata, a newer format version, JSON that is not strict
  or is larger than 4 MiB, and a stored tensor referenced from more than one
  place or from none (all checked before any tensor is read), then malformed
  tags, nesting deeper than `save` writes, and dict keys whose hashes repeat.
  safetensors refuses tensors whose bytes overlap or run past the file. Each
  tensor is read once. Recipes or settings that describe a model larger than
  the weights stored in the file are refused too. Known limit: the work done
  after loading (embedding the support set) grows with the number of support
  rows times the embedding's width, so it can exceed the file's size even for
  an honest file.
- Settings keys that the model class does not accept are refused with a
  `ValueError`. `EquineGP` files saved by EQUINE 0.1.5 or earlier store
  `use_temperature`, which EquineGP has not accepted since 0.1.6 (such files
  failed to load); `equine.upgrade` drops it, and predictions are unchanged.
- **Protonet OOD statistic.** With `relative_mahal=True` (the
  default), a new `EquineProtonet` scores OOD on the squared Mahalanobis
  distance to the predicted class's prototype minus the squared Mahalanobis
  distance to the mean of the support embeddings; EQUINE 0.1.8 subtracted the
  distances themselves. The squared distances are computed without the
  temperature, like the distances the outlier-score KDEs are fitted on; with
  `use_temperature=True` the temperature still scales the class
  probabilities. OOD scores of new models differ from 0.1.8's on the same
  data and stay in [0, 1]. The new `EquineProtonet` parameter `ood_distance`
  selects the statistic: `"rmd_sq"` (the default) or `"rmd_sqrt"` (0.1.8's,
  including its division of the class distance by the temperature). `save`
  writes the setting; a model file without it, which includes every file
  `equine.upgrade` writes from an EQUINE 0.1.x file, loads as `"rmd_sqrt"` and
  keeps its scores. `load`
  builds the model from the file's settings and then sets `ood_distance`, so a
  subclass whose `__init__` does not take `ood_distance` loads files without
  the setting; like every saved setting, a file that holds it needs a class
  whose `__init__` takes it. The outlier-score KDEs are fitted for the
  statistic in use: after changing `ood_distance` on a model, refit them with
  `update_support` or `train_model` (#210).
- A model file with an `equine_format_version` newer than this EQUINE's is
  refused with "Unsupported EQUINE model format version N".
- `Equine` declares `get_support()` and `update_support(support_x,
  support_y, *args, **kwargs)` as abstract methods, so a subclass of `Equine`
  must implement both. `update_support` on both model classes takes its
  options by keyword: `calib_frac` and `label_names` on `EquineProtonet`,
  `support_size` on `EquineGP` (passing them by position is deprecated, see
  below) (#215).
- `EquineProtonet.support`, `support_embeddings` and `prototypes` return the
  model's support examples, their embeddings and the class prototypes (the
  objects `get_support()` and `get_prototypes()` return; empty, and an empty
  tensor for `prototypes`, before the first support update) and are
  read-only: assigning one raises `AttributeError`. In EQUINE 0.1.8 they were
  empty containers that the model never filled. `Equine.__init__` no longer
  creates these attributes; `EquineGP.__init__` creates its own (#214).
- `EquineProtonet.predict` computes the OOD scores of the examples
  predicted as one class in one batched pass, instead of one
  `gaussian_kde.integrate_box_1d` call per example. The scores are the
  same apart from rounding in the last bits (#178).
- `generate_episode` and `generate_support` gather only the rows they
  keep (the support and the query examples) instead of every row of each
  selected class, and `update_support` on both classes no longer runs
  `torch.unique` over all the labels once per class. The random draws are
  the same `torch.randperm` calls, in the same order, so a seed selects the
  same rows (#196, #213).
- `EquineProtonet.train_model` finds the labels and each class's training
  examples once per call instead of once per episode. `EquineGP.train_model`
  (training and validation) and `calibrate_model` read a `TensorDataset` of
  dense tensors with at most 4 dimensions and of dtype uint8, int8 to int64,
  float16, bfloat16, float32, float64, complex64 or complex128 one batch at
  a time instead of one example at a time; subclasses of `TensorDataset`,
  other tensors and other datasets keep the `DataLoader`.
  The random draws are the same, so a seed selects the same episodes and
  batches (#284).
- `EquineGP.train_model` adds each batch to the precision matrix with `addmm_`
  on accelerators, and on the CPU from a row-major copy of the transposed
  features instead of a transposed view. On macOS the update for a 64-row
  batch with 1,024 random features takes 0.30 ms instead of 2.1 ms; on Linux
  arm64 the copy makes it about 6% slower. For float32 and float64 models
  whose precision matrix has the model's dtype, trained outside autocast, each
  batch's update can change by rounding, and inverting the precision matrix in
  `predict` can enlarge the change in the covariance, probabilities and OOD
  scores (#285).
- `EquineProtonet.predict` reads each OOD score from a float64 table of the CDF
  of the predicted class's outlier-score KDE, kept on the model, instead of
  summing a normal CDF over every KDE point. A `predict` that predicts a class
  with no table for its KDE's current points, weights and bandwidth builds one
  when it predicts the class for at least one eighth as many examples as the
  table has nodes (1,024 examples always suffice), which can make that call
  slower than before; a call with fewer examples of the class sums its KDE as
  before. Every call reads a class's table while its KDE keeps those values. An
  example's score can differ by up to 1e-13 depending on whether an earlier
  call built its class's table. Later calls that score many examples of a class
  with a table are faster; a call that scores a few examples of a class can
  take slightly longer than before. A table is not read after its KDE's points,
  weights or bandwidth change, pickling and `copy.deepcopy` leave the tables
  out, and they are never saved. A KDE is summed over its points as before when
  its table would need more than 8,192 nodes or more than 2**27 node-point
  pairs, or when it has a NaN or infinite point or weight, a negative weight,
  unequal weights whose sum differs from 1 by more than 1e-14, a bandwidth that
  is not a positive finite number, or all points equal with a bandwidth too
  small to separate two nodes. The table is within 1e-13 of the float64 sum.
  The difference between the OOD distance and each KDE point is now taken in
  float64; when the distances and the points were both float32 or narrower, as
  in float32 models, it was taken in the wider of their two dtypes, so a
  float32 model's scores can change (by at most 6e-8 in the cases measured)
  (#286).
- `EquineGP.train_model` returns the mean of each epoch's batch losses as
  `"train_loss"`, a list of floats. It sums the losses on their device and
  reads the sum once per epoch, where it read every batch's loss from the
  device and discarded the sum (#298).
- In evaluation mode (`predict`, and `forward` after `eval()`), `EquineGP`
  computes only the predictive variance of each example, the diagonal of the
  Laplace predictive covariance, instead of the whole batch-by-batch
  covariance matrix (1 GiB at 16384 rows in float32). Memory grows linearly
  with the batch size and large batches run faster; predictions are equal up
  to floating-point rounding (#181).
- **Type annotations.** `load` on `Equine`, `EquineGP` and `EquineProtonet`
  is annotated to return `Self`, so a type checker reads
  `EquineProtonet.load(path)` as an `EquineProtonet` and `EquineGP.load(path)`
  as an `EquineGP` (EQUINE 0.1.8 annotated both as returning `Equine`).
  `EquineProtonet.model` is declared as a `Protonet` instead of a
  `torch.nn.Module`. The datasets that `EquineGP.train_model` and
  `EquineGP.calibrate_model` take are declared as `Dataset[Any]` instead of
  `Dataset`; they accept the same values. pyright reports no errors in `src/`
  and a 100% type-completeness score for the public API, and CI runs both
  checks (`tox -e pyright`) (#172, #208, #233).

### Deprecated
- `EquineGP.device_type`: read `EquineGP.device` (a `str`) instead. Reading
  or assigning it emits a `DeprecationWarning`; assigning it moves the model
  with `model.to(value)` and then sets `device` to `value`. To move a model,
  call `model.to(value)`.
- Passing the options of `update_support` by position, as in
  `EquineProtonet.update_support(x, y, 0.2)` or
  `EquineGP.update_support(x, y, 25)`: pass `calib_frac=`, `label_names=`
  and `support_size=` by keyword. A positional call emits a
  `DeprecationWarning` and then runs as the keyword call; it stops working in
  the next release (#215).

### Removed
- TorchScript in model files: `save(..., allow_executable=True)`,
  `load(..., trust_executable=True)` and the `trust_executable` argument of
  `load_equine_model`. A file that stores its embedding model as TorchScript
  is refused with a message that points to `equine.upgrade`, which converts
  it.
- The `torch < 2.10` cap; torch is now `>= 2.14.1, < 3`.

### Fixed
- An `EquineProtonet` given its support with `update_support` alone (no
  `train_model`, e.g. on a frozen embedding) saved a file that
  `load_equine_model` refused with `Unknown model type ''`: `modelType` was
  set only by `train_model`. It is now set when the model is built.
- `EquineProtonet.calibrate_temperature` and `EquineGP.calibrate_model` could
  drive the temperature to zero or below with more epochs or a larger
  learning rate, which reverses every class ranking. Both now keep it at or
  above 1e-3.
- With `relative_mahal=False` and a temperature, `EquineProtonet` scored OOD
  on temperature-scaled distances against outlier-score KDEs fitted on
  unscaled ones, so changing the temperature moved the OOD scores (#279).
  With `ood_distance="rmd_sq"`, the default for new models, the OOD
  statistic now reads unscaled distances whether or not `relative_mahal` is
  set. A model file without `ood_distance` loads as `"rmd_sqrt"`, the 0.1.8
  statistic, and keeps its OOD scores.
- Opening an untrusted `.eq` file could execute arbitrary code (#168).
- Saved files could contain the training rows that the support examples were
  sliced from, not just the support examples themselves. Support tensors are
  now stored as copies, so a file holds only its support rows.
- Training with `cov_type=CovType.FULL` could write NaN into the
  embedding's weights and then fail in `linalg.svd` when the embedding was
  wider than the support set; a 64-wide embedding at the default
  `support_size=25` failed in every run (#257). `mahalanobis_distance_nosq`
  now computes `x^T C^-1 x` from a Cholesky factorization, which is never
  negative and has finite gradients; its docstring had stated `x^T C x`
  (#211). A covariance whose Cholesky factorization fails now raises
  `torch.linalg.LinAlgError` instead of returning NaN or meaningless values
  (a singular covariance that rounding leaves positive definite still gives
  very large distances), and an unbatched `(dim, dim)` covariance is
  refused. In full-covariance training, a class covariance that does not
  factor in its dtype (float32 rounding of a rank-deficient or large
  embedding) gets the smallest diagonal jitter, a power of ten times machine
  epsilon times its largest variance, that lets it factor. Predictions are
  unchanged: they use diagonal covariance.
- `generate_model_metrics` and `generate_model_summary` passed the true labels
  and the predictions to torchmetrics in the wrong order, so `accuracy` was
  wrong and `confusionMatrix` was transposed. `confusionMatrix` has one row per
  true class and one column per predicted class (#174).
- `EquineGP.train_model` reported each validation metric accumulated over all
  epochs so far instead of for that epoch, failed when a metric was not on the
  model's device, and computed nothing when `val_metrics` was a generator. Each
  metric is now moved to the model's device in place and reset before each
  epoch's validation pass (#171).
- `update_support` failed with `IndexError` when a class had a single example.
  `EquineGP.update_support` now accepts it. `EquineProtonet.update_support`
  raises `ValueError` naming the class when the `calib_frac` split leaves a
  class with fewer than 2 support or 2 calibration examples, and
  `EquineProtonet.train_model` does so when a class has too few training
  examples for `support_size` (one more when episodes run, for a query
  example) or fewer than 2 calibration examples, before training starts. Both failed inside scipy or partway through training, or silently
  dropped a class with no training examples (#184, #176).
- A failed `EquineProtonet.update_support` leaves the model as it was.
- A Protonet class with a single support example got a NaN variance that
  prediction treated as zero distance, so every query was assigned to that
  class. A diagonal or full covariance now needs at least 2 examples per class,
  and loading a file whose support has a class with fewer than 2 examples
  raises `ValueError`. EQUINE 0.1.8 wrote such a file only when training with
  `support_size=1` and `num_episodes=0`, or under `python -O`; that model
  predicted uniform probabilities. `train_model` now refuses a `support_size`
  below 2 (#193).
- Fitting a Protonet's outlier scores raises `ValueError` naming the class when
  all its calibration examples have the same OOD distance, instead of a scipy
  `LinAlgError`.
- `save` on a model that was never trained wrote a file that `load` refused; it
  now raises `RuntimeError` and writes nothing (#218).
- `predict` on an empty batch raised an icontract error (Protonet) or returned
  an empty result (GP), and on input whose embeddings hold NaN or infinite
  values raised an icontract error (GP) or returned finite-looking scores
  (Protonet). Both models now raise `ValueError` in both cases; an embedding
  that maps NaN input to finite values keeps working. The same `ValueError`
  comes from `EquineProtonet.forward` and `calibrate_temperature` in eval mode,
  and from the calibration pass of `EquineProtonet.train_model` and
  `update_support`; `EquineGP.forward` is unchanged (#219).
- Checks written as `assert` were skipped under `python -O`. The GP's precision
  matrix checks raise `RuntimeError` (a failed inversion raises
  `torch.linalg.LinAlgError`), and support and episode sampling raise
  `ValueError`; an episode now needs more than `support_size` examples of each
  class, which was checked only by an `assert` (#217).
- `copy.deepcopy` on a trained `EquineGP` or `EquineProtonet`, or on one
  loaded from a file with a support set, raised `RuntimeError`: the
  tensors a model stores (the GP's precision matrix, support embeddings
  and prototypes, the Protonet's support embeddings, prototypes and
  covariances) kept autograd history, through which a trained GP also
  held its last epoch's training features. Training, `update_support` and
  `load` now compute them without autograd, and a deep copy's GP
  projection uses the copy's own random matrix (#250).
- `EquineGP.predict` and `EquineGP.forward` (`model(x)`) raised
  `RuntimeError` for a single unbatched example in eval mode; in training
  mode `forward` returned 1-D logits, counted the example's features as
  training rows and added a constant to every entry of the precision
  matrix. Both now treat it as a batch of one, as `EquineProtonet.predict`
  does (#247).
- `EquineProtonet` assumed its class labels were 0 to C-1. With other labels,
  such as `{1, 2, 3}` or `{0, 2, 5}`, `predict` raised `KeyError` or scored OOD
  with another class's KDE; `train_model` on a model built with
  `use_temperature=True` raised `IndexError` on the CPU and calibrated against
  the wrong classes on MPS; and
  `generate_model_summary` failed or reported wrong metrics. Probability
  column `j` is the `j`-th label of `get_support()`; OOD scores, temperature
  calibration and the summary map columns to labels through it, and
  `generate_model_metrics` takes a keyword-only `label_order` to do the same.
  `update_support` with a different label set drops the outlier-score KDEs of
  labels no longer in the support, and `predict` on a model or file whose
  support has a label without a KDE raises `ValueError` naming it (refit with
  `update_support`) instead of using another class's KDE (#169).
- `EquineGP(..., emb_out_dim=0)` never returned: the random Fourier features
  loop did not end and memory grew until the process was killed. `EquineGP`
  now raises `ValueError` unless `emb_out_dim`, `num_classes` and
  `num_random_features` are positive integers; zero `num_classes` or
  `num_random_features` built an empty head and a negative size raised a torch
  `RuntimeError` (#264).
- Docstrings that contradicted the code now match it: `Equine.predict`
  returns probabilities, `EquineGP.forward` returns logits,
  `EquineProtonet.forward` and `Protonet.compute_classes` return
  probabilities, `Protonet.create_model_head` returns an `Identity`,
  `EquineProtonet.calibrate_temperature` is run by `train_model` only when
  `use_temperature` is True, `generate_support` and `generate_episode`
  describe their arguments and return values, and
  `EquineGP.train_model` documents `scheduler`, `vis_support` and
  `support_size` (#208).
- The `EquineGP.calibrate_model` docstring said `train_model` also runs it; it
  does not, and `temperature` stays at `init_temperature` until
  `calibrate_model` is called. The `calibration_batch_size` parameter is now
  documented (#192).
- The `EquineGP.train_model` docstring said it takes a `TensorDataset`.
  It needs a `dataset` whose `dataset[:]` returns the inputs and labels
  (a `TensorDataset`, a `Subset` of one or a `random_split` part), and
  `vis_support=True` needs a `TensorDataset` (#171).
- `generate_episode` gave 0-d labels to an episode with a single query
  example, which its postcondition refused with `IndexError` while contracts
  are on; the labels now have shape `(n,)`. When `num_episodes` is positive,
  `EquineProtonet.train_model` raises `ValueError` before training starts
  unless `way` is between 2 and the number of classes and `episode_size` is at
  least `way`. Calls that could run before and now raise include `way=1` with
  a custom `loss_fn`, and a positive `episode_size` below `way` (#280).
- One-class models could not predict: `EquineProtonet.train_model` and
  `update_support` failed partway, and a one-class `EquineGP` gave NaN
  outlier scores, which `predict`'s postcondition refuses while contracts are
  on. `EquineGP` now raises `ValueError` for a `num_classes` of 1, and
  `EquineProtonet.train_model` and `update_support` raise it for labels of
  fewer than 2 classes before the model changes; loading or upgrading a
  one-class model file raises it too. Calls that could run before and now
  raise include constructing, training and saving a one-class `EquineGP`,
  and loading a one-class file (#281).
- `EquineProtonet.update_support` could merge labels with the same integer
  part, such as 0 and 0.5, into one class without an error, and so could
  `train_model` under `python -O`. Both now raise `ValueError` before the
  model changes when the labels are not 1-D or not whole numbers (NaN and
  infinities included). Calls that could run before and now raise include
  those merges and, under `python -O`, `train_model` and `update_support` on
  labels holding NaN (#289).
- `EquineGP.train_model`, `calibrate_model` and `update_support` truncated
  labels that are not whole numbers, so labels such as 0 and 0.5 trained,
  calibrated or entered the support as one class without an error. They now
  raise `ValueError` when the labels are not 1-D or not whole numbers (NaN
  and infinities included): `train_model` and `update_support` before the
  model changes, `calibrate_model` at the first batch holding such a label.
  A `calibrate_model` call that raises now leaves the temperature as it was,
  without gradients enabled. `train_model` reads `dataset[:]` before training
  instead of after it, so a dataset that draws random numbers when indexed
  trains differently from the same seed. Labels of `validation_dataset` still
  go to `val_metrics` as given. Calls that could run before and now raise
  include those merges, NaN labels on some platforms and, under `python -O`,
  `train_model` on 2-D labels with a `loss_fn` that accepts them (#290).
- `EquineGP.train_model` and `calibrate_model` accepted labels outside
  `[0, num_classes)`. With the default loss, rows labelled -100 were left out
  of it without an error (the default `ignore_index` of `CrossEntropyLoss`
  and `cross_entropy`). Other such labels raised `IndexError` on the CPU at
  the first batch holding one, after any earlier batches had changed the
  model; on MPS, `train_model` raised `AcceleratorError` with torch 2.15 and
  trained on them without an error with torch 2.14. `update_support` stored
  them as classes. The three methods now raise `ValueError` for them, right
  after the check for labels that are not whole numbers. Calls that could run
  before and now raise include rows labelled -100, `train_model` with a
  `loss_fn` that ignores or masks such labels (for example
  `CrossEntropyLoss(ignore_index=-1)`), `train_model` on MPS with torch 2.14,
  and `update_support` with such labels (#291).
- A float64 `EquineGP` accumulated its precision matrix in float32: the
  identity `train_model` starts each epoch from had the default dtype. Its
  covariance therefore had float32 accuracy (relative error up to about 5e-3
  on 200,000 rows). The precision matrix now takes the dtype that the GP's
  weight dtype and the default dtype at `train_model` promote to. With the
  float32 or float64 default, only models whose weights are wider than the
  default change, such as float64 models with the float32 default: their
  probabilities and OOD scores change slightly (up to about 1e-3 on 200,000
  rows), predicted classes do not, and their saved files hold a float64
  precision matrix. Under a float16 or bfloat16 default, where the precision
  was half precision, it is now float32 or wider unless the model has that
  same dtype (#294).
- In a float16 or bfloat16 `EquineGP`, and in a float32 `EquineGP` trained
  under `torch.autocast`, each batch's precision update `k.t() @ k` ran in
  half precision. Its rounding could leave the precision matrix without
  positive definiteness (bfloat16 after a few epochs, on CPU and MPS), and
  `predict` then raised an error, or under `python -O` returned NaN OOD
  scores. The update now runs in the precision's dtype and outside autocast
  (on accelerators it is an in-place `addmm_`, which autocast does not
  change): float32 for those models under the float32 default, and float64
  under a float64 default. A half-precision model trained with its own dtype
  as the default dtype keeps a half-precision precision matrix, which
  `predict` cannot invert. When the precision holds NaN or infinite values, as
  after training that diverged, `predict` raises an error that says so, where
  the inversion could return a NaN or wrong inverse without an error (#297).
- `EquineGP.predict` computed each OOD score in the probabilities' dtype and
  divided it by the entropy of a uniform distribution built in the default
  dtype, and built the jitter added before the precision matrix is inverted in
  the default dtype. A float16 or bfloat16 model's scores were therefore
  computed in half precision, and a float16 or bfloat16 default rounded the
  normalizing entropy to half precision; either could push scores above 1, and
  `predict` then raised an error (under `python -O` it returned them), or
  shift them slightly. Under a float64 default a float32 model's precision was
  inverted in float64 on the CPU, and `predict` raised on MPS. The scores are
  now computed in float32 or wider, clamped to [0, 1] and returned in the
  probabilities' dtype, and the jitter is float32, or the precision's dtype if
  wider, so `predict` gives the same results under every default (#296).
- Under `python -O`, an `EquineProtonet` whose `cov_type` is not a `CovType`
  member, such as the string `"full"`, raised `ValueError` without a message
  from `train_model`. The message now names the `CovType` members and the
  value received, and that `CovType(value)` converts `'unit'`, `'diag'` or
  `'full'`.
