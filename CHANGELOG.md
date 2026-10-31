# Changelog

## Unreleased

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
  (`>= 0.23` on Python 3.15) and icontract `>= 2.6.2`. The `tests` extra needs
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
- Model files are read with `torch.load(weights_only=True)` (#168). Files
  written by earlier releases are refused. Rewrite each one once, passing the
  embedding architecture it was built from (only for files you trust: the
  upgrade unpickles the file without restrictions and loads its TorchScript):

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
  reads the file with the safe loader first, and unpickles a file from an
  earlier release only after the recipe, the architecture, the check inputs
  and the output path have been validated. Reading the old layout is
  deprecated: `equine.upgrade` is removed in a later release.
- `load_equine_model` accepts `embedding_model` (keyword-only), like
  `EquineProtonet.load` and `EquineGP.load`.
- Requires `torch >= 2.14.1`, which has MPS kernels for the Cholesky
  inversion and the entropy that `EquineGP` uses, so `EquineGP` trains and
  predicts on MPS (with EQUINE 0.1.8 and torch 2.6 or 2.9, `EquineGP.predict`
  raised `NotImplementedError` there). On macOS this means an Apple silicon
  Mac with macOS 14 or later: torch 2.12 and later publish macOS wheels only
  for those, so Intel Macs are not supported. Releases before 2.6 also have a
  known bypass of `weights_only=True` (CVE-2025-32434), and the safe load
  path refuses to run on them.
- `save()` refuses a `train_summary` entry or embedding extra state that is not a
  plain value, a tensor, a `torch.Size` or an `OrderedDict`, and a tensor in either
  that is quantized or carries Python attributes. Dict keys must be `str` or `int`;
  in `train_summary` they may also be `float`, `bool` or `None`.
- Loading now refuses several kinds of files crafted to exhaust memory:
  compressed or oversized archives, non-zip files, pickles that call a function
  or class `save` does not write (such as a dtype conversion or a `bytearray`
  that allocates far more than the file holds), tensors whose shape is
  larger than their storage (expanded views), weights laid out as views of
  one storage too small for the network they describe, support tensors that
  alias one another (`equine.upgrade` copies them apart), and
  recipes or settings that describe a model larger than the weights stored in
  the file. Known limit: the work done after loading (embedding the support
  set) grows with the number of support rows times the embedding's width, so
  it can exceed the file's size even for an honest file.
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
  `equine.upgrade` writes, loads as `"rmd_sqrt"` and keeps its scores. `load`
  builds the model from the file's settings and then sets `ood_distance`, so a
  subclass whose `__init__` does not take `ood_distance` loads files without
  the setting; like every saved setting, a file that holds it needs a class
  whose `__init__` takes it. The outlier-score KDEs are fitted for the
  statistic in use: after changing `ood_distance` on a model, refit them with
  `update_support` or `train_model` (#210).
- Model files carry `equine_format_version` 3 (EQUINE 0.1.8 wrote none),
  `EquineGP` files too, whose contents are otherwise unchanged. An EQUINE that
  reads format version 2 at most refuses them with "Unsupported EQUINE model
  format version 3" (#210).
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
