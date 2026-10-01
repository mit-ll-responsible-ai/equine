# Changelog

## Unreleased

### Changed
- **Inference embeds each input once and builds no autograd graph** (#173,
  #182, #212). `predict` on both classes runs under `torch.no_grad()` and
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
  calibration pass there (previously twice). `EquineGP.update_support`
  and `EquineGP.load` embed the support set once per class; loaded GP models
  expose `support_embeddings`. The public `EquineGP.compute_prototypes()` still
  re-embeds the support and now refreshes `support_embeddings` as well.
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
  samples its episodes and support on the CPU and moves each batch at that
  boundary. `EquineGP.device`
  is a `str` like `EquineProtonet.device`; `EquineGP.device_type` is
  deprecated (see below). `EquineGP.load` and `load_equine_model`
  accept `device=` (positional second argument) like `EquineProtonet.load`;
  an explicit device is checked before the file is read, and `meta`, an
  unavailable device or an index at or above the device count (`cuda:7` on a
  one-GPU machine, `mps:1`) raises `ValueError`; so does loading a file
  whose tensors have a dtype the device lacks (float64 on MPS), naming the
  dtype. For a model on MPS the GP covariance inversion and the
  entropy in `EquineGP.predict` are computed on the CPU, because torch 2.6 to
  2.9 lack some of those MPS kernels; CPU and CUDA results are unchanged.
  Files saved after training on an
  accelerator hold accelerator tensors; pass `device="cpu"` when loading them
  on a machine without one.
- **Model files no longer embed executable code by default** (#191). `save()`
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
  written by earlier releases need a one-time migration: load the file with
  `allow_unsafe_legacy_format=True` (only for files you trust: it unpickles
  without restrictions) and `embedding_model=` an instance of your registered
  architecture, then call `save()` to rewrite it in the data-only format:

  ```python
  model = equine.load_equine_model("old.eq", allow_unsafe_legacy_format=True,
                                   embedding_model=Encoder(6, 3))
  model.save("new.eq")   # data-only from here on
  ```

  Without `embedding_model=` the loaded embedding is the file's TorchScript
  module, which `save()` refuses; `save(path, allow_executable=True)` keeps
  it as a flagged TorchScript copy instead.

- `load_equine_model` accepts `trust_executable` and `embedding_model`, like
  `EquineProtonet.load` and `EquineGP.load`. These flags,
  `allow_unsafe_legacy_format`, and `save`'s `allow_executable` are
  keyword-only.
- Requires `torch >= 2.6`: earlier releases have a known bypass of
  `weights_only=True` (CVE-2025-32434), and the safe load path refuses to run
  on them. Intel-macOS installs are no longer supported (no torch 2.6 builds
  exist for that platform).
- Loading now refuses several kinds of files crafted to exhaust memory:
  compressed or oversized archives, non-zip files (unless
  `allow_unsafe_legacy_format=True`), tensors whose shape is larger than
  their storage (expanded views), weights laid out as views of one storage
  too small for the network they describe, support tensors that alias one
  another (copied apart instead with `allow_unsafe_legacy_format=True`), and
  recipes or settings that describe a model larger than the weights stored in
  the file. Known limit: the work done after loading (embedding the support
  set) grows with the number of support rows times the embedding's width, so
  it can exceed the file's size even for an honest file.
- Settings keys that the model class does not accept are refused with a
  `ValueError`. `EquineGP` files saved by EQUINE 0.1.5 or earlier store
  `use_temperature`, which EquineGP has not accepted since 0.1.6 (such files
  failed to load); it is now dropped with a warning, and predictions are
  unchanged. Migrate the file as shown above to remove the warning.

### Deprecated (removed in the next release)
- `save(path, allow_executable=True)` embeds a TorchScript copy of an
  unregistered embedding model and flags the file; such files need
  `load(..., trust_executable=True)`. Both flags, the TorchScript archive
  support and the `torch < 2.10` ceiling are removed in the next release;
  saving with `allow_executable=True`, or loading a TorchScript embedding
  under `trust_executable=True`, emits a `FutureWarning`. Migrate flagged
  files with the snippet above (using `trust_executable=True`).
- `EquineGP.device_type`: read `EquineGP.device` (a `str`) instead. Reading
  or assigning it emits a `DeprecationWarning`; assigning it moves the module
  and then sets `device`. To move a model yourself, set `model.device = value`
  and call `model.to(value)`: either one alone leaves the inputs and the
  module on different devices.

### Fixed
- Opening an untrusted `.eq` file could execute arbitrary code (#168).
- Saved files could contain the training rows that the support examples were
  sliced from, not just the support examples themselves. Support tensors are
  now stored as copies, so a file holds only its support rows.
