# Changelog

## Unreleased

### Changed
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

### Fixed
- Opening an untrusted `.eq` file could execute arbitrary code (#168).
- Saved files could contain the training rows that the support examples were
  sliced from, not just the support examples themselves. Support tensors are
  now stored as copies, so a file holds only its support rows.
