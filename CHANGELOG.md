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
  written by earlier releases are refused; rewrite each one once with
  `python -m equine.upgrade old.eq` (only for files you trust: the upgrade
  unpickles the file without restrictions). The upgraded file keeps its
  TorchScript embedding, so it is flagged and loads with
  `trust_executable=True`; to make it data-only, load it with
  `embedding_model=` an instance of your registered architecture and call
  `save()`:

  ```python
  # once, from the command line: python -m equine.upgrade old.eq
  model = equine.load_equine_model("old.eq", trust_executable=True,
                                   embedding_model=Encoder(6, 3))
  model.save("new.eq")   # data-only from here on
  ```

  Reading the old layout is deprecated: `equine.upgrade` is removed in a later
  release.
- `load_equine_model` accepts `trust_executable` and `embedding_model`, like
  `EquineProtonet.load` and `EquineGP.load`. These flags and `save`'s
  `allow_executable` are keyword-only.
- Requires `torch >= 2.6`: earlier releases have a known bypass of
  `weights_only=True` (CVE-2025-32434), and the safe load path refuses to run
  on them. Intel-macOS installs are no longer supported (no torch 2.6 builds
  exist for that platform).
- Loading now refuses several kinds of files crafted to exhaust memory:
  compressed or oversized archives, non-zip files, tensors whose shape is
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
