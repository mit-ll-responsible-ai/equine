# Establishing Quantified Uncertainty in Neural Networks 
<p align="center"><img src="assets/equine_full_logo.svg" width="720"\></p>


[![Build Status](https://github.com/mit-ll-responsible-ai/equine/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/mit-ll-responsible-ai/equine/actions/workflows/Tests.yml)
![python_passing_tests](https://img.shields.io/badge/Tests%20Passed-100%25-green)
![python_coverage](https://img.shields.io/badge/Coverage-97%25-green)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)

## Usage
Deep neural networks (DNNs) for supervised labeling problems are known to
produce accurate results on a wide variety of learning tasks. However, when
accuracy is the only objective, DNNs frequently make over-confident predictions,
and they also always make a label prediction regardless of whether or not the
test data belongs to any known labels. 

EQUINE was created to simplify two kinds of uncertainty quantification for supervised labeling problems:
1) Calibrated probabilities for each predicted label
2) An in-distribution score, indicating whether any of the model's known labels should be trusted.

Dive into our [documentation examples](https://mit-ll-responsible-ai.github.io/equine/)
to get started. Additionally, we provide a [companion web application](https://mit-ll-responsible-ai.github.io/equine-webapp/).

## Installation
Users are recommended to install a virtual environment such as Anaconda, as is also recommended
in the [pytorch installation](https://github.com/pytorch/pytorch). EQUINE has relatively
few dependencies beyond torch. 
```console
pip install equine
```
Users interested in contributing should refer to `CONTRIBUTING.md` for details.

## Design
EQUINE extends pytorch's `nn.Module` interface using a `predict` method that returns both
the class predictions and the extra OOD scores. 

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2023 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

- Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
- SPDX-License-Identifier: MIT

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

The software/firmware is provided to you on an As-Is basis.
