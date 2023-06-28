import os
import re
from setuptools import find_packages, setup

base = os.path.abspath(os.path.dirname(__file__))


def version():
    with open(os.path.join(base, "src/equine", "__init__.py")) as fid:
        return re.search('__version__ = "(.*)"', fid.read()).groups()[0]


setup(
    name="equine",
    version=version(),
    description="EQUINE^2: Establishing Quantified Uncertainty for Neural Networks",
    url="https://github.com/mit-ll-responsible-ai/equine",
    author="",
    author_email="",
    license="MIT",
    packages=find_packages(where="src", exclude=["tests", "*.tests", "*.tests.*"]),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchmetrics",
        "numpy",
        "tqdm",
        "typeguard<3.0",
        "icontract",
        "scikit-learn",  # TODO: remove dependency on train_test_split
        "scipy",  # TODO: remove dependency on gaussian_kde
    ],
    extras_require={
        "tests": ["pytest-cov >= 3.8", "hypothesis >= 6.41.0", "pre-commit >= 2.19"],
        "docs": [
            "mkdocs>=1.3",
            "mkdocs-material",
            "mkdocstrings-python",
            "mkdocs-jupyter",
            "mkdocs-literate-nav",
            "mkdocs-gen-files",
            "mkdocs-section-index",
        ],
    },
    zip_safe=True,
)

