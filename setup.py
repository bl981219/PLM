from setuptools import setup, find_packages

setup(
    name="plm-antibody",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "umap-learn>=0.5.0",
        "tqdm>=4.62.0",
    ],
)
