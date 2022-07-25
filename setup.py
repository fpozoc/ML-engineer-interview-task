from setuptools import setup, find_packages


test_deps = [
    "pytest>=6.2.3",
    "pytest-flask>=1.2.0",
    "pip>=21.0.1",
    "flake8>=3.9.2",
    "flake8-annotations>=2.6.2",
    "pytest-cov>=2.12.1",
    "black>=21.7b0"
]

extra_deps = [
    "explainerdashboard",
    "DataPrep",
    "matplotlib",
    "seaborn",
    "dploy-kickstart",
]

extras = {
    "test": test_deps, 
    "extra": extra_deps}

setup(
    name="Nomoko-ML-engineer-interview-task",
    version="0.0.1",
    url="https://github.com/fpozoc/Nomoko-ML-engineer-interview-task",
    author="Fernando Pozo",
    author_email="fpozoc@gmx.com",
    description="Description of my ml-skeleton package",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "loguru",
        "xgboost",
        "textblob",
        "transformers",
        "vaderSentiment",
        "deep_translator", 
        ],
    tests_require=test_deps,
    extras_require=extras,
)