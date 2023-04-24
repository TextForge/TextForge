from setuptools import setup

setup(
    name="TextForge",
    version="0.0.3",
    description="TextForge Metadata Extraction Library",
    packages=["TextForge"],
    install_requires=[
        "numpy",
        "pandas",
        "textstat",
        "scipy",
        "nltk",
        "scikit-learn",
        "textblob"
    ]
)
