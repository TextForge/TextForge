from setuptools import setup

setup(
    name="textforge-metafeatures",
    version="0.0.2",
    description="TextForge Metadata Extraction Library",
    packages=["textforge-metafeatures"],
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
