from setuptools import setup, find_packages

setup(
    name="TextForge",
    version="0.1.3",
    description="TextForge Metadata Extraction Library",
    packages=find_packages(include=['TextForge', 'TextForge.*']),
    install_requires=[
        # "numpy",
        # "pandas",
        # "textstat",
        # "scipy",
        # "nltk",
        # "scikit-learn",
        # "textblob"
    ]
)
