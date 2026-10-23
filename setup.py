from setuptools import setup

setup(
    name="EmbDI",
    version="0.7.6",
    packages=["embdi"],
    license="",
    description="A Data Integration algorithm based on embeddings.",
    install_requires=["gensim", "datasketch", "pandas", "numpy", "scikit-learn"],
)
