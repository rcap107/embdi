from setuptools import setup
import os

setup(
    name="EmbDI",
    version="0.7.6",
    packages=["EmbDI"],
    license="",
    description="A Data Integration algorithm based on embeddings.",
    install_requires=["gensim", "datasketch", "pandas", "numpy", "scikit-learn"],
)

os.makedirs("pipeline/", exist_ok=True)
os.makedirs("pipeline/config_files", exist_ok=True)
os.makedirs("pipeline/datasets", exist_ok=True)
os.makedirs("pipeline/dump", exist_ok=True)
os.makedirs("pipeline/embeddings", exist_ok=True)
os.makedirs("pipeline/experiments", exist_ok=True)
os.makedirs("pipeline/info", exist_ok=True)
os.makedirs("pipeline/matches", exist_ok=True)
os.makedirs("pipeline/sim_files", exist_ok=True)
os.makedirs("pipeline/test_dir", exist_ok=True)
os.makedirs("pipeline/walks", exist_ok=True)
os.makedirs("pipeline/generated-matches", exist_ok=True)
os.makedirs("pipeline/run-logs", exist_ok=True)
