from setuptools import setup, find_packages

setup(
    name="audio_analysis_deep_learning",
    version="1.0.0",
    author="B. Anadhyanth",
    description="Deep Learning based Audio Analysis System",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "librosa",
        "tensorflow",
        "keras",
        "scikit-learn"
    ],
    python_requires=">=3.8",
)