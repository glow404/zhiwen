from setuptools import setup, find_packages

setup(
    name="fingerprint-recognition",
    version="1.0.0",
    description="Fingerprint recognition system for mobile unlock",
    author="Fingerprint Team",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.10.0",
        "opencv-python>=4.7.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "scikit-image>=0.19.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "Pillow>=9.0.0",
    ],
    python_requires=">=3.8",
)

