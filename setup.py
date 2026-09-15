import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flash-attn-triton",
    version="0.1.1",
    author="Alyssa Vance",
    author_email="alyssamvance@gmail.com",
    description="Triton-based backend for Flash Attention 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rationalism/flash-attn-triton",
    project_urls={
        "Bug Tracker": "https://github.com/rationalism/flash-attn-triton/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.11", # Python 3.11.13
    install_requires=[
        "einops",
        "torch==2.6.0", # torch==2.8.0 (2025.10) known to fail. 2.5.1+cu124 verified (with triton 3.1.0).
        "triton==3.2.0", # Turing support is dropped in 3.3. 3.1.0 verified (with torch 2.5.1).
        "packaging",
        "pytest"
    ],
)
