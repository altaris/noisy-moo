"""
Installation script.
"""

import setuptools

name = "nmoo"
version = "5.0.0"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split()

packages = ["nmoo"] + [
    "nmoo." + p for p in setuptools.find_packages(where="./nmoo")
]

setuptools.setup(
    author="Cédric HT",
    author_email="hothanh@nii.ac.jp",
    description="A wrapper-based framework for pymoo problem modification.",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering",
    ],
    entry_points={
        "console_scripts": [
            "nmoo = nmoo.__main__:main",
        ],
    },
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    name=name,
    packages=packages,
    platforms="any",
    project_urls={
        "Issues": "https://github.com/altaris/noisy-moo/issues",
    },
    python_requires=">=3.7",
    url="https://github.com/altaris/noisy-moo",
    version=version,
)
