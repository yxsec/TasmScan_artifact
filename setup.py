#!/usr/bin/env python3
"""Setup script for tasmscan - Independent TVM Security Scanner."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README from parent directory
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

setup(
    name="tasmscan",
    version="1.0.0",
    description="Independent TVM security scanner with IR analysis, data flow tracking, and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yxsec/tasmscan",
    packages=find_packages(),
    package_data={
        'tasmscan': [
            'disassembler/*.json',
            'spec/*.json',
        ],
    },
    install_requires=[
        "pytoniq-core>=0.1.0",
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'tasmscan=tasmscan.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Software Development :: Disassemblers",
    ],
    keywords="ton tvm security vulnerability scanner blockchain independent standalone",
)
