"""
setup.py
========
Package configuration — makes src/ importable as a package from any notebook
or script in the project root.

Usage:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="credit_risk",
    version="0.1.0",
    description="Enterprise Credit Risk Modeling & Validation Framework — Lending Club Dataset",
    packages=find_packages(),
    python_requires=">=3.10",
)
