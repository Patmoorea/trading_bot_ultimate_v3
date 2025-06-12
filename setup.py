"""
Trading Bot Ultimate Setup
Version 1.0.0 - Created: 2025-05-26 05:40:31 by Patmoorea
"""

from setuptools import setup, find_packages

setup(
    name="trading_bot_ultimate",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pandas",
        "numpy",
        "ccxt",
        "python-dotenv",
        "pytest",
        "pytest-asyncio",
    ]
)
