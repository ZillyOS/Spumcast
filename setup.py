#!/usr/bin/env python3
"""
Setup script for Spumcast - Conversational AI Telegram Bot
"""

from setuptools import setup, find_packages
import pathlib

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = ""
if (here / "README.md").exists():
    long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="spumcast-llm-integration",
    version="1.0.0",
    description="LLM Integration Module for Spumcast Conversational AI Telegram Bot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Spumcast Team",
    author_email="team@spumcast.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="telegram, bot, llm, openrouter, openai, conversational-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "aiohttp>=3.8.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "structlog>=23.0.0",
        "backoff>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/spumcast/issues",
        "Source": "https://github.com/yourusername/spumcast",
    },
) 