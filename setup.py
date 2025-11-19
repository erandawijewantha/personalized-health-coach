"""
Setup configuration for Personalized Health Coach package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="personalized-health-coach",
    version="1.0.0",
    author="Health Coach Team",
    author_email="contact@healthcoach.ai",
    description="Agentic AI system for personalized health recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/personalized-health-coach",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-cov>=4.1.0",
            "flake8>=7.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "health-coach-api=app.api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["*.yaml", "*.json"],
    },
    zip_safe=False,
)