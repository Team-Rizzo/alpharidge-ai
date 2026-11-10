import re
import os
import codecs
from os import path
from setuptools import setup, find_packages


def read_requirements(path):
    with open(path, "r") as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    out = []
    for line in lines:
        if line.startswith("-r "):
            out += read_requirements(os.path.join(os.path.dirname(path), line[3:].strip()))
        elif line and line not in out:
            out.append(line)
    return out


requirements = read_requirements("requirements.txt")

VERIFY = read_requirements("requirements-verify.txt")

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# The version from alpharidge_ai/__init__.py
with codecs.open(
    os.path.join(here, "alpharidge_ai/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="alpharidge",
    version=version_string,
    description="AlphaRidge, Bittensor subnet 45: miners, validators and the shared page check",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Team-Rizzo/alpharidge-ai",
    author="Team Rizzo",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.10",
    package_data={"alpharidge_ai.miner": ["data/*"], "alpharidge_ai.schema": ["data/*.json"],
                  "alpharidge_ai.verify": ["data/*.json"]},
    install_requires=requirements,
    extras_require={
        "verify": VERIFY,
        "dev": ["pytest==9.0.2", "pytest-asyncio==1.4.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
    ],
)
