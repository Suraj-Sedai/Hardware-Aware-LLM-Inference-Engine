from setuptools import setup, find_packages

# Debug: print what packages are found
pkgs = find_packages()
print(f"Found packages: {pkgs}")

setup(
    name="llm-inference-engine",
    version="0.1.0",
    packages=pkgs,
)
