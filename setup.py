from setuptools import setup, find_packages

setup(
    name="matrix_lib",
    version="0.1.0",
    description="Library for matrix computations",
    author="Danil",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.23"
    ],
    extras_require={
        "dev": ["pytest"]
    },
)

