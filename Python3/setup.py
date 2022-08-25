import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simuclustfactor",
    version="1.0.0",
    author="Ablordeppey Prosper",
    author_email="prablordeppey@gmail.com",
    description="Simultaneous Component and Clustering Models for Three-way Data: Within and Between Approaches.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prablordeppey/simuclustfactor",
    project_urls={
        "Bug Tracker": "https://github.com/prablordeppey/simuclustfactor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy>=1.19.2', 'sklearn>=1.0.2'],
    license='MIT',
)