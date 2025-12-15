import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def read_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()
    # Remove comments and empty lines
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return requirements

setuptools.setup(
    name="decaypy",
    version="3.0.0",
    author="Christopher Fichtlscherer",
    author_email="fichtlscherer@mailbox.org",
    description="A Python tool to calculate decay including spontaneous and induced fission, based on ENSDF data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "test": ["pytest"],
    },
    packages=setuptools.find_packages(),
    package_data={
        "NNDC_decay": ["data/*",
                       "data/data_processed/*",
                       "data/Q_values/*",
                       "data/ENDF_FY/ENDF-B-VIII.0_nfy/*",
                       "data/ENDF_FY/ENDF-B-VIII.0_sfy/*"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
