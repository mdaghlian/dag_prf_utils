from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()
    
setup(
    name="dag_prf_utils",
    packages=find_packages(),
    install_requires=read_requirements()
)
