from setuptools import setup, find_packages
setup(
    name="dag_prf_utils",
    packages=find_packages(),
    install_requires=[
        'plotly',  # Add Plotly as a dependency
        'reportlab',
        'PIL',
        # other dependencies...
    ],    
)
