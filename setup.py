from setuptools import setup, find_packages

setup(
    name='adbcr',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/sschrod/ADBCR',
    license='BSD 2-Clause "Simplified" License',
    author='Stefan Schrod',
    author_email='stefan.schrod@bioinf.med.uni-goettingen.de',
    description='ADBCR: Adversarial Distribution Balancing for Counterfactual Reasoning'
)
