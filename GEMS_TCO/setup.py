from setuptools import setup, find_packages

setup(
    name='GEMS_TCO',                      # Name of your package
    version='0.2',                    # Version of your package
    packages=find_packages(),         # Automatically find all packages and modules
    install_requires=[                # List any dependencies here
        'numpy>=1.22,<2.1',
        'requests',
        'tqdm',
        # add other dependencies
    ],
)