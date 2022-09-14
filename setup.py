
import io
import os
import re

from setuptools import find_packages

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='SING',
    version=find_version("SING", "__init__.py"),
    description='Synthetic dIstribution Network Generator',
    author='Aadil Latif',
    author_email='Aadil.Latif@nrel.gov',
    url='http://www.github.com/nrel/SING',
    packages=find_packages(),
    install_requires=["pandas"],
    include_package_data=True,
    package_data={
        'PyDSS': [
            'data_sources/*.csv',
        ]
    },
    license='BSD 3 clause',
    python_requires='==3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    )
