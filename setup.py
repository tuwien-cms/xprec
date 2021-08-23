"""
Setup script for irbasis_x
"""
import io
import os.path
import re

from distutils import log
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


def readfile(*parts):
    """Return contents of file with path relative to script directory"""
    herepath = os.path.abspath(os.path.dirname(__file__))
    fullpath = os.path.join(herepath, *parts)
    with io.open(fullpath, 'r') as f:
        return f.read()


def extract_version(*parts):
    """Extract value of __version__ variable by parsing python script"""
    initfile = readfile(*parts)
    version_re = re.compile(r"(?m)^__version__\s*=\s*['\"]([^'\"]*)['\"]")
    match = version_re.search(initfile)
    return match.group(1)


def rebase_links(text, base_url):
    """Rebase links to doc/ directory to ensure they work online."""
    doclink_re = re.compile(
                        r"(?m)^\s*\[\s*([^\]\n\r]+)\s*\]:\s*(doc/[./\w]+)\s*$")
    result, nsub = doclink_re.subn(r"[\1]: %s/\2" % base_url, text)
    return result


class BuildExtWithNumpy(build_ext):
    """Wrapper class for building numpy extensions"""
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._link_openmp = True

    def check_extensions_list(self, extensions):
        """Strip out all cython extensions that need not be recompiled.

        Cython monkey-patches this method in distutils's build_ext class,
        such that it compiles all out-of-date cython modules.  However, we
        do not want to require Cython in, e.g., a source tarball, so we
        duplicate that functionality here.
        """
        for ext in extensions:
            if '-fopenmp' not in ext.extra_compile_args:
                ext.extra_compile_args.append('-fopenmp')
            if self._link_openmp and '-fopenmp' not in ext.extra_link_args:
                ext.extra_link_args.append('-fopenmp')

        return super().check_extensions_list(extensions)

    def finalize_options(self):
        """Add numpy and scipy include directories to the include paths."""
        super().finalize_options()

        # Add numpy headers
        import numpy
        self.include_dirs.append(numpy.get_include())


VERSION = extract_version('pysrc', 'quadruple', '__init__.py')
REPO_URL = "https://github.com/mwallerb/quadruple"
DOCTREE_URL = "%s/tree/v%s" % (REPO_URL, VERSION)
LONG_DESCRIPTION = rebase_links(readfile('README.md'), DOCTREE_URL)

setup(
    name='quadruple',
    version=VERSION,

    description='quadruple precision numpy extension',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords=' '.join([
        'double-double'
        ]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        ],

    url=REPO_URL,
    author=', '.join([
        'Markus Wallerberger'
        ]),
    author_email='markus.wallerberger@tuwien.ac.at',

    python_requires='>=3, <4',
    install_requires=[
        'numpy',
        'scipy',
        ],
    extras_require={
        'test': ['pytest'],
        },

    ext_modules=[
        Extension("quadruple._fma", ["csrc/fma.c"]),
        ],
    setup_requires=[
        'numpy',
        'scipy',
        ],
    cmdclass={
        'build_ext': BuildExtWithNumpy
        },

    package_dir={'': 'pysrc'},
    packages=find_packages(where='pysrc'),
    entry_points={
        'console_scripts': [
            ],
        },
    )
