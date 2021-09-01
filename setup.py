"""
Setup script for irbasis_x
"""
import io
import os.path
import re

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext


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


def append_if_absent(list, arg):
    """Append argument to list if absent"""
    if arg not in list:
        list.append(arg)


class BuildExtWithNumpy(build_ext):
    """Wrapper class for building numpy extensions"""
    user_options = build_ext.user_options + [
        ("with-openmp=", None, "use openmp to build (default: true)"),
        ("numpy-include-dir=", None, "numpy include directory"),
        ]

    def initialize_options(self):
        super().initialize_options()
        self.with_openmp = "false"
        self.numpy_include_dir = None

    def finalize_options(self):
        """Add numpy and scipy include directories to the include paths."""
        super().finalize_options()

        # options are always passed as stings
        _convert_to_bool = {"true": True, "false": False}
        self.with_openmp = _convert_to_bool[self.with_openmp.lower()]

        # Numpy headers: numpy must be imported here rather than
        # globally, because otherwise it may not be available at the time
        # when the setup script is run.
        if self.numpy_include_dir is None:
            import numpy
            self.numpy_include_dir = numpy.get_include()

        self._augment_build()

    def _augment_build(self):
        """Modify paths according to options"""
        append_if_absent(self.include_dirs, self.numpy_include_dir)

        for ext in self.extensions:
            if self.with_openmp:
                append_if_absent(ext.extra_compile_args, '-fopenmp')
                append_if_absent(ext.extra_link_args, '-fopenmp')


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

    python_requires='>=3',
    install_requires=[
        'numpy>=1.16',      # we need matmul to be an ufunc -> 1.16
        ],
    extras_require={
        'test': ['pytest'],
        },

    ext_modules=[
        Extension("quadruple._raw", ["csrc/_raw.c"]),
        ],
    setup_requires=[
        'numpy>=1.16',
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
