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
        self.with_openmp = None
        self.numpy_include_dir = None

    def finalize_options(self):
        """Add numpy and scipy include directories to the include paths."""
        super().finalize_options()

        _convert_to_bool = {None: None, "true": True, "false": False}
        if self.with_openmp is not None:
            self.with_openmp = _convert_to_bool[self.with_openmp.lower()]
        if self.numpy_include_dir is not None:
            if not os.path.isdir(self.numpy_include_dir):
                raise ValueError("include directory must exist")

    def build_extensions(self):
        """Modify paths according to options"""
        # This must be deferred to build time, because that is when
        # self.compiler starts being a compiler instance
        platform = self.compiler.compiler_type

        # First, let us clean up the mess of compiler options a little bit:
        # Move flags out into a dictionary, thereby removing the myriad of
        # duplicates
        cc_so, *cflags_so = self.compiler.compiler_so
        def _splitflag(arg):
            arg = arg.split("=", 1)
            if len(arg) == 1:
                arg = arg + [None]
            return arg
        cflags_so = {k: v for (k,v) in map(_splitflag, cflags_so)}

        # Replace architecture flag with native to take advantage of the
        # intrinsics
        if platform == 'unix':
            cflags_so["-march"] = "native"
            cflags_so["-mtune"] = "native"
        cflags_so = [k + ("=" + v if v is not None else "")
                     for (k,v) in cflags_so.items()]
        self.compiler.compiler_so = [cc_so] + cflags_so

        # This has to be set to false because MacOS does not ship openmp
        if self.with_openmp is None:
            self.with_openmp = False

        # Numpy headers: numpy must be imported here rather than
        # globally, because otherwise it may not be available at the time
        # when the setup script is run.
        if self.numpy_include_dir is None:
            import numpy
            self.numpy_include_dir = numpy.get_include()

        for ext in self.extensions:
            append_if_absent(ext.include_dirs, self.numpy_include_dir)
            if self.with_openmp:
                append_if_absent(ext.extra_compile_args, '-fopenmp')
                append_if_absent(ext.extra_link_args, '-fopenmp')

        super().build_extensions()


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
