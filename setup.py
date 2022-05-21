# Setup script - embracing the setuptools madness.
#
# Copyright (C) 2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import io
import os.path
import platform
import re

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as BuildExt


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


def update_flags(exec, update):
    # First, let us clean up the mess of compiler options a little bit:  Move
    # flags out into a dictionary, thereby removing the myriad of duplicates
    cc_so, *cflags_so = exec
    def _splitflag(arg):
        arg = arg.split("=", 1)
        if len(arg) == 1:
            arg = arg + [None]
        return arg
    cflags_so = {k: v for (k,v) in map(_splitflag, cflags_so)}

    # Now update the flags
    cflags_so.update(update)
    cflags_so = [k + ("=" + v if v is not None else "")
                 for (k,v) in cflags_so.items()]
    return [cc_so] + cflags_so


class OptionsMixin:
    _convert_to_bool = {"true": True, "false": False}
    user_options = [
        ("with-openmp=", None, "use openmp to build (default: false)"),
        ("opt-arch=", None, "optimized for architecture"),
        ("numpy-include-dir=", None, "numpy include directory"),
        ]

    def initialize_options(self):
        super().initialize_options()
        self.with_openmp = None
        self.numpy_include_dir = None
        self.opt_arch = None

    def finalize_options(self):
        if self.with_openmp is not None:
            self.with_openmp = self._convert_to_bool[self.with_openmp.lower()]
        if self.opt_arch is not None:
            self.opt_arch = self._convert_to_bool[self.opt_arch.lower()]
        if self.numpy_include_dir is not None:
            if not os.path.isdir(self.numpy_include_dir):
                raise ValueError("include directory must exist")
        super().finalize_options()


class BuildExtWithNumpy(OptionsMixin, BuildExt):
    """Wrapper class for building numpy extensions"""
    user_options = BuildExt.user_options + OptionsMixin.user_options

    def build_extensions(self):
        """Modify paths according to options"""
        # This must be deferred to build time, because that is when
        # self.compiler starts being a compiler instance (before, it is
        # a flag)  *slow-clap*
        # compiler type is either 'unix', 'msvc' or 'mingw'
        compiler_type = self.compiler.compiler_type

        compiler_binary = getattr(self.compiler, 'compiler', [''])[0]
        compiler_binary = os.path.basename(compiler_binary)
        compiler_make = ''
        if 'gcc' in compiler_binary or 'g++' in compiler_binary:
            compiler_make = 'gcc'
        elif 'clang' in compiler_binary:
            compiler_make = 'clang'
        elif 'icc' in compiler_binary:
            compiler_make = 'icc'
        elif compiler_type == 'msvc':
            # See msvccompiler.py:206 - a comment worth reading in its
            # entirety.  distutils sets up an abstraction which it immediately
            # break with its own derived classes.  *slow-clap*
            compiler_make = 'msvc'

        if compiler_type != 'msvc':
            new_flags = {"-Wextra": None, "-std": "c11"}
            # By default, we do not optimize for the architecture by default,
            # because this is harmful when building a binary package.
            if self.opt_arch:
                new_flags["-mtune"] = new_flags["-march"] = "native"
            self.compiler.compiler_so = update_flags(
            self.compiler.compiler_so, new_flags)

        # This has to be set to false because MacOS does not ship openmp
        # by default.
        if self.with_openmp is None:
            self.with_openmp = platform.system() == 'Linux'

        # Numpy headers: numpy must be imported here rather than
        # globally, because otherwise it may not be available at the time
        # when the setup script is run.  *slow-cl ... ah, f*ck it.
        if self.numpy_include_dir is None:
            import numpy
            self.numpy_include_dir = numpy.get_include()

        for ext in self.extensions:
            append_if_absent(ext.include_dirs, self.numpy_include_dir)
            if self.with_openmp:
                append_if_absent(ext.extra_compile_args, '-fopenmp')
                append_if_absent(ext.extra_link_args, '-fopenmp')
                if compiler_make == 'clang':
                    append_if_absent(ext.extra_link_args, '-lomp')

        super().build_extensions()

    def get_source_files(self):
        """Return list of files to include in source dist"""
        # Specifying include_dirs= argument in Extension adds headers from that
        # directory to the sdist ... on some machines.  On others, not.  Note
        # that overriding sdist will not save you, since this is not called
        # from sdist.add_defaults(), as you might expect. (With setuptools, it
        # is never what you expect.)  Instead, sdist requires egg_info, which
        # hooks into a hidden manifest_maker class derived from sdist, where
        # add_defaults() called, the list passed back to sdist, sidestepping
        # the method in the orginal class.  Kudos.
        #
        # Really, if you have monkeys type out 1000 pages on typewriters, use
        # the result as toilet paper for a month, unfold it, scan it at 20 dpi,
        # and run it through text recognition software, it would still yield
        # better code than setuptools.
        source_files = super().get_source_files()
        header_regex = re.compile(r"\.(?:h|hh|hpp|hxx|H|HH|HPP|HXX)$")

        include_dirs = set()
        for ext in self.extensions:
            include_dirs.update(ext.include_dirs)
        for dir in include_dirs:
            for entry in os.scandir(dir):
                if not entry.is_file():
                    continue
                if not header_regex.search(entry.name):
                    continue
                source_files.append(entry.path)

        return source_files


VERSION = extract_version('pysrc', 'xprec', '__init__.py')
REPO_URL = "https://github.com/tuwien-cms/xprec"
DOCTREE_URL = "%s/tree/v%s" % (REPO_URL, VERSION)
LONG_DESCRIPTION = rebase_links(readfile('README.md'), DOCTREE_URL)

setup(
    name='xprec',
    version=VERSION,

    description='xprec precision numpy extension',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords=' '.join([
        'double-double'
        ]),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        ],

    url=REPO_URL,
    author=', '.join([
        'Markus Wallerberger'
        ]),
    author_email='markus.wallerberger@tuwien.ac.at',

    python_requires='>=3',
    install_requires=[
        # we need matmul to be an ufunc -> 1.16
        'numpy>=1.16',
        ],
    extras_require={
        'test': ['pytest'],
        },

    ext_modules=[
        Extension("xprec._dd_ufunc",
                  ["csrc/_dd_ufunc.c", "csrc/dd_arith.c"],
                  include_dirs=["csrc"]),
        Extension("xprec._dd_linalg",
                  ["csrc/_dd_linalg.c", "csrc/dd_arith.c", "csrc/dd_linalg.c"],
                  include_dirs=["csrc"]),
        ],
    setup_requires=[
        'numpy>=1.16'
        ],
    cmdclass={
        'build_ext': BuildExtWithNumpy
        },

    package_dir={'': 'pysrc'},
    packages=find_packages(where='pysrc'),
    )
