#!/usr/bin/env python3

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = '0.0.1'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'olfsysm',
        ['libolfsysm/src/olfsysm.cpp', 'bindings/python/pyolfsysm.cpp'],
        include_dirs = [
            'libolfsysm/api',
            'libolfsysm/include',
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        libraries=['gomp'],
        extra_compile_args = ['-fopenmp', '-fpic'],
        extra_link_args = ['-lgomp'],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())

            opts.append('-std=c++17')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            for a in ext.extra_compile_args:
                assert has_flag(self.compiler, a)

            ext.extra_compile_args += opts

        build_ext.build_extensions(self)


setup(
    name='olfsysm',
    version=__version__,
    author="",
    author_email='',
    url='https://github.com/bauersmatthew/olfsysm',
    # TODO
    description='',
    # TODO from readme
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,

    # TODO try to get hc_data.csv installed
    #
    # installs under root of venv (which is both sys.prefix and sys.exec_prefix in my
    # testing), which i'm not thrilled about. that would also be /usr if for some reason
    # pip was run as root
    #data_files=[('', ['hc_data.csv'])],
    #data_files=[('olfsysm', ['hc_data.csv'])],
    #
    # did not work
    #include_package_data=True,
    #
    # MANIFEST.in specifying `include hc_data.csv` also did not work
)

