#!/usr/bin/env python3

import os
from pathlib import Path
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


# TODO more proper way than an env var? where do pip's new --config-settings go? e.g.
# https://stackoverflow.com/questions/76112858 (though other things in this setup.py
# would already probably fail on pip that new. using 22.3.1 as of 2025-08-01)
#
# TODO also force recompilation if this is set (to avoid having to delete stuff)?
# (how? possible?) maybe build system is set up wrong somehow? seemed even some code
# changes required deleting prior build artifacts, and that shouldn't be the case,
# right?
# TODO TODO try to have this set something in the code, so we can log how the code was
# compiled
#
# NOTE: seems (not always) I need to `rm -rf build/ tmp/ *.so`, prior to `pip install`
# command (from olfsysm root. same place I'm running `pip install` from) like tianpei
# had, to get this change to be reflected in build outputs.
#
# install via `FORCE_SINGLE_THREAD=1 pip install -v .` to disable multithreading
# (to help with debugging)
force_single_thread = bool(int(os.environ.get('FORCE_SINGLE_THREAD', False)))
if not force_single_thread:
    libraries = ['gomp']
    extra_compile_args = ['-fopenmp', '-fpic']
    extra_link_args = ['-lgomp']
else:
    libraries = []
    # -fpic doesn't seem to be (exclusively, at least) OpenMP related
    extra_compile_args = ['-fpic']
    extra_link_args = []

# TODO TODO how to actually debug this now? also need --no-clean on pip?
# https://stackoverflow.com/questions/71125094
# https://marketplace.visualstudio.com/items?itemName=benjamin-simmonds.pythoncpp-debug
# TODO can i call python script w/ gdb? what about pytest?
# TODO TODO maybe attach gdb to existing running process?
#
# TODO don't use 'DEBUG' (uppercase)? b/c of what it might itself do to compile args /
# etc? or does it not affect them? (should be fine?)
#
# compile w/ debug support
debug = bool(int(os.environ.get('DEBUG', False)))
if debug:
    # -g alone should not cause slowdown, but we probably also need the -O0/1 flags for
    # valgrind, which do.
    #
    # -g for debug symbols (for gdb / valgrind). valgrind docs say at least -O1 and
    # maybe -O0 (disabling optimizations) would probably be quite helpful in getting
    # useful output. currently -O2 seems set otherwise, though not sure why.
    # https://valgrind.org/docs/manual/quick-start.html
    # TODO could also see how much less helpful -O1 debug valgrind/gdb output is (and
    # maybe it'd be worth it b/c of less slowdown, so faster debug iteration?)
    #
    # this puts -O0 at end (w/ -O2 still existing earlier in args). last option is the
    # effective one (probably for many flags, but explicitly in docs about -O flag)
    extra_compile_args += ['-g', '-O0']

# TODO how to only add this if we have it available?
use_cnpy = False
cnpy_lib_dir = '/usr/local/lib'
cnpy_shared_lib_file = Path(cnpy_lib_dir) / 'libcnpy.so'
if cnpy_shared_lib_file.exists():
    use_cnpy = True

# TODO can i (easily?) integrate the cmake + make (+ make install? needed?) steps of
# it's compilation, as long as we have the submodule?
if use_cnpy:
    # `sudo make install` (in cnpy build dir, after other steps in README) installs
    # libcnpy.so (and .a) in /usr/local/lib
    #
    # "-Ldir adds directory to list to be searched for -l" (but still needed args
    # mentioned below)
    #
    # needed the `-Wl,-rpath=...` component added (on top of most args recommended by
    # cnpy README), recommended from: https://stackoverflow.com/questions/72052512
    # until then `ldd <venv>/lib/python<x>.<y>/site-packages/olfsysm.cpython*.so` had
    # "libcnpy.so => not found" in output, and I'd get ImportError complaining about
    # missing *.so file on import of olfsysm.
    extra_link_args.extend([f'-Wl,-rpath={cnpy_lib_dir}', f'-L{cnpy_lib_dir}', '-lcnpy',
        '-lz'
    ])


ext_modules = [
    Extension(
        'olfsysm',
        ['libolfsysm/src/olfsysm.cpp', 'bindings/python/pyolfsysm.cpp'],
        include_dirs=[
            'libolfsysm/api',
            'libolfsysm/include',
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
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

            # TODO cnpy README wants: --std=c++11 (presumably, -std=c++11 here?) in
            # args. backward compatible?)
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

    # so that hc_data.csv, which is also referenced in MANIFEST.csv (in order for this
    # to work), is installed too
    # TODO TODO how to make sure we can actually reference hc_data.csv in C++ code?
    # install to specific path by other args here?
    include_package_data=True,
    # TODO try to get hc_data.csv installed
    # TODO copy latest approach from drosolf (2026-02-20). still using setup.py +
    # MANIFEST.in there (that was just manifest + include_package_data=True, which still
    # seems to not be working here?)
    #
    # TODO TODO retry this, just specifying LHS properly? happen enough w/ 2nd commented
    # line, where LHS is 'olfsysm'?
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

