#!/usr/bin/env python3

import datetime
import importlib.util
import os
from pathlib import Path
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import sysconfig
import setuptools

# the minimal pyproject.toml now guarantees we should have this in the build step
import pybind11

# TODO look over when trying to decide if worth migrating to pyproject.toml:
# https://stackoverflow.com/questions/77933206
# https://setuptools.pypa.io/en/latest/userguide/quickstart.html#setuppy-discouraged
# main point is that some use of setup.py may still be supported, as long as it's
# relatively limited. could combine that w/ pyproject.toml, if there's some advantage

__version__ = '0.0.1'


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


# ---------------------------------------------------------------------------
# Build provenance, baked into the extension so a loaded `olfsysm` module can say exactly
# which code it was compiled from (`olfsysm.build_info()`, `olfsysm.__git_hash__`, ...):
# the git commit, whether tracked files were modified (dirty), the diff of a dirty build,
# whether the compiled C++ sources / headers differed from that commit (cpp_dirty) and their
# diff, a SHA-256 over the compiled sources and headers, the build time, flags, compiler and
# host. Written to a generated source file (gitignored) on every build. Which files are
# compiled, and how they are hashed, is defined in pyolfsysm/build_sources.py, which
# pyolfsysm also uses to compare a build with its checkout's current C++ sources (so commits
# that only change the Python package need no rebuild).
HERE = Path(__file__).resolve().parent
BUILD_INFO_CPP = HERE / 'bindings' / 'python' / 'olfsysm_build_info.cpp'


def _load_build_sources():
    # by path, as pyolfsysm is not installed (and its dependencies may not be) at build time
    spec = importlib.util.spec_from_file_location('_olfsysm_build_sources',
        HERE / 'pyolfsysm' / 'build_sources.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_sources = _load_build_sources()
SOURCES = build_sources.SOURCES
HEADERS = build_sources.headers(HERE)
MAX_DIFF_CHARS = 60_000


def _git(*args):
    try:
        return subprocess.run(['git', *args], cwd=HERE, capture_output=True, text=True,
            check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _truncate_diff(diff):
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f'\n... [diff truncated at {MAX_DIFF_CHARS} characters]\n'
    return diff


def git_build_info():
    head = _git('rev-parse', 'HEAD')
    if head is None:
        # not built from a git checkout (e.g. an sdist): nothing known
        return dict(git_hash='', git_dirty=-1, git_untracked=-1, git_diff='', cpp_dirty=-1,
            cpp_diff=''
        )
    status = _git('status', '--porcelain', '--untracked-files=no') or ''
    untracked = _git('ls-files', '--others', '--exclude-standard') or ''
    diff = _git('diff', 'HEAD', '--') or ''
    return dict(git_hash=head.strip(), git_dirty=int(bool(status.strip())),
        git_untracked=len([l for l in untracked.splitlines() if l.strip()]),
        git_diff=_truncate_diff(diff), cpp_dirty=build_sources.cpp_dirty(HERE),
        cpp_diff=_truncate_diff(build_sources.cpp_git_diff(HERE)),
    )


def compiler_version():
    cxx = (sysconfig.get_config_var('CXX') or 'c++').split()[0]
    try:
        return subprocess.run([cxx, '--version'], capture_output=True, text=True,
            check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return cxx


def write_build_info(build_flags):
    info = git_build_info()
    info.update(source_sha256=build_sources.source_sha256(HERE),
        build_time=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        build_flags=build_flags, compiler=compiler_version(), build_host=platform.node(),
    )
    delim = 'OLFSYSM_BI'   # raw-string delimiters are limited to 16 characters
    def raw(x):
        return 'R"' + delim + '(' + x.replace(f'){delim}"', f'){delim[:-1]}_"') + ')' + \
            delim + '"'
    def cstr(x):
        return '"' + x.replace('\\', '\\\\').replace('"', '\\"') + '"'
    BUILD_INFO_CPP.write_text(
        '// Generated by setup.py at build time (gitignored); see olfsysm.build_info().\n'
        'namespace olfsysm_build_info {\n'
        f'extern const char* const git_hash = {cstr(info["git_hash"])};\n'
        f'extern const int git_dirty = {info["git_dirty"]};\n'
        f'extern const int git_untracked = {info["git_untracked"]};\n'
        f'extern const char* const git_diff = {raw(info["git_diff"])};\n'
        f'extern const int cpp_dirty = {info["cpp_dirty"]};\n'
        f'extern const char* const cpp_diff = {raw(info["cpp_diff"])};\n'
        f'extern const char* const source_sha256 = {cstr(info["source_sha256"])};\n'
        f'extern const char* const build_time = {cstr(info["build_time"])};\n'
        f'extern const char* const build_flags = {cstr(info["build_flags"])};\n'
        f'extern const char* const compiler = {cstr(info["compiler"])};\n'
        f'extern const char* const build_host = {cstr(info["build_host"])};\n'
        '}\n'
    )
    return info


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
        # Always recompile every source: setuptools only rebuilds a .cpp whose own mtime is
        # newer than its cached object, so header-only edits or changed flags could
        # otherwise leave stale objects in the extension. Also refreshes the build info.
        self.force = True
        info = write_build_info(f'openmp={int(not force_single_thread)} debug={int(debug)} '
            f'cnpy={int(use_cnpy)}')
        print(f"olfsysm build info: commit {info['git_hash'][:12] or '(none)'}"
            f"{' DIRTY' if info['git_dirty'] == 1 else ''}"
            f"{' (C++ sources DIRTY)' if info['cpp_dirty'] == 1 else ''}, source sha256 "
            f"{info['source_sha256'][:12]}, {info['build_time']}")
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
    url='https://github.com/ejhonglab/olfsysm',
    ext_modules=[
        Extension(
            'olfsysm',
            SOURCES + [str(BUILD_INFO_CPP.relative_to(HERE))],
            depends=HEADERS,
            include_dirs=[
                'libolfsysm/api',
                'libolfsysm/include',
                # Matt was previously passing True/False to this (he called it user=),
                # and adding the result from each call in a separate entry, but I
                # currently seem to be getting same output either way, so just appending
                # one.
                pybind11.get_include(),
            ],
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language='c++'
        ),
    ],

    cmdclass={'build_ext': BuildExt},
    zip_safe=False,

    # the model's Python side (moved from al_analysis, 2026-11), with the data it reads
    packages=['pyolfsysm'],
    package_data={'pyolfsysm': ['data/**/*']},
    python_requires='>=3.8',
    # Only pyolfsysm needs these (the extension itself needs nothing). hong2p and natmix are
    # not on PyPI: install them from GitHub first (al_analysis's requirements, or
    # `pip install 'hong2p[all] @ git+https://github.com/ejhonglab/hong2p'` and
    # `pip install 'natmix @ git+https://github.com/ejhonglab/natmix'`). They are named
    # without a URL here, so installing this package never replaces an existing (e.g.
    # editable) install of them.
    install_requires=[
        'numpy', 'pandas', 'scipy', 'xarray', 'netCDF4', 'matplotlib', 'seaborn',
        'scikit-learn', 'statsmodels', 'rastermap', 'colorcet', 'tqdm', 'termcolor',
        'pyarrow', 'openpyxl', 'drosolf', 'hong2p', 'natmix',
        # importlib.resources.files was added in Python 3.9
        "importlib_resources; python_version < '3.9'",
    ],
    extras_require={'dev': ['pytest >= 8.3.5']},
    entry_points={'console_scripts': [
        'plot-dynamics = pyolfsysm.mb_model:load_and_plot_dynamics_cli',
        'step_model_pn_apl = pyolfsysm.step_model_pn_apl:main',
    ]},

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

