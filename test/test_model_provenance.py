"""Tests of the model-provenance functions in `pyolfsysm.mb_model` (which olfsysm build
and which commits of our code produced an output). None of these compare version
information against reference outputs: the repro tests are meant to work across versions.
"""
import glob
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from pyolfsysm import build_sources, mb_model
from pyolfsysm.mb_model import (olfsysm_build_info, olfsysm_checkout_info, olfsysm_is_clean,
    olfsysm_clean_problems, require_clean_olfsysm, repo_info, model_provenance,
    write_model_provenance, OlfsysmBuildError, ALLOW_DIRTY_OLFSYSM_ENV_VAR,
    model_provenance_name
)


def test_olfsysm_build_info_keys():
    info = olfsysm_build_info()
    for k in ('version', 'git_hash', 'git_dirty', 'git_untracked', 'git_diff', 'cpp_dirty',
            'cpp_diff', 'source_sha256', 'build_time', 'build_flags', 'compiler',
            'build_host', 'module_file', 'legacy_build'):
        assert k in info, k
    assert info['git_dirty'] in (-1, 0, 1)
    # None: built before the C++ dirty state was recorded
    assert info['cpp_dirty'] in (None, -1, 0, 1)
    if not info['legacy_build']:
        assert re.fullmatch(r'[0-9a-f]{40}', info['git_hash']) or info['git_hash'] == ''
        assert re.fullmatch(r'[0-9a-f]{64}', info['source_sha256'])
        # a clean build has no diff; a dirty one must carry it
        assert (info['git_dirty'] == 1) == (len(info['git_diff']) > 0)
        if info['cpp_dirty'] == 0:
            assert info['cpp_diff'] == ''


def test_source_sha_matches_checkout_when_clean():
    """The sha baked into the extension is over the checkout's sources and headers."""
    info = olfsysm_build_info()
    checkout = olfsysm_checkout_info()
    if info['legacy_build'] or not olfsysm_is_clean():
        pytest.skip('needs a clean olfsysm build matching its checkout')
    root = Path(checkout['path'])
    sources = ['libolfsysm/src/olfsysm.cpp', 'bindings/python/pyolfsysm.cpp']
    headers = sorted(str(Path(p).relative_to(root)) for pat in
        ('libolfsysm/api/**/*', 'libolfsysm/include/**/*')
        for p in glob.glob(str(root / pat), recursive=True) if Path(p).is_file()
    )
    h = hashlib.sha256()
    for rel in sources + headers:
        h.update(rel.encode()); h.update(b'\0')
        h.update((root / rel).read_bytes()); h.update(b'\0')
    assert h.hexdigest() == info['source_sha256']
    # what the clean check compares the build with
    assert checkout['source_sha256'] == info['source_sha256']
    assert build_sources.source_sha256(root) == info['source_sha256']


def test_checkout_info_keys():
    c = olfsysm_checkout_info()
    for k in ('path', 'editable', 'source_sha256', 'git_hash', 'dirty', 'cpp_dirty',
            'untracked', 'remote'):
        assert k in c, k
    if c['git_hash'] is not None:
        assert re.fullmatch(r'[0-9a-f]{40}', c['git_hash'])
        assert isinstance(c['cpp_dirty'], bool)


def _fake_build(**kw):
    info = dict(version='0.0.1', git_hash='a' * 40, git_dirty=0, git_untracked=0, git_diff='',
        cpp_dirty=0, cpp_diff='', source_sha256='1' * 64, build_time='', build_flags='',
        compiler='', build_host='', legacy_build=False, module_file='olfsysm.so'
    )
    info.update(kw)
    return info


def _fake_checkout(**kw):
    info = dict(path='/olfsysm', editable=True, source_sha256='1' * 64, git_hash='b' * 40,
        dirty=False, cpp_dirty=False, untracked=0, remote=None
    )
    info.update(kw)
    return info


@pytest.mark.parametrize('build_kw, checkout_kw, n_problems', [
    # a later commit, and uncommitted changes, to the Python package only: clean
    (dict(), dict(), 0),
    (dict(git_dirty=1, git_diff='diff of pyolfsysm'), dict(dirty=True), 0),
    # the checkout's C++ sources are not those compiled (stale build)
    (dict(), dict(source_sha256='2' * 64), 1),
    # uncommitted C++ changes in the checkout (also stale, since sources differ)
    (dict(), dict(source_sha256='2' * 64, cpp_dirty=True), 2),
    # built from uncommitted C++ changes
    (dict(git_dirty=1, cpp_dirty=1, cpp_diff='diff'), dict(), 1),
    # a build from before cpp_dirty was recorded: the whole checkout's state decides
    (dict(cpp_dirty=None, cpp_diff=None, git_dirty=1), dict(), 1),
    (dict(cpp_dirty=None, cpp_diff=None), dict(), 0),
    # not built from a git checkout
    (dict(git_hash='', git_dirty=-1, cpp_dirty=-1), dict(), 1),
    # not a source checkout (e.g. a non-editable install): nothing to compare with
    (dict(), dict(source_sha256=None, git_hash=None, dirty=None, cpp_dirty=None), 0),
])
def test_clean_problems_compare_cpp_sources(monkeypatch, build_kw, checkout_kw, n_problems):
    """The clean check compares the compiled C++ sources with the checkout's, not commits.
    """
    monkeypatch.setattr(mb_model, 'olfsysm_build_info', lambda: _fake_build(**build_kw))
    monkeypatch.setattr(mb_model, 'olfsysm_checkout_info',
        lambda: _fake_checkout(**checkout_kw)
    )
    problems = olfsysm_clean_problems()
    assert len(problems) == n_problems, problems


def _git(cwd, *args):
    subprocess.run(['git', *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_build_sources_python_change_needs_no_rebuild(tmp_path):
    """Python-only commits and changes leave the C++ hash and dirty state unchanged.
    """
    root = Path(build_sources.__file__).resolve().parents[1]
    if build_sources.source_sha256(root) is None:
        pytest.skip('pyolfsysm is not in an olfsysm source checkout')
    repo = tmp_path / 'olfsysm'
    for rel in build_sources.SOURCES + build_sources.headers(root):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, repo / rel)
    (repo / 'pyolfsysm').mkdir()
    (repo / 'pyolfsysm' / 'x.py').write_text('x = 1\n')
    _git(repo, 'init', '-q')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'add', '.')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'a')
    sha = build_sources.source_sha256(repo)
    assert sha == build_sources.source_sha256(root)
    assert build_sources.cpp_dirty(repo) == 0

    # an uncommitted, then committed, change to the Python package
    (repo / 'pyolfsysm' / 'x.py').write_text('x = 2\n')
    assert build_sources.cpp_dirty(repo) == 0
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-am', 'b')
    assert build_sources.source_sha256(repo) == sha
    assert build_sources.cpp_dirty(repo) == 0

    # a change to a compiled source, and an untracked header
    src = repo / build_sources.SOURCES[0]
    src.write_text(src.read_text() + '\n// change\n')
    assert build_sources.cpp_dirty(repo) == 1
    assert build_sources.source_sha256(repo) != sha
    assert 'change' in build_sources.cpp_git_diff(repo)
    _git(repo, 'checkout', '--', build_sources.SOURCES[0])
    assert build_sources.cpp_dirty(repo) == 0
    header = repo / build_sources.HEADER_DIRS[0] / 'new_header.hpp'
    header.write_text('// new\n')
    assert build_sources.cpp_dirty(repo) == 1
    assert build_sources.source_sha256(repo) != sha


def test_is_clean_consistent_with_problems():
    assert olfsysm_is_clean() == (len(olfsysm_clean_problems()) == 0)


def test_require_clean_error_and_override(monkeypatch):
    monkeypatch.setattr(mb_model, 'olfsysm_clean_problems', lambda: ['pretend it is dirty'])
    monkeypatch.setattr(mb_model, 'STRICT_CLEAN_OLFSYSM', False)
    monkeypatch.delenv(ALLOW_DIRTY_OLFSYSM_ENV_VAR, raising=False)
    with pytest.raises(OlfsysmBuildError) as e:
        require_clean_olfsysm()
    msg = str(e.value)
    assert 'pretend it is dirty' in msg
    # the message must say how to run anyway during development
    assert f'export {ALLOW_DIRTY_OLFSYSM_ENV_VAR}=1' in msg
    # the override turns it into a warning
    monkeypatch.setenv(ALLOW_DIRTY_OLFSYSM_ENV_VAR, '1')
    with pytest.warns(UserWarning):
        require_clean_olfsysm()
    # but not where a clean build is always required
    with pytest.raises(OlfsysmBuildError) as e:
        require_clean_olfsysm(allow_env_override=False, context='this test')
    assert 'this test' in str(e.value)
    monkeypatch.setattr(mb_model, 'STRICT_CLEAN_OLFSYSM', True)
    with pytest.raises(OlfsysmBuildError):
        require_clean_olfsysm()


def test_require_clean_passes_when_clean(monkeypatch):
    monkeypatch.setattr(mb_model, 'olfsysm_clean_problems', lambda: [])
    require_clean_olfsysm()
    require_clean_olfsysm(allow_env_override=False)


def test_repo_info_this_repo():
    info = repo_info(mb_model)
    assert info['git_hash'] is not None and re.fullmatch(r'[0-9a-f]{40}', info['git_hash'])
    assert isinstance(info['dirty'], bool)
    assert (info['diff'] is not None) == info['dirty']
    assert repo_info('/')['git_hash'] is None


def test_model_provenance_and_write(tmp_path):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.arange(6.0).reshape(2, 3), index=['a', 'b'], columns=list('xyz'))
    prov = model_provenance(orn_deltas=df, scale=1.5, nothing=None)
    for k in ('timestamp', 'host', 'platform', 'cpu', 'python', 'packages', 'olfsysm',
            'pyolfsysm', 'hong2p', 'inputs'):
        assert k in prov, k
    # al_analysis only when it is the caller (it has been imported)
    assert ('al_analysis' in prov) == ('al_analysis' in sys.modules)
    # this package's checkout is the olfsysm repository (with the C++ sources)
    assert prov['pyolfsysm']['path'] == repo_info(mb_model)['path']
    assert prov['inputs']['orn_deltas']['sha256'] == model_provenance(orn_deltas=df
        )['inputs']['orn_deltas']['sha256']
    assert prov['inputs']['scale'] == 1.5 and prov['inputs']['nothing'] is None
    assert set(prov['olfsysm']) >= {'build', 'checkout', 'clean', 'problems'}
    assert prov['olfsysm']['clean'] == olfsysm_is_clean()
    path = write_model_provenance(tmp_path, prov)
    assert path.name == model_provenance_name
    assert json.loads(path.read_text())['python'] == prov['python']
