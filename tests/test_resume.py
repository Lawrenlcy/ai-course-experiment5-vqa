import pytest
from vqa_lab import common

def test_run_identity_and_environment_guard(tmp_path,monkeypatch):
    env={'python':'3.11','torch':'2.8','cuda':'12.8','gpu':'test','packages':{'numpy':'2.2'}}
    monkeypatch.setattr(common,'environment',lambda:env.copy())
    common.begin_run(tmp_path,{'seed':42})
    common.begin_run(tmp_path,{'seed':42})
    with pytest.raises(ValueError,match='configuration'):common.begin_run(tmp_path,{'seed':43})
    env['torch']='different'
    with pytest.raises(ValueError,match='environment'):common.begin_run(tmp_path,{'seed':42})
