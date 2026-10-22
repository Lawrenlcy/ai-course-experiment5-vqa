"""Official VQA scorer adapter; retain official normalization and leave-one-out."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path


def evaluate(annotations, predictions):
    expected=[a['question_id'] for a in annotations]
    actual=[p['question_id'] for p in predictions]
    if len(expected)!=len(set(expected)) or len(actual)!=len(set(actual)) or set(expected)!=set(actual):
        raise ValueError('Predictions must cover every question exactly once')
    path=Path(__file__).resolve().parents[1]/'third_party/vqaEval.py'
    spec=importlib.util.spec_from_file_location('official_vqa',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class View:
        def __init__(self,rows):self.qa={r['question_id']:r for r in rows}
        def getQuesIds(self):return list(self.qa)
    evaluator=module.VQAEval(View(copy.deepcopy(annotations)),View(copy.deepcopy(predictions)),n=6)
    with contextlib.redirect_stdout(io.StringIO()):evaluator.evaluate()
    return evaluator
