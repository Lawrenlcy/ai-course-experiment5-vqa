import pytest
from vqa_lab.evaluation import evaluate

def annotation(answers):
    return {'question_id':1,'question_type':'unit','answer_type':'other',
            'answers':[{'answer_id':i+1,'answer':a} for i,a in enumerate(answers)]}

@pytest.mark.parametrize('matches,expected',[(0,0),(1,30),(2,60),(3,90),(4,100),(10,100)])
def test_official_leave_one_out(matches,expected):
    ev=evaluate([annotation(['cat']*matches+['dog']*(10-matches))],[{'question_id':1,'answer':'cat'}])
    assert ev.accuracy['overall']==pytest.approx(expected)

def test_prediction_normalization():
    ev=evaluate([annotation(['2']*9+['two'])],[{'question_id':1,'answer':'The two.'}])
    assert ev.accuracy['overall']==100

def test_preserve_official_unanimous_answer_quirk():
    # Official upstream only normalizes punctuation/digits for diverse GT answers.
    ev=evaluate([annotation(['2']*10)],[{'question_id':1,'answer':'The two.'}])
    assert ev.accuracy['overall']==0

def test_missing_or_duplicate_predictions_rejected():
    a=[annotation(['cat']*10)]
    with pytest.raises(ValueError):evaluate(a,[])
    with pytest.raises(ValueError):evaluate(a,[{'question_id':1,'answer':'cat'}]*2)
