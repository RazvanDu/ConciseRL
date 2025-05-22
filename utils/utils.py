from utils.parser import extract_answer, extract_theoremqa_answer
from utils.grader import math_equal, symbolic_equal
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
from functools import partial
import re
from sympy.parsing.sympy_parser import parse_expr

def _strip(text: str):
    # basic cleaner: lowercase, collapse whitespace, strip punctuation at ends
    return re.sub(r'\s+', ' ', text).strip(" \n\r\t.").lower()

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l_f1_score(target, prediction):
    if target == None or prediction == None:
        #print("NONE")
        return -1000
    result = scorer.score(target, prediction)
    #print("TTT", target, prediction, result)
    return float(result['rougeL'].fmeasure)  # return F1 score only

DATASET_KEYS = {
    'openai/gsm8k': {'question': 'question', 'answer': 'answer'},
    'hendrycks/competition_math': {'question': 'problem', 'answer': 'solution'},
    'datasets/converted_aime_dataset': {'question': 'problem', 'answer': 'solution'},
    'di-zhang-fdu/MATH500': {'question': 'problem', 'answer': 'solution'},
    'datasets/compression_dataset': {'question': 'problem', 'answer': 'solution'},
    'opencompass/AIME2025': {'question': 'question', 'answer': 'answer'},
    'TIGER-Lab/MMLU-Pro': {'question': 'question', 'answer': 'answer'},
    'Idavidrein/gpqa': {'question': 'question', 'answer': 'solution'},
    'isaiahbjork/cot-logic-reasoning': {'question': 'prompt', 'answer': 'response'},
    'TIGER-Lab/TheoremQA': {'question': 'Question', 'answer': 'Answer'},
}

RESPONSE_EXTRACTOR = {
    'openai/gsm8k': lambda x: extract_answer(x, data_name='gsm8k'),
    'hendrycks/competition_math': lambda x: extract_answer(x, data_name='math'),
    'di-zhang-fdu/MATH500': lambda x: extract_answer(x, data_name='math'),
    'datasets/compression_dataset': lambda x: extract_answer(x, data_name='math'),
    'datasets/converted_aime_dataset': lambda x: extract_answer(x, data_name='math'),
    'opencompass/AIME2025': lambda x: extract_answer(x, data_name='math'),
    'TIGER-Lab/MMLU-Pro': lambda x: extract_answer(x, data_name='mmlu'),
    'Idavidrein/gpqa': lambda x: extract_answer(x, data_name='gpqa'),
    'isaiahbjork/cot-logic-reasoning': lambda x: _strip(x),
    'TIGER-Lab/TheoremQA': lambda x: extract_theoremqa_answer(x),
    'TheoremQAFalse': lambda x: extract_theoremqa_answer(x, False),
}

RESPONSE_COMPARATOR = {
    'openai/gsm8k': lambda x, y: math_equal(x, y, timeout=True),
    'hendrycks/competition_math': lambda x, y: math_equal(x, y, timeout=True),
    'di-zhang-fdu/MATH500': lambda x, y: math_equal(x, y, timeout=True),
    'datasets/compression_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'datasets/converted_aime_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'opencompass/AIME2025': lambda x, y: math_equal(x, y, timeout=True),
    'TIGER-Lab/MMLU-Pro': lambda x, y: symbolic_equal(x, y),
    'Idavidrein/gpqa': lambda x, y: symbolic_equal(x, y),
    'isaiahbjork/cot-logic-reasoning': lambda x, y: float(rouge_l_f1_score(x, y)),
    'TIGER-Lab/TheoremQA': lambda x, y: symbolic_equal(x, y),
}
