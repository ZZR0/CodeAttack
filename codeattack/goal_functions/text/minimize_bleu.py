"""
Goal Function for Attempts to minimize the BLEU score
-------------------------------------------------------


"""

import functools

import nltk

import codeattack

from .text_to_text_goal_function import TextToTextGoalFunction

import sys, math, re, xml.sax.saxutils
import subprocess
import os

# Added to bypass NIST-style pre-processing of hyp and ref files -- wade
nonorm = 0

preserve_case = False
eff_ref_len = "shortest"

normalize1 = [
    ('<skipped>', ''),         # strip "skipped" tags
    (r'-\n', ''),              # strip end-of-line hyphenation and join lines
    (r'\n', ' '),              # join lines
#    (r'(\d)\s+(?=\d)', r'\1'), # join digits
]
normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

normalize2 = [
    (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])',r' \1 '), # tokenize punctuation. apostrophe is missing
    (r'([^0-9])([\.,])',r'\1 \2 '),              # tokenize period and comma unless preceded by a digit
    (r'([\.,])([^0-9])',r' \1 \2'),              # tokenize period and comma unless followed by a digit
    (r'([0-9])(-)',r'\1 \2 ')                    # tokenize dash when preceded by a digit
]
normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]

def normalize(s):
    '''Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl.'''
    # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
    if (nonorm):
        return s.split()
    if type(s) is not str:
        s = " ".join(s)
    # language-independent part:
    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {'&quot;':'"'})
    # language-dependent part (assuming Western languages):
    s = " %s " % s
    if not preserve_case:
        s = s.lower()         # this might not be identical to the original
    for (pattern, replace) in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()

def count_ngrams(words, n=4):
    counts = {}
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] = counts.get(ngram, 0)+1
    return counts

def cook_refs(refs, n=4):
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''
    
    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for (ngram,count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return ([len(ref) for ref in refs], maxcounts)

def cook_test(test, item, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    (reflens, refmaxcounts)=item
    test = normalize(test)
    result = {}
    result["testlen"] = len(test)

    # Calculate effective reference sentence length.
    
    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens))/len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen-len(test)) < min_diff:
                min_diff = abs(reflen-len(test))
                result['reflen'] = reflen

    result["guess"] = [max(len(test)-k+1,0) for k in range(1,n+1)]

    result['correct'] = [0]*n
    counts = count_ngrams(test, n)
    for (ngram, count) in counts.items():
        result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)

    return result

def score_cooked(allcomps, n=4, ground=0, smooth=1):
    totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*n, 'correct':[0]*n}
    for comps in allcomps:
        for key in ['testlen','reflen']:
            totalcomps[key] += comps[key]
        for key in ['guess','correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    all_bleus = []
    for k in range(n):
      correct = totalcomps['correct'][k]
      guess = totalcomps['guess'][k]
      addsmooth = 0
      if smooth == 1 and k > 0:
        addsmooth = 1
      logbleu += math.log(correct + addsmooth + sys.float_info.min)-math.log(guess + addsmooth+ sys.float_info.min)
      if guess == 0:
        all_bleus.append(-10000000)
      else:
        all_bleus.append(math.log(correct + sys.float_info.min)-math.log( guess ))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(0,1-float(totalcomps['reflen'] + 1)/(totalcomps['testlen'] + 1))
    for i in range(len(all_bleus)):
      if i ==0:
        all_bleus[i] += brevPenalty
      all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus

def bleu(refs,  candidate, ground=0, smooth=1):
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], ground=ground, smooth=smooth)

def splitPuncts(line):
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))

def computeMaps(predictions, golds):
    predictionMap = {}
    goldMap = {}
    for rid, (pred, gold) in enumerate(zip(predictions, golds)):
        predictionMap[rid] = [splitPuncts(pred.strip().lower())]
        if rid not in goldMap:
            goldMap[rid] = []
        goldMap[rid].append(splitPuncts(gold.strip().lower()))

    #   sys.stderr.write('Total: ' + str(len(goldMap)) + '\n')
    return (goldMap, predictionMap)

#m1 is the reference map
#m2 is the prediction map
def bleuFromMaps(m1, m2):
    score = [0] * 5
    num = 0.0

    for key in m1:
        if key in m2:
            bl = bleu(m1[key], m2[key][0])
            score = [ score[i] + bl[i] for i in range(0, len(bl))]
            num += 1
    return [s * 100.0 / num for s in score]
    
class MinimizeBleu(TextToTextGoalFunction):
    """Attempts to minimize the BLEU score between the current output
    translation and the reference translation.

        BLEU score was defined in (BLEU: a Method for Automatic Evaluation of Machine Translation).

        `ArxivURL`_

    .. _ArxivURL: https://www.aclweb.org/anthology/P02-1040.pdf

        This goal function is defined in (It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations).

        `ArxivURL2`_

    .. _ArxivURL2: https://www.aclweb.org/anthology/2020.acl-main.263
    """

    EPS = 1e-10

    def __init__(self, *args, target_bleu=0.0, **kwargs):
        self.target_bleu = target_bleu
        super().__init__(*args, **kwargs)

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()
        get_bleu.cache_clear()

    def _is_goal_complete(self, model_output, _):
        bleu_score = 1.0 - self._get_score(model_output, _)
        return bleu_score <= (self.target_bleu + MinimizeBleu.EPS)

    def _get_score(self, model_output, _):
        model_output_at = model_output
        ground_truth_at = self.ground_truth_output
        bleu_score = get_bleu(model_output_at, ground_truth_at)
        return 1.0 - bleu_score
    
    def _process_model_outputs(self, _, outputs):
        """Processes and validates a list of model outputs."""
        return outputs

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_bleu"]


@functools.lru_cache(maxsize=2 ** 12)
def get_bleu(a, b):
    predictions, golds = [a], [b]
    (goldMap, predictionMap) = computeMaps(predictions, golds) 
    bleu_score = bleuFromMaps(goldMap, predictionMap)[0]
    return bleu_score
