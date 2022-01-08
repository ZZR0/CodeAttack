"""
Word Embedding Distance
--------------------------
"""

from codeattack.constraints import Constraint
import re

def classify_tok(tok):
    PY_KEYWORDS = re.compile(
      r'^(False|class|finally|is|return|None|continue|for|lambda|try|True|def|from|nonlocal|while|and|del|global|not|with|as|elif|if|or|yield|assert|else|import|pass|break|except|in|raise)$'
    )

    JAVA_KEYWORDS = re.compile(
      r'^(abstract|assert|boolean|break|byte|case|catch|char|class|continue|default|do|double|else|enum|exports|extends|final|finally|float|for|if|implements|import|instanceof|int|interface|long|module|native|new|package|private|protected|public|requires|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|void|volatile|while)$'
    )

    C_KEYWORDS = re.compile(
      r'^(auto|short|int|long|float|double|char|struct|union|enum|typedef|const|unsigned|signed|extern|register|static|volatile|void|if|else|switch|case|for|do|while|goto|continue|break|default|sizeof|return)$'
    )

    NUMBER = re.compile(
      r'^\d+(\.\d+)?$'
    )

    BRACKETS = re.compile(
      r'^(\{|\(|\[|\]|\)|\})$'
    )

    OPERATORS = re.compile(
      r'^(=|!=|<=|>=|<|>|\?|!|\*|\+|\*=|\+=|/|%|@|&|&&|\||\|\|)$'
    )

    PUNCTUATION = re.compile(
      r'^(;|:|\.|,)$'
    )

    WORDS = re.compile(
      r'^(\w+)$'
    )


    if PY_KEYWORDS.match(tok):
        return 'KEYWORD'
    elif JAVA_KEYWORDS.match(tok):
        return 'KEYWORD'
    elif C_KEYWORDS.match(tok):
        return 'KEYWORD'
    elif NUMBER.match(tok):
        return 'NUMBER'
    elif BRACKETS.match(tok):
        return 'BRACKET'
    elif OPERATORS.match(tok):
        return 'OPERATOR'
    elif PUNCTUATION.match(tok):
        return 'PUNCTUATION'
    elif WORDS.match(tok):
        return 'WORDS'
    else:
        return 'OTHER'

class KeyWord(Constraint):

    def __init__(
        self,
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply key word constraint without `newly_modified_indices`"
            )

        # FIXME The index i is sometimes larger than the number of tokens - 1
        if any(
            i >= len(reference_text.words) or i >= len(transformed_text.words)
            for i in indices
        ):
            return False

        for i in indices:
            transformed_word = transformed_text.words[i]
            if '" != "' in transformed_word:
                transformed_word = transformed_word.replace('" != "', "")[1:-1]
            if '" == "' in transformed_word:
                transformed_word = transformed_word.replace('" == "', "")[1:-1]
            if classify_tok(transformed_word) != 'WORDS':
                return False

        return True
