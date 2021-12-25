"""
SentenceTransformation class
-----------------------------------

https://github.com/makcedward/nlpaug

"""


from codeattack.transformations import Transformation


class SentenceTransformation(Transformation):
    def _get_transformations(self, current_text, indices_to_modify):
        raise NotImplementedError()
