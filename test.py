import re
pattern = re.compile(r'@R_\d+@')
s= 'aqiuefnka@R_1@wiefsidjf@R_14@soidjf'
r = [(m.span()) for m in re.finditer(pattern, s)]
tokens = []
prev = 0
end = len(s)
for span in r:
    tokens += [s[prev:span[0]]]
    tokens += [s[span[0]:span[1]]]
    prev = span[1]
if prev < end:
    tokens += [s[prev:end]]

out = {}
for idx, token in enumerate(tokens):
    if pattern.match(token) is not None:
        id = int(token[3:-1])
        if id in out: 
            out[id] += [idx]
        else:
            out[id] = [idx]
print(out)
print(tokens)