from math import factorial as fact
from math import log2

num_combs = lambda n, k: fact(n) // (fact(k) * fact(n - k))
num_msets = lambda n, k: fact(n + k - 1, k)

def re():
    exec(open('repl.py').read(), globals())
