Small module for the generation of Sobol' sequences.

"Heavily inspired" by https://github.com/naught101/sobol_seq which itself is based on the code by John Burkardt and Corrado Chisari (http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html).

# Key features
- Simple OOP interface
- Leaping and skipping
- Up to 40 dimensions
- Up to 2<sup>30</sup> - 1 elements

# Examples
```{python}
from sobol_gen import SobolGenerator

# Make a generator for a 2D sequence
gen = SobolGenerator(2)

# Produce 100 elements of the sequence
output = gen.generate(100)

# Produce 100 elements, skipping the first 10
output = gen.generate(100, seed = 10)

# Produce 100 elements, skipping every second element
output = gen.generate(100, leap = 1)
```
