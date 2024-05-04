.. Adiabatic Compiler documentation master file, created by
   sphinx-quickstart on Sat Jan 27 16:30:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Adiabatic Compiler's documentation!
==============================================

Introduction
------------

**Adiabatic Compiler** is a Python package that aims to implement the theory
of the translations from an arbritrary gate-based quantum circuit into an adibatic program, 
which we can run it as adibatic quantum computation.

The translations from an gate-based quantum computation to adibatic quantum computation 
are mostly in a theoritical stage but have been proved to work. It motivates us to
seek the practicality and applicability of the theories.
So I chose to implement the translations from the foundational paper (1) which proved
the equivalence between gate-based quantum computation and adibatic quantum computation, 
and another recently published paper (2) with connections with the foundational paper:

1. `D. Aharonov, W. van Dam, J. Kempe, Z. Landau, S. Lloyd, and O. Regev,
“Adiabatic quantum computation is equivalent to standard quantum com-
putation,” 2005. <https://arxiv.org/abs/quant-ph/0405098>`_

2. `A. Anshu, N. P. Breuckmann, and Q. T. Nguyen, “Circuit-to-hamiltonian
from tensor networks and fault tolerance,” 2023. <https://arxiv.org/abs/2309.16475>`_

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/installation
   tutorial/get_started


.. toctree::
   :maxdepth: 2
   :caption: Modules

   interpreter
   frontend
   language
   backend

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
