Register components
===================

Compiler-owned components use explicit typed registers and the existing
``circuit.append(op)`` execution entry. Each component's evidence distinguishes
emitted primitive operations from declared resource estimates.

.. toctree::
   :maxdepth: 1

   ../REGISTER_CONSTANT_ADDITION
   ../REGISTER_INVERSE_ADDITION
   ../REGISTER_CONTROLLED_ADDITION
   ../REGISTER_SUBTRACTION
   ../REGISTER_EQUALITY
   ../REGISTER_COMPARISON
   ../REGISTER_GREATER
   ../REGISTER_HALF_COMPARISON
   ../REGISTER_CONTROLLED_COMPARISON
   ../REGISTER_LESS_EQUAL
   ../REGISTER_HAMMING_WEIGHT
   ../REGISTER_MODULAR_ARITHMETIC
   ../REGISTER_PERMUTATION
   ../BINARY_FIELD_ARITHMETIC
