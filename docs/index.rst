FlagQuantum
===========

A PyTorch-first framework for differentiable quantum computing and quantum AI.
Native programs share circuit construction, register components, compilation,
execution and resource inspection through the existing FlagQuantum interfaces.
The :doc:`reference/PUBLICATION_VALIDATION` records tested workflows and their
current evidence boundaries.

.. toctree::
   :maxdepth: 2
   :hidden:

   Documentation <README>
   Concepts <concepts/README>
   Register components <reference/registers/index>
   API and contracts <reference/README>
   Guides <guides/README>
   Development <development/README>

Installation
------------

Use a supported Python environment and install the native package.

.. code-block:: console

   python -m pip install flagquantum

For source development, use the repository's existing development extra.

.. code-block:: console

   git clone https://github.com/flagos-ai/FlagQuantum.git
   cd FlagQuantum
   python -m pip install -e '.[dev]'

The :doc:`development/DEPENDENCY_POLICY` defines supported Python and PyTorch
versions and optional provider dependencies. The :doc:`development/TESTING`
manual describes local and hardware acceptance. Existing diagram exporters
record their optional Graphviz requirements at their owning tool boundary.

First circuit
-------------

.. code-block:: python

   import flagquantum as fq

   circuit = fq.Circuit(2).h(0).cx(0, 1)
   state = circuit.state()
   assert state.shape[-1] == 4

Register algorithms compose through ``circuit.append(op)``. Continue with the
:doc:`reference/registers/index`, :doc:`reference/API` or
:doc:`reference/KNOWN_LIMITATIONS` for the workflow and its supported scope.
