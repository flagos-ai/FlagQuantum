FlagQuantum
===========

.. Navigation and installation roles adapted from Qualtran v0.7.0 docs/index.rst,
   Google LLC, Apache-2.0. Native policy and interfaces remain authoritative.

A PyTorch-first framework for differentiable quantum computing and quantum AI.
Native programs share circuit construction, register components, compilation,
execution and resource inspection through the existing FlagQuantum interfaces.
The :doc:`reference/PUBLICATION_VALIDATION` records tested workflows and their
current evidence boundaries.

Fundamentals
------------

The :doc:`concepts/REGISTER_PROGRAMS` introduces typed register operations,
composite programs, decomposition, simulation, diagrams and resource reasoning.
Its tutorials use native FlagQuantum components and the existing
``circuit.append(op)`` entry point.

Component library
-----------------

The :doc:`reference/components` connects the complete published notebook
catalog to its domain-owned operations and examples. It covers reusable
primitives and higher-level algorithms. Individual pages retain their supported
domains, original-source discrepancies and execution evidence; inclusion in the
catalog does not certify arbitrary inputs or unfinished algorithm bodies.

API reference
-------------

Use :doc:`reference/API` for the curated Python interface and
:doc:`reference/README` for contracts and support policies. The
:doc:`generated/STABLE_API` records protected exports. Component examples and generated
reference pages have their own acceptance scopes; a successful documentation
build does not prove all numerical or hardware capabilities.

.. toctree::
   :maxdepth: 2
   :hidden:

   Documentation <README>
   Concepts <concepts/README>
   Fundamentals <concepts/REGISTER_PROGRAMS>
   Component library <reference/components>
   Register components <reference/registers/index>
   API and contracts <reference/README>
   Guides <guides/README>
   Development <development/README>

Installation
------------

Use CPython 3.10 through 3.12 and install the native package.

.. code-block:: console

   python -m pip install flagquantum

An isolated Python environment keeps project dependencies separate. For example,
with an existing CPython 3.12 installation on a POSIX shell:

.. code-block:: console

   python3.12 -m venv .venv
   . .venv/bin/activate
   python -m pip install flagquantum

On Windows, activate the environment with its ``Scripts`` activation script
appropriate to your shell. Activate the chosen environment in each new terminal
before following the installation or development commands. Existing environment
managers may be used instead; FlagQuantum does not install or own them.

For source development, use the repository's existing development extra.

.. code-block:: console

   git clone https://github.com/flagos-ai/FlagQuantum.git
   cd FlagQuantum
   python -m pip install -e '.[dev]'

The :doc:`development/DEPENDENCY_POLICY` defines supported Python and PyTorch
versions and optional provider dependencies. The :doc:`development/TESTING`
manual describes local and hardware acceptance. Existing diagram exporters
record their optional Graphviz requirements at their owning tool boundary.
Graphviz-backed renderers need the external executable as well as their Python
adapter; installing the core package alone does not install that executable.
Native execution does not require Graphviz. Optional providers and documentation
tools remain separate from the core installation.

First circuit
-------------

.. code-block:: python

   import flagquantum as fq

   circuit = fq.Circuit(2).h(0).cx(0, 1)
   state = circuit.state()
   assert state.shape[-1] == 4

Register algorithms compose through ``circuit.append(op)``. Continue with the
:doc:`concepts/REGISTER_PROGRAMS`, :doc:`reference/components`,
:doc:`reference/registers/index`, :doc:`reference/API` or
:doc:`reference/KNOWN_LIMITATIONS` for the workflow and its supported scope.

For feedback, report a reproducible input and environment through the
`GitHub issue tracker <https://github.com/flagos-ai/FlagQuantum/issues>`_.
Consult the current limitations and release notes before using an estimate as
a physical execution claim. Release communication follows the project's
reviewed release notes and authorized channels.
