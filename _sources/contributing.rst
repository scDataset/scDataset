Contributing
============

We welcome contributions to ``scDataset``! This document outlines how to contribute to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/scDataset.git
      cd scDataset

3. **Create a development environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

4. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Setup
-----------------

Install development dependencies:

.. code-block:: bash

   pip install -e ".[dev,test,docs]"

This installs:

* **Core dependencies**: ``torch``, ``numpy``
* **Development tools**: ``black``, ``flake8``, ``mypy``
* **Testing**: ``pytest``, ``pytest-cov``
* **Documentation**: ``sphinx``, ``sphinx-book-theme``

Code Style
----------

We use several tools to maintain code quality:

**Black** for code formatting:

.. code-block:: bash

   black src/ tests/

**Flake8** for linting:

.. code-block:: bash

   flake8 src/ tests/

**MyPy** for type checking:

.. code-block:: bash

   mypy src/

**Pre-commit hooks** (optional but recommended):

.. code-block:: bash

   pre-commit install

Testing
-------

Run the test suite:

.. code-block:: bash

   pytest

Run with coverage:

.. code-block:: bash

   pytest --cov=scdataset --cov-report=html

Test specific modules:

.. code-block:: bash

   pytest tests/test_strategy.py

Writing Tests
~~~~~~~~~~~~~

* Place tests in the ``tests/`` directory
* Use descriptive test names: ``test_streaming_strategy_returns_correct_indices``
* Test both success and failure cases
* Add tests for any new functionality

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   make html

View the built documentation:

.. code-block:: bash

   open build/html/index.html  # On macOS
   # On Linux: xdg-open build/html/index.html

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

* Use **reStructuredText** format for documentation files
* Add **docstrings** to all public functions and classes
* Follow **NumPy docstring style**
* Include **examples** in docstrings when helpful

Example docstring:

.. code-block:: python

   def my_function(param1: int, param2: str = "default") -> bool:
       """
       Brief description of the function.

       Parameters
       ----------
       param1 : int
           Description of param1.
       param2 : str, default="default"
           Description of param2.

       Returns
       -------
       bool
           Description of return value.

       Examples
       --------
       >>> my_function(42, "test")
       True
       """

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* **Clear description** of the problem
* **Minimal example** to reproduce the issue
* **System information** (OS, Python version, package versions)
* **Expected vs actual behavior**

Feature Requests
~~~~~~~~~~~~~~~~

For new features:

* **Describe the use case** and motivation
* **Provide examples** of how it would be used
* **Consider backwards compatibility**

Code Contributions
~~~~~~~~~~~~~~~~~~

* **Start with an issue** to discuss the change
* **Keep changes focused** - one feature/fix per PR
* **Add tests** for new functionality
* **Update documentation** as needed
* **Follow code style** guidelines

Pull Request Process
--------------------

1. **Create an issue** first (unless it's a small fix)
2. **Fork and clone** the repository
3. **Create a feature branch**
4. **Make your changes**:
   
   * Write code
   * Add tests
   * Update documentation
   * Run tests and style checks

5. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "feat: add new sampling strategy"

6. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/your-feature-name

7. **Create a Pull Request** on GitHub

Commit Message Guidelines
-------------------------

We follow `Conventional Commits <https://www.conventionalcommits.org/>`_:

* ``feat:``: New feature
* ``fix:``: Bug fix
* ``docs:``: Documentation changes
* ``test:``: Adding tests
* ``refactor:``: Code refactoring
* ``style:``: Code style changes
* ``ci:``: CI/CD changes

Examples:

.. code-block:: bash

   feat: add weighted sampling strategy
   fix: resolve memory leak in block shuffling
   docs: improve quickstart examples
   test: add integration tests for scDataset

Review Process
--------------

All submissions require review. We use GitHub pull requests for this purpose.

* **Automated checks** must pass (tests, linting, etc.)
* **At least one maintainer** must approve
* **All conversations** must be resolved
* **Documentation** must be updated if needed

Release Process
---------------

Releases are handled by maintainers:

1. **Version bump** following semantic versioning
2. **Update changelog**
3. **Create GitHub release**
4. **Publish to PyPI**

Community Guidelines
--------------------

* **Be respectful** and inclusive
* **Follow the code of conduct**
* **Help others** and share knowledge
* **Stay on topic** in discussions

Getting Help
------------

* **GitHub Issues**: Bug reports and feature requests
* **GitHub Discussions**: Questions and general discussion
* **Documentation**: Check the docs first!

Thank you for contributing to ``scDataset``! 🎉
