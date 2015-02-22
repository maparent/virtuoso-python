
Virtuoso from Python
====================

This package is intended to be a collection of modules for
interacting with `OpenLink Virtuoso`_ from python.

The goal is to provide drivers for `SQLAlchemy` and `RDFLib`.

At the time of this writing it requires a fork of pyodbc, which you can find on this branch:
https://github.com/maparent/pyodbc/tree/v3-virtuoso

For more information see http://packages.python.org/virtuoso

.. _OpenLink Virtuoso: http://virtuoso.openlinksw.com
.. _SQLAlchemy: http://www.sqlalchemy.org/
.. _RDFLib: http://rdflib.net/

This package also contains an experimental generator of Virtuoso
Linked Data Views, which was presented as a poster at ISWC2014:
http://ceur-ws.org/Vol-1272/paper_60.pdf

The original design of this library is by `William Waites`, and development is continued by `Marc-Antoine Parent` for the `Assembl` project.

.. _William Waites: https://bitbucket.org/ww/virtuoso/src
.. _Marc-Antoine Parent: https://github.com/maparent/virtuoso-python
.. _Assembl: https://github.com/imaginationforpeople/assembl
