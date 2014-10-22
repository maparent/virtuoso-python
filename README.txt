
Virtuoso from Python
====================

This package is intended to be a collection of modules for
interacting with `OpenLink Virtuoso`_ from python.

The goal is to provide drivers for `SQLAlchemy` and `RDFLib`.

At the time of this writing it requires a patch for pyodbc
as documented in: http://river.styx.org/ww/2010/10/pyodbc-spasql/index
You can also get a pyodbc branch here:
http://github.com/maparent/pyodbc/tarball/v3-virtuoso

For more information see http://packages.python.org/virtuoso

.. _OpenLink Virtuoso: http://virtuoso.openlinksw.com
.. _SQLAlchemy: http://www.sqlalchemy.org/
.. _RDFLib: http://rdflib.net/

This package also contains an experimental generator of Virtuoso
Linked Data Views, which was presented as a poster at ISWC2014:
http://ceur-ws.org/Vol-1272/paper_60.pdf
