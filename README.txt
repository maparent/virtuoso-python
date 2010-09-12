OpenLink Virtuoso Support for SQLAlchemy
========================================

This package defines a SQLAlchemy dialect for using
Virtuoso. Note that this is in the very early stages
of development and is probably not useful for anything
yet.

It requires unixODBC. A simple configuration is to 
have an ''/etc/unixODBC/odbcinst.ini'' containing::

    [VirtuosoODBC]
    Description = Virtuoso Universal Server
    Driver      = /usr/lib/virtodbc.so

and an ''/etc/unixODBC/odbc.ini'' or ''~/.odbc.ini''
containing::

    [VOS]
    Description = Virtuoso
    Driver      = VirtuosoODBC
    Servername  = localhost
    Port        = 1111

You can then create a SQLAlchemy engine by doing the
following:

.. code-block:: python

    from sqlalchemy.engine import create_engine
    engine = create_engine("virtuoso://dba:dba@VOS")

OpenLink Virtuoso Support for RDFLib
====================================

To be continued...
