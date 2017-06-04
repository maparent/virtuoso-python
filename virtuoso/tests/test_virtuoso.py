from builtins import object
import pyodbc

from . import rdflib_connection

class Test01Virtuoso(object):
    def test_01_is_virtuoso(self):
        conn = pyodbc.connect(rdflib_connection)
        assert conn.getinfo(pyodbc.SQL_DBMS_NAME) == "OpenLink Virtuoso"
