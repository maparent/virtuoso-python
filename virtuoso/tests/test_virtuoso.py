import pyodbc

class Test01Virtuoso(object):
    def test_01_is_virtuoso(self):
        conn = pyodbc.connect("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
        assert conn.getinfo(pyodbc.SQL_DBMS_NAME) == "OpenLink Virtuoso"
