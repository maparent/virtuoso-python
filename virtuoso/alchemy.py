assert __import__("pkg_resources").get_distribution("sqlalchemy").version.split('.') >= ['0', '6'], \
    "requires sqlalchemy version 0.6 or greater"

from sqlalchemy.connectors.pyodbc import PyODBCConnector
from sqlalchemy.dialects.sybase.base import SybaseDialect
from sqlalchemy.sql import text, bindparam
from sqlalchemy.engine import default


class VirtuosoExecutionContext(default.DefaultExecutionContext):
    def get_lastrowid(self):
        self.cursor.execute("SELECT identity_value() AS lastrowid")
        lastrowid = self.cursor.fetchone()[0]
        return int(lastrowid)


class VirtuosoDialect(PyODBCConnector, SybaseDialect):
    execution_ctx_cls = VirtuosoExecutionContext
    name = 'virtuoso'

    def initialize(self, connection):
        self.supports_unicode_statements = False
        self.supports_unicode_binds = False
        SybaseDialect.initialize(self, connection)

    def _get_default_schema_name(self, connection):
        return 'DBA'

    def has_table(self, connection, tablename, schema=None):
        if schema is None:
            schema = self.default_schema_name
        result = connection.execute(
            text("SELECT TABLE_NAME FROM TABLES WHERE "
                 "TABLE_SCHEMA=:schemaname AND "
                 "TABLE_NAME=:tablename",
                 bindparams=[
                     bindparam("schemaname", schema),
                     bindparam("tablename", tablename)
                     ])
            )
        return result.scalar() is not None

    def get_table_names(self, connection, schema=None, **kw):
        if schema is None:
            schema = self.default_schema_name
        result = connection.execute(
            text("SELECT TABLE_NAME FROM TABLES WHERE TABLE_SCHEMA=:schemaname",
                 bindparams=[bindparam("schemaname", schema)])
            )
        return [r[0] for r in result]
