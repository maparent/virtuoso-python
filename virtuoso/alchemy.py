assert __import__("pkg_resources").get_distribution(
    "sqlalchemy").version.split('.') >= ['0', '6'], \
    "requires sqlalchemy version 0.6 or greater"

import warnings
import pdb

import uricore
from sqlalchemy import schema, Table
from sqlalchemy.schema import Constraint
from sqlalchemy.sql import text, bindparam, compiler, operators
from sqlalchemy.sql.expression import BindParameter
from sqlalchemy.connectors.pyodbc import PyODBCConnector
from sqlalchemy.engine import default
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import (
    CHAR, VARCHAR, TIME, NCHAR, NVARCHAR, TEXT, DATETIME, FLOAT, String,
    NUMERIC, BIGINT, INT, INTEGER, SMALLINT, BINARY, VARBINARY, DECIMAL,
    TIMESTAMP, UnicodeText, REAL, Text, Float, Binary, UserDefinedType)
from sqlalchemy.orm import column_property
from sqlalchemy.ext.hybrid import hybrid_property


class VirtuosoExecutionContext(default.DefaultExecutionContext):
    def get_lastrowid(self):
        self.cursor.execute("SELECT identity_value() AS lastrowid")
        lastrowid = int(self.cursor.fetchone()[0])
        #print "idvalue: %d, lser: %d" % (lastrowid, self.cursor.lastserial)
        return lastrowid


RESERVED_WORDS = set([
    '__cost', '__elastic', '__tag', '__soap_doc', '__soap_docw',
    '__soap_header', '__soap_http', '__soap_name', '__soap_type',
    '__soap_xml_type', '__soap_fault', '__soap_dime_enc', '__soap_enc_mime',
    '__soap_options', 'ada', 'add', 'admin', 'after', 'aggregate', 'all',
    'alter', 'and', 'any', 'are', 'array', 'as', 'asc', 'assembly', 'attach',
    'attribute', 'authorization', 'autoregister', 'backup', 'before', 'begin',
    'best', 'between', 'bigint', 'binary', 'bitmap', 'breakup', 'by', 'c',
    'call', 'called', 'cascade', 'case', 'cast', 'char', 'character', 'check',
    'checked', 'checkpoint', 'close', 'cluster', 'clustered', 'clr',
    'coalesce', 'cobol', 'collate', 'column', 'commit', 'committed',
    'compress', 'constraint', 'constructor', 'contains', 'continue',
    'convert', 'corresponding', 'create', 'cross', 'cube', 'current',
    'current_date', 'current_time', 'current_timestamp', 'cursor', 'data',
    'date', 'datetime', 'decimal', 'declare', 'default', 'delete', 'desc',
    'deterministic', 'disable', 'disconnect', 'distinct', 'do', 'double',
    'drop', 'dtd', 'dynamic', 'else', 'elseif', 'enable', 'encoding', 'end',
    'escape', 'except', 'exclusive', 'execute', 'exists', 'external',
    'extract', 'exit', 'fetch', 'final', 'float', 'for', 'foreach', 'foreign',
    'fortran', 'for_vectored', 'for_rows', 'found', 'from', 'full',
    'function', 'general', 'generated', 'go', 'goto', 'grant', 'group',
    'grouping', 'handler', 'having', 'hash', 'identity', 'identified', 'if',
    'in', 'incremental', 'increment', 'index', 'index_no_fill', 'index_only',
    'indicator', 'inner', 'inout', 'input', 'insert', 'instance', 'instead',
    'int', 'integer', 'intersect', 'internal', 'interval', 'into', 'is',
    'isolation', 'iri_id', 'iri_id_8', 'java', 'join', 'key', 'keyset',
    'language', 'left', 'level', 'library', 'like', 'locator', 'log', 'long',
    'loop', 'method', 'modify', 'modifies', 'module', 'mumps', 'name',
    'natural', 'nchar', 'new', 'nonincremental', 'not', 'no', 'novalidate',
    'null', 'nullif', 'numeric', 'nvarchar', 'object_id', 'of', 'off', 'old',
    'on', 'open', 'option', 'or', 'order', 'out', 'outer', 'overriding',
    'partition', 'pascal', 'password', 'percent', 'permission_set',
    'persistent', 'pli', 'position', 'precision', 'prefetch', 'primary',
    'privileges', 'procedure', 'public', 'purge', 'quietcast', 'rdf_box',
    'read', 'reads', 'real', 'ref', 'references', 'referencing', 'remote',
    'rename', 'repeatable', 'replacing', 'replication', 'resignal',
    'restrict', 'result', 'return', 'returns', 'revoke', 'rexecute', 'right',
    'rollback', 'rollup', 'role', 'safe', 'same_as', 'uncommitted',
    'unrestricted', 'schema', 'select', 'self', 'serializable', 'set', 'sets',
    'shutdown', 'smallint', 'snapshot', 'soft', 'some', 'source', 'sparql',
    'specific', 'sql', 'sqlcode', 'sqlexception', 'sqlstate', 'sqlwarning',
    'static', 'start', 'style', 'sync', 'system', 't_cycles_only',
    't_direction', 't_distinct', 't_end_flag', 't_exists', 't_final_as',
    't_in', 't_max', 't_min', 't_no_cycles', 't_no_order', 't_out',
    't_shortest_only', 'table', 'temporary', 'text', 'then', 'ties', 'time',
    'timestamp', 'to', 'top', 'type', 'transaction', 'transitive', 'trigger',
    'under', 'union', 'unique', 'update', 'use', 'user', 'using', 'validate',
    'value', 'values', 'varbinary', 'varchar', 'variable', 'vector',
    'vectored', 'view', 'when', 'whenever', 'where', 'while', 'with',
    'without', 'work', 'xml', 'xpath'])


class VirtuosoIdentifierPreparer(compiler.IdentifierPreparer):
    reserved_words = RESERVED_WORDS

    def quote_schema(self, schema, force=None):
        if '.' in schema:
            cat, schema = schema.split('.', 1)
            return self.quote(cat, force) + '.' + self.quote(schema, force)
        else:
            # Virtuoso needs an extra dot to indicate absent username
            return self.quote(schema, force) + '.'


class VirtuosoSQLCompiler(compiler.SQLCompiler):
    ansi_bind_rules = True
    extract_map = {
        'day': 'dayofmonth(%s)',
        'dow': 'dayofweek(%s)',
        'doy': 'dayofyear(%s)',
        'epoch': 'msec_time()',
        'hour': 'hour(%s)',
        'microseconds': '0',
        'milliseconds': 'atoi(substring(datestring(%s), 20, 6))',
        'minute': 'minute(%s)',
        'month': 'month(%s)',
        'quarter': 'quarter(%s)',
        'second': 'second(%s)',
        'timezone_hour': 'floor(timezone(%s)/60)',
        'timezone_minute': 'mod(timezone(%s),60)',
        'week': 'week(%s)',
        'year': 'year(%s)'
    }

    def get_select_precolumns(self, select):
        s = select._distinct and "DISTINCT " or ""
        # TODO: check if Virtuoso supports
        # bind params for FIRST / TOP
        if select._limit or select._offset:
            if select._offset:
                limit = select._limit or '100000'
                s += "TOP %s, %s " % (limit, select._offset + 1)
            else:
                s += "TOP %s " % (select._limit,)
        return s

    def limit_clause(self, select):
        # Limit in virtuoso is after the select keyword
        return ""

    def visit_now_func(self, fn, **kw):
        return "GETDATE()"

    def visit_extract(self, extract, **kw):
        func = self.extract_map.get(extract.field)
        if not func:
            raise exc.CompileError(
                "%s is not a valid extract argument." % extract.field)
        return func % (self.process(extract.expr, **kw), )

    def visit_true(self, expr, **kw):
        return '1'

    def visit_false(self, expr, **kw):
        return '0'

    def visit_binary(self, binary, **kwargs):
        if binary.operator == operators.ne:
            if isinstance(binary.left, BindParameter)\
                    and isinstance(binary.right, BindParameter):
                kwargs['literal_binds'] = True
            return self._generate_generic_binary(
                binary, ' <> ', **kwargs)

        return super(VirtuosoSQLCompiler, self).visit_binary(binary, **kwargs)

    def render_literal_value(self, value, type_):
        if isinstance(value, IRI_ID_Literal):
            return value
        return super(VirtuosoSQLCompiler, self)\
            .render_literal_value(value, type_)

    def visit_bindparam(self, bindparam, within_columns_clause=False,
                        literal_binds=False,
                        skip_bind_expression=False,
                        **kwargs):
        return super(VirtuosoSQLCompiler, self)\
            .visit_bindparam(bindparam, within_columns_clause,
                             literal_binds, skip_bind_expression,
                             **kwargs)


class LONGVARCHAR(Text):
    __visit_name__ = 'LONG VARCHAR'


class LONGNVARCHAR(UnicodeText):
    __visit_name__ = 'LONG NVARCHAR'


class DOUBLEPRECISION(Float):
    __visit_name__ = 'DOUBLE PRECISION'


class LONGVARBINARY(Binary):
    __visit_name__ = 'LONG VARBINARY'


class IRI_ID_Literal(str):
    "An internal virtuoso IRI ID, of the form #innnnn"
    def __repr__(self):
        return 'IRI_ID_Literal("%s")' % (self, )


class IRI_ID(UserDefinedType):
    "A column type for IRI ID"
    __visit_name__ = 'IRI_ID'

    def __init__(self):
        super(IRI_ID, self).__init__()

    def get_col_spec(self):
        return "IRI_ID"

    def bind_processor(self, dialect):
        def process(value):
            if value:
                return IRI_ID_Literal(value)
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value:
                return IRI_ID_Literal(value)
        return process


class iri_id_num(GenericFunction):
    "Convert IRI IDs to int values"
    type = INTEGER
    name = "iri_id_num"

    def __init__(self, iri_id, **kw):
        if not isinstance(iri_id, IRI_ID_Literal)\
                and not isinstance(iri_id.__dict__.get('type'), IRI_ID):
            warnings.warn("iri_id_num() accepts an IRI_ID object as input.")
        super(iri_id_num, self).__init__(iri_id, **kw)


class iri_id_from_num(GenericFunction):
    "Convert numeric IRI IDs to IRI ID literal type"
    type = IRI_ID
    name = "iri_id_from_num"

    def __init__(self, num, **kw):
        if not isinstance(num, int)\
                and not isinstance(iri_id.__dict__.get('type'), Integer):
            warnings.warn("iri_id_num() accepts an Integer as input.")
        super(iri_id_from_num, self).__init__(num, **kw)


class id_to_iri(GenericFunction):
    "Get the IRI from a given IRI ID"
    type = String
    name = "id_to_iri"

    def __init__(self, iri_id, **kw):
        if not isinstance(iri_id, IRI_ID_Literal)\
                and not isinstance(iri_id.__dict__.get('type'), IRI_ID):
            warnings.warn("iri_id_num() accepts an IRI_ID as input.")
        super(id_to_iri, self).__init__(iri_id, **kw)


class iri_to_id(GenericFunction):
    """Get an IRI ID from an IRI.
    If the IRI is new to virtuoso, the IRI ID may be created on-the-fly,
    according to the second argument."""
    type = IRI_ID
    name = "iri_to_id"

    def __init__(self, iri, create=True, **kw):
        if isinstance(iri, unicode):
            iri = str(uricore.URI(uricore.IRI(iri)))
        if not isinstance(iri, str)\
                and not isinstance(iri_id.__dict__.get('type'), String):
            warnings.warn("iri_id_num() accepts an IRI (VARCHAR) as input.")
        super(iri_to_id, self).__init__(iri, create, **kw)


def iri_property(iri_id_colname, iri_propname):
    """Class decorator to add access to an IRI_ID column as an IRI.
    The name of the IRI property will be iri_propname."""
    def iri_class_decorator(klass):
        iri_hpropname = '_'+iri_propname
        setattr(klass, iri_hpropname,
                column_property(id_to_iri(getattr(klass, iri_id_colname))))

        def iri_accessor(self):
            return getattr(self, iri_hpropname)

        def iri_expression(klass):
            return id_to_iri(getattr(klass, iri_id_colname))

        def iri_setter(self, val):
            setattr(self, iri_hpropname, val)
            setattr(self, iri_id_colname, iri_to_id(val))

        def iri_deleter(self):
            setattr(self, iri_id_colname, None)

        col = getattr(klass, iri_id_colname)
        if not col.property.columns[0].nullable:
            iri_deleter = None
        prop = hybrid_property(
            iri_accessor, iri_setter, iri_deleter, iri_expression)
        setattr(klass, iri_propname, prop)
        return klass
    return iri_class_decorator


class XML(Text):
    __visit_name__ = 'XML'


class LONGXML(Text):
    __visit_name__ = 'LONG_XML'


class VirtuosoTypeCompiler(compiler.GenericTypeCompiler):
    def visit_boolean(self, type_):
        return self.visit_SMALLINT(type_)

    def visit_unicode(self, type_):
        return self.visit_NVARCHAR(type_)

    def visit_LONGVARCHAR(self, type_):
        return 'LONG VARCHAR'

    def visit_LONGNVARCHAR(self, type_):
        return 'LONG NVARCHAR'

    def visit_DOUBLEPRECISION(self, type_):
        return 'DOUBLE PRECISION'

    def visit_BIGINT(self, type_):
        return "INTEGER"

    def visit_DATE(self, type_):
        return "CHAR(10)"

    def visit_CLOB(self, type_):
        return self.visit_LONGVARCHAR(type_)

    def visit_NCLOB(self, type_):
        return self.visit_LONGNVARCHAR(type_)

    def visit_TEXT(self, type_):
        return self._render_string_type(type_, "LONG VARCHAR")

    def visit_BLOB(self, type_):
        return "LONG VARBINARY"

    def visit_BINARY(self, type_):
        return self.visit_VARBINARY(type_)

    def visit_VARBINARY(self, type_):
        return "VARBINARY" + (type_.length and "(%d)" % type_.length or "")

    def visit_LONGVARBINARY(self, type_):
        return 'LONG VARBINARY'

    def visit_large_binary(self, type_):
        return self.visit_LONGVARBINARY(type_)

    def visit_unicode(self, type_):
        return self.visit_NVARCHAR(type_)

    def visit_text(self, type_):
        return self.visit_TEXT(type_)

    def visit_unicode_text(self, type_):
        return self.visit_LONGNVARCHAR(type_)

    def visit_IRI_ID(self, type_):
        return "IRI_ID"

    def visit_XML(self, type_):
        return "XML"

    def visit_LONG_XML(self, type_):
        return "LONG XML"

    # def visit_user_defined(self, type_):
    # TODO!
    #     return type_.get_col_spec()


class VirtuosoDDLCompiler(compiler.DDLCompiler):
    def get_column_specification(self, column, **kwargs):
        colspec = (self.preparer.format_column(column) + " "
                   + self.dialect.type_compiler.process(column.type))

        if column.nullable is not None:
            if not column.nullable or column.primary_key or \
                    isinstance(column.default, schema.Sequence):
                colspec += " NOT NULL"
            else:
                colspec += " NULL"

        if column.table is None:
            raise exc.CompileError(
                "virtuoso requires Table-bound columns "
                "in order to generate DDL")

        # install an IDENTITY Sequence if we either a sequence
        # or an implicit IDENTITY column
        if isinstance(column.default, schema.Sequence):
            if column.default.start == 0:
                start = 0
            else:
                start = column.default.start or 1

            colspec += " IDENTITY (START WITH %s)" % (start,)
        elif column is column.table._autoincrement_column:
            colspec += " IDENTITY"
        else:
            default = self.get_column_default_string(column)
            if default is not None:
                colspec += " DEFAULT " + default

        return colspec

    def visit_under_constraint(self, constraint):
        table = constraint.table
        parent_table = constraint.parent_table
        return "UNDER %s.%s " % (
            self.preparer.quote_schema(
                parent_table.schema, table.quote_schema),
            self.preparer.quote(parent_table.name, table.quote))

ischema_names = {
    'bigint': INTEGER,
    'int': INTEGER,
    'integer': INTEGER,
    'smallint': SMALLINT,
    'tinyint': SMALLINT,
    'unsigned bigint': INTEGER,
    'unsigned int': INTEGER,
    'unsigned smallint': SMALLINT,
    'numeric': NUMERIC,
    'decimal': DECIMAL,
    'dec': DECIMAL,
    'float': FLOAT,
    'double': DOUBLEPRECISION,
    'double precision': DOUBLEPRECISION,
    'real': REAL,
    'smalldatetime': DATETIME,
    'datetime': DATETIME,
    'date': CHAR,
    'time': TIME,
    'char': CHAR,
    'character': CHAR,
    'varchar': VARCHAR,
    'character varying': VARCHAR,
    'char varying': VARCHAR,
    'nchar': NCHAR,
    'national char': NCHAR,
    'national character': NCHAR,
    'nvarchar': NVARCHAR,
    'nchar varying': NVARCHAR,
    'national char varying': NVARCHAR,
    'national character varying': NVARCHAR,
    'text': LONGVARCHAR,
    'unitext': LONGNVARCHAR,
    'binary': VARBINARY,
    'varbinary': VARBINARY,
    'long varbinary': LONGVARBINARY,
    'long varchar': LONGVARCHAR,
    'timestamp': TIMESTAMP,
}


# DO NOT USE! Deprecated in Columnar view.
class UnderConstraint(Constraint):
    __visit_name__ = 'under_constraint'

    def __init__(self, parent_table, **kw):
        super(UnderConstraint, self).__init__(**kw)
        if not isinstance(parent_table, Table)\
                and parent_table.__dict__.get('__table__') is not None:
            parent_table = parent_table.__table__
        assert isinstance(parent_table, Table)
        self.parent_table = parent_table


class VirtuosoDialect(PyODBCConnector, default.DefaultDialect):
    name = 'virtuoso'
    execution_ctx_cls = VirtuosoExecutionContext
    preparer = VirtuosoIdentifierPreparer
    statement_compiler = VirtuosoSQLCompiler
    type_compiler = VirtuosoTypeCompiler
    ischema_names = ischema_names
    supports_unicode_statements = False
    supports_unicode_binds = True
    supports_native_boolean = False
    ddl_compiler = VirtuosoDDLCompiler

    def _get_default_schema_name(self, connection):
        res = connection.execute(
            'select U_DEF_QUAL, get_user() from DB.DBA.SYS_USERS where U_NAME=get_user()')
        catalog, schema = res.fetchone()
        if catalog:
            return '.'.join(res.fetchone())

    def has_table(self, connection, tablename, schema=None):
        if schema is None:
            schema = self.default_schema_name
        result = connection.execute(
            text("SELECT TABLE_NAME FROM DB..TABLES WHERE "
                 "TABLE_CATALOG=:schemaname AND "
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
        if schema is None:
            result = connection.execute(
                text("SELECT TABLE_NAME FROM DB..TABLES"))
            return [r[0] for r in result]
        if '.' not in schema:
            schema += '.'
        catalog, schema = schema.split('.', 1)
        if catalog:
            if schema:
                result = connection.execute(
                    text("SELECT TABLE_NAME FROM DB..TABLES WHERE"
                         "TABLE_CATALOG=:catalog AND TABLE_SCHEMA = :schema"),
                    catalog=catalog, schema=schema)
            else:
                result = connection.execute(
                    text("SELECT TABLE_NAME FROM DB..TABLES WHERE"
                         "TABLE_CATALOG=:catalog"), catalog=catalog)
        else:
            result = connection.execute(
                text("SELECT TABLE_NAME FROM DB..TABLES WHERE"
                     "TABLE_SCHEMA=:schema"), schema=schema)
        return [r[0] for r in result]
