assert __import__("pkg_resources").get_distribution(
    "sqlalchemy").version.split('.') >= ['0', '6'], \
    "requires sqlalchemy version 0.6 or greater"

import warnings
from datetime import datetime

from werkzeug.urls import iri_to_uri
from sqlalchemy import schema, Table, exc, util
from sqlalchemy.schema import Constraint
from sqlalchemy.sql import (text, bindparam, compiler, operators)
from sqlalchemy.sql.expression import (
    BindParameter, TextClause, cast, ColumnElement)
from sqlalchemy.sql.schema import Sequence
from sqlalchemy.sql.compiler import BIND_PARAMS, BIND_PARAMS_ESC
from sqlalchemy.sql.ddl import _CreateDropBase
from sqlalchemy.connectors.pyodbc import PyODBCConnector
from sqlalchemy.engine import default
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import (
    CHAR, VARCHAR, TIME, NCHAR, NVARCHAR, DATETIME, FLOAT, String, NUMERIC,
    INTEGER, SMALLINT, VARBINARY, DECIMAL, TIMESTAMP, UnicodeText, REAL,
    Unicode, Text, Float, Binary, UserDefinedType, TypeDecorator)
from sqlalchemy.orm import column_property
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.elements import Grouping, ClauseList
import past.builtins


class VirtuosoExecutionContext(default.DefaultExecutionContext):
    def get_lastrowid(self):
        return self.cursor.lastserial

    def fire_sequence(self, seq, type_):
        return self._execute_scalar((
            "select sequence_next('%s')" %
            self.dialect.identifier_preparer.format_sequence(seq)), type_)


class VirtuosoSequence(Sequence):
    def upcoming_value(self, connection):
        # This gives the upcoming value without advancing the sequence
        preparer = connection.bind.dialect.identifier_preparer
        (val,) = next(iter(connection.execute(
            "SELECT sequence_set('%s', 0, 1)" %
            (preparer.format_sequence(self),))))
        return int(val)

    def set_value(self, value, connection):
        preparer = connection.bind.dialect.identifier_preparer
        connection.execute(
            "SELECT sequence_set('%s', %d, 0)" %
            (preparer.format_sequence(self), value))


RESERVED_WORDS = {
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
    'compress', 'constraint', 'constructor', 'contains', 'continue', 'convert',
    'corresponding', 'create', 'cross', 'cube', 'current', 'current_date',
    'current_time', 'current_timestamp', 'cursor', 'data', 'date', 'datetime',
    'decimal', 'declare', 'default', 'delete', 'desc', 'deterministic',
    'disable', 'disconnect', 'distinct', 'do', 'double', 'drop', 'dtd',
    'dynamic', 'else', 'elseif', 'enable', 'encoding', 'end', 'escape',
    'except', 'exclusive', 'execute', 'exists', 'external', 'extract', 'exit',
    'fetch', 'final', 'float', 'for', 'foreach', 'foreign', 'fortran',
    'for_vectored', 'for_rows', 'found', 'from', 'full', 'function', 'general',
    'generated', 'go', 'goto', 'grant', 'group', 'grouping', 'handler',
    'having', 'hash', 'identity', 'identified', 'if', 'in', 'incremental',
    'increment', 'index', 'index_no_fill', 'index_only', 'indicator', 'inner',
    'inout', 'input', 'insert', 'instance', 'instead', 'int', 'integer',
    'intersect', 'internal', 'interval', 'into', 'is', 'isolation', 'iri_id',
    'iri_id_8', 'java', 'join', 'key', 'keyset', 'language', 'left', 'level',
    'library', 'like', 'locator', 'log', 'long', 'loop', 'method', 'modify',
    'modifies', 'module', 'mumps', 'name', 'natural', 'nchar', 'new',
    'nonincremental', 'not', 'no', 'novalidate', 'null', 'nullif', 'numeric',
    'nvarchar', 'object_id', 'of', 'off', 'old', 'on', 'open', 'option', 'or',
    'order', 'out', 'outer', 'overriding', 'partition', 'pascal', 'password',
    'percent', 'permission_set', 'persistent', 'pli', 'position', 'precision',
    'prefetch', 'primary', 'privileges', 'procedure', 'public', 'purge',
    'quietcast', 'rdf_box', 'read', 'reads', 'real', 'ref', 'references',
    'referencing', 'remote', 'rename', 'repeatable', 'replacing',
    'replication', 'resignal', 'restrict', 'result', 'return', 'returns',
    'revoke', 'rexecute', 'right', 'rollback', 'rollup', 'role', 'safe',
    'same_as', 'uncommitted', 'unrestricted', 'schema', 'select', 'self',
    'serializable', 'set', 'sets', 'shutdown', 'smallint', 'snapshot', 'soft',
    'some', 'source', 'sparql', 'specific', 'sql', 'sqlcode', 'sqlexception',
    'sqlstate', 'sqlwarning', 'static', 'start', 'style', 'sync', 'system',
    't_cycles_only', 't_direction', 't_distinct', 't_end_flag', 't_exists',
    't_final_as', 't_in', 't_max', 't_min', 't_no_cycles', 't_no_order',
    't_out', 't_shortest_only', 'table', 'temporary', 'text', 'then', 'ties',
    'time', 'timestamp', 'to', 'top', 'type', 'transaction', 'transitive',
    'trigger', 'under', 'union', 'unique', 'update', 'use', 'user', 'using',
    'validate', 'value', 'values', 'varbinary', 'varchar', 'variable',
    'vector', 'vectored', 'view', 'when', 'whenever', 'where', 'while', 'with',
    'without', 'work', 'xml', 'xpath'}


class VirtuosoIdentifierPreparer(compiler.IdentifierPreparer):
    reserved_words = RESERVED_WORDS

    def quote_schema(self, schema, force=None):
        if '.' in schema:
            cat, schema = schema.split('.', 1)
            return self.quote(cat, force) + '.' + self.quote(schema, force)
        else:
            # Virtuoso needs an extra dot to indicate absent username
            return self.quote(schema, force) + '.'

    def format_sequence(self, sequence, use_schema=True):
        res = super(VirtuosoIdentifierPreparer, self).format_sequence(
            sequence, use_schema=use_schema)
        # unquote
        return res.strip('"')


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

    def get_select_precolumns(self, select, **kw):
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

    def visit_sequence(self, seq):
        return "sequence_next('%s')" % self.preparer.format_sequence(seq)

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

    def visit_in_op_binary(self, binary, operator, **kw):
        """ This is beyond absurd. Virtuoso gives weird results on other columns
        when doing a single-value IN clause. Avoid those. """
        if (isinstance(binary.right, Grouping)
                and isinstance(binary.right.element, ClauseList)
                and len(binary.right.element.clauses) == 1):
            el = binary.right.element.clauses[0]
            return "%s = %s" % (
                self.process(binary.left, **kw),
                self.process(el, **kw))
        return self._generate_generic_binary(binary, " IN ", **kw)

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

    def visit_sparqlclause(self, sparqlclause, **kw):
        def do_bindparam(m):
            name = m.group(1)
            if name in sparqlclause._bindparams:
                self.process(sparqlclause._bindparams[name], **kw)
            return '??'

        # un-escape any \:params
        text = BIND_PARAMS_ESC.sub(
            lambda m: m.group(1),
            BIND_PARAMS.sub(
                do_bindparam,
                self.post_process_text(sparqlclause.text))
        )
        if sparqlclause.quad_storage:
            text = 'define input:storage %s %s' % (
                sparqlclause.quad_storage, text)
        return 'SPARQL ' + text


class SparqlClause(TextClause):
    __visit_name__ = 'sparqlclause'

    def __init__(self, text, bind=None, quad_storage=None):
        super(SparqlClause, self).__init__(text, bind)
        self.quad_storage = quad_storage

    def columns(self, *cols, **types):
        textasfrom = super(SparqlClause, self).columns(*cols, **types)
        return textasfrom.alias()


class LONGVARCHAR(Text):
    __visit_name__ = 'LONG VARCHAR'


class LONGNVARCHAR(UnicodeText):
    __visit_name__ = 'LONG NVARCHAR'


class DOUBLEPRECISION(Float):
    __visit_name__ = 'DOUBLE PRECISION'


class LONGVARBINARY(Binary):
    __visit_name__ = 'LONG VARBINARY'


class CoerceUnicode(TypeDecorator):
    impl = Unicode
    # Maybe TypeDecorator should delegate? Another story
    python_type = past.builtins.unicode

    def process_bind_param(self, value, dialect):
        if util.py2k and isinstance(value, util.binary_type):
            value = value.decode(dialect.encoding)
        return value

    def bind_expression(self, bindvalue):
        return _cast_nvarchar(bindvalue)


class _cast_nvarchar(ColumnElement):
    def __init__(self, bindvalue):
        self.bindvalue = bindvalue


@compiles(_cast_nvarchar)
def _compile(element, compiler, **kw):
    return compiler.process(cast(element.bindvalue, Unicode), **kw)


class dt_set_tz(GenericFunction):
    "Convert IRI IDs to int values"
    type = DATETIME
    name = "dt_set_tz"

    def __init__(self, adatetime, offset, **kw):
        if not (isinstance(adatetime, (datetime, DATETIME))
                or isinstance(adatetime.__dict__.get('type'), DATETIME)):
            warnings.warn(
                "dt_set_tz() accepts a DATETIME object as first input.")
        if not (isinstance(offset, (int, INTEGER))
                or isinstance(offset.__dict__.get('type'), INTEGER)):
            warnings.warn(
                "dt_set_tz() accepts a INTEGER object as second input.")
        super(dt_set_tz, self).__init__(adatetime, offset, **kw)


class Timestamp(TypeDecorator):
    impl = TIMESTAMP
    # Maybe TypeDecorator should delegate? Another story
    python_type = datetime

    def column_expression(self, colexpr):
        return dt_set_tz(cast(colexpr, DATETIME), 0)


TEXT_TYPES = (CHAR, VARCHAR, NCHAR, NVARCHAR, String, UnicodeText,
              Unicode, Text, LONGVARCHAR, LONGNVARCHAR, CoerceUnicode)


class IRI_ID_Literal(str):
    "An internal virtuoso IRI ID, of the form #innnnn"
    def __str__(self):
        return 'IRI_ID_Literal("%s")' % (self, )

    def __repr__(self):
        return str(self)


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
        if not isinstance(num, int):
            warnings.warn("iri_id_num() accepts an Integer as input.")
        super(iri_id_from_num, self).__init__(num, **kw)


class id_to_iri(GenericFunction):
    "Get the IRI from a given IRI ID"
    type = String
    name = "id_to_iri"

    def __init__(self, iri_id, **kw):
        # TODO: Handle deferred.
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
        if isinstance(iri, past.builtins.unicode):
            iri = iri_to_uri(iri)
        if not isinstance(iri, str):
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



class AddForeignKey(_CreateDropBase):
    """Represent an ALTER TABLE ADD CONSTRAINT statement."""

    __visit_name__ = "add_foreign_key"

    def __init__(self, element, *args, **kw):
        super(AddForeignKey, self).__init__(element, *args, **kw)
        element._create_rule = util.portable_instancemethod(
            self._create_rule_disable)


class DropForeignKey(_CreateDropBase):
    """Represent an ALTER TABLE DROP CONSTRAINT statement."""

    __visit_name__ = "drop_foreign_key"

    def __init__(self, element, cascade=False, **kw):
        self.cascade = cascade
        super(DropForeignKey, self).__init__(element, **kw)
        element._create_rule = util.portable_instancemethod(
            self._create_rule_disable)


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

    def visit_drop_foreign_key(self, drop):
        # Make sure the constraint has no name, ondelete, deferrable, onupdate
        constraint = drop.element.constraint
        names = ("name", "ondelete", "deferrable", "onupdate")
        temp = {name: getattr(constraint, name, None) for name in names}
        for name in names:
            setattr(constraint, name, None)
        result = "ALTER TABLE %s DROP %s" % (
            self.preparer.format_table(drop.element.parent.table),
            self.visit_foreign_key_constraint(constraint),
        )
        for name in names:
            setattr(constraint, name, temp[name])
        return result


    def visit_add_foreign_key(self, create):
        return "ALTER TABLE %s ADD %s" % (
            self.preparer.format_table(create.element.parent.table),
            self.visit_foreign_key_constraint(create.element.constraint),
        )

    def visit_create_text_index(self, create, include_schema=False,
                           include_table_schema=True):
        text_index = create.element
        column = text_index.column
        params = dict(table=column.table.name, column=column.name)
        for x in ('xml','clusters','key','with_insert','transform','language','encoding'):
            params[x] =''
        if isinstance(column.type, (XML, LONGXML)):
            params['xml'] = 'XML'
        else:
            assert isinstance(column.type, TEXT_TYPES)
        if text_index.clusters:
            params['clusters'] = 'CLUSTERED WITH (' + ','.join((
                self.preparer.quote(c.name) for c in text_index.clusters)) + ')'
        if text_index.key:
            params['key'] = 'WITH KEY ' + self.preparer.quote(text_index.key.name)
        if not text_index.do_insert:
            params['with_insert'] = 'NO INSERT'
        if text_index.transform:
            params['transform'] = 'USING ' + self.preparer.quote(text_index.transform)
        if text_index.language:
            params['language'] = "LANGUAGE '" + text_index.language + "'"
        if text_index.encoding:
            params['encoding'] = 'ENCODING ' + text_index.encoding
        return ('CREATE TEXT {xml} INDEX ON "{table}" ( "{column}" ) {key} '
                '{with_insert} {clusters} {transform} {language} {encoding}'
                ).format(**params)

    def visit_drop_text_index(self, drop):
        text_index = drop.element
        name = "{table}_{column}_WORDS".format(
            table=text_index.column.table.name,
            column=text_index.column.name)
        return '\nDROP TABLE %s.%s' % (
            self.preparer.quote_schema(text_index.table.schema),
            self.preparer.quote(name))

# TODO: Alter is weird. Use MODIFY with full new thing. Eg:
# ALTER TABLE assembl..imported_post MODIFY body_mime_type NVARCHAR NOT NULL


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
    supports_unicode_statements = True
    supports_unicode_binds = True
    returns_unicode_strings = True
    supports_native_boolean = False
    ddl_compiler = VirtuosoDDLCompiler
    supports_right_nested_joins = False
    supports_multivalues_insert = False

    supports_sequences = True
    postfetch_lastrowid = True

    def _get_default_schema_name(self, connection):
        res = connection.execute(
            'select U_DEF_QUAL, get_user() from DB.DBA.SYS_USERS where U_NAME=get_user()')
        catalog, schema = res.fetchone()
        if catalog:
            return '.'.join((catalog, schema))

    def has_table(self, connection, tablename, schema=None):
        if schema is None:
            schema = self.default_schema_name
        if '.' not in schema:
            schema += '.'
        catalog, schema = schema.split('.', 1)
        result = connection.execute(
            text("SELECT TABLE_NAME FROM DB..TABLES WHERE "
                 "TABLE_CATALOG=:schemaname AND "
                 "TABLE_NAME=:tablename",
                 bindparams=[
                     bindparam("schemaname", catalog),
                     bindparam("tablename", tablename)
                 ])
        )
        return result.scalar() is not None

    def has_sequence(self, connection, sequence_name, schema=None):
        # sequences are auto-created in virtuoso
        return True

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
                    text("SELECT TABLE_NAME FROM DB..TABLES WHERE "
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
