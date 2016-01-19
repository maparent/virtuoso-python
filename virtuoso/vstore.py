"""
"""
__dist__ = __import__("pkg_resources").get_distribution("rdflib")

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import os

from rdflib.graph import Graph
from rdflib.term import URIRef, BNode, Literal, Variable
from rdflib.namespace import XSD, Namespace, NamespaceManager
from rdflib.query import Result
from rdflib.store import Store, VALID_STORE

import pyodbc

__all__ = ['Virtuoso', 'OperationalError', 'resolve', 'VirtRDF']

VirtRDF = Namespace('http://www.openlinksw.com/schemas/virtrdf#')

from virtuoso.common import READ_COMMITTED
import logging
log = logging.getLogger(__name__)

## hack to change BNode's random identifier generator to be
## compatible with Virtuoso's needs
from time import time
from random import choice, seed
from string import ascii_letters, digits
seed(time())

__bnode_old_new__ = BNode.__new__


def __bnode_new__(cls, value=None, *av, **kw):
    if value is None:
        value = choice(ascii_letters) + \
            "".join(choice(ascii_letters + digits) for x in range(7))
    return __bnode_old_new__(cls, value, *av, **kw)
BNode.__new__ = staticmethod(__bnode_new__)
## end hack

import re
_ws_re = r'(\s*#[^\n]*\n)*\s*'
_start_re = r'^SPARQL\s+' \
            r'(DEFINE[ \t]+\S+[ \t]+("[^"]*"|<[^>]*>|[0-9]+)\s+)*' \
            r'{WS}' \
            r'(BASE\b{WS}<[^>]*>{WS})?' \
            r'(PREFIX\b{WS}\w*:{WS}<[^>]*>{WS})*' \
            .format(WS=_ws_re)
_ask_re = re.compile(_start_re + r'(ASK)\b', re.IGNORECASE + re.MULTILINE)
_construct_re = re.compile(_start_re + r'(CONSTRUCT|DESCRIBE)\b', re.IGNORECASE + re.MULTILINE)
_select_re = re.compile(_start_re + r'SELECT\b', re.IGNORECASE + re.MULTILINE)

_base_re = re.compile(r'(BASE[ \t]+<[^>]*>\s+)?', re.IGNORECASE + re.MULTILINE)

class OperationalError(Exception):
    """
    Raised when transactions are mis-managed
    """

import threading


class Cursor(object):
    def __init__(self, connection, isolation=READ_COMMITTED):
        self.__cursor__ = connection.cursor()
        self.__refcount__ = 0
        self.log = logging.getLogger("Cursor[%x]" % id(self))
        if "VSTORE_DEBUG" in os.environ:
            print u"INIT Cursor(%X) Thread(%X)" % (id(self.__cursor__), threading.currentThread().ident)
        self.execute("SET TRANSACTION ISOLATION LEVEL %s" % isolation)

    def __enter__(self):
        self.__refcount__ += 1
        return self

    def __exit__(self, type, value, traceback):
        self.__refcount__ -= 1
        if self.__refcount__ == 0:
            self.close()

    def __getattr__(self, attr):
        return getattr(self.__cursor__, attr)

    def commit(self):
        if self.__cursor__ is None:
            raise OperationalError("No transaction in progress")
        self.execute("COMMIT WORK")

    def rollback(self):
        if self.__cursor__ is None:
            raise OperationalError("No transaction in progress")
        self.execute("ROLLBACK WORK")

    def execute(self, q, *av, **kw):
        if "VSTORE_DEBUG" in os.environ:
            print u"EXEC Cursor(%X) Thread(%X)" % (id(self.__cursor__), threading.currentThread().ident), q
        return self.__cursor__.execute(q)

    def close(self):
        if "VSTORE_DEBUG" in os.environ:
            print u"CLOSE Cursor(%X) Thread(%X)" % (id(self.__cursor__), threading.currentThread().ident)
        if self.__cursor__ is not None:
            self.__cursor__.execute("ROLLBACK WORK")
            self.__cursor__.close()
            self.__cursor__ = None
        else:
            self.log.warn("already closed. set VSTORE_DEBUG in the environment to enable debugging")

    def isOpen(self):
        return self.__cursor__ is not None

class EagerIterator(object):
    """A wrapper for an iterator that calculates one element ahead.
    Allows to start context handlers within the inner generator."""
    def __init__(self, g):
        self.g = g
        self.done = False
        try:
            # Work around stupid virtuoso bug that can return
            # (None, None, None) if you ask for an empty graph on a store.
            while True:
                self.next_val = g.next()
                if self.next_val[0] is not None:
                    break
        except StopIteration:
            self.done = True
    def __iter__(self):
        return self
    def next(self):
        if self.done:
            raise StopIteration()
        a = self.next_val
        try:
            while True:
                self.next_val = self.g.next()
                if self.next_val[0] is not None:
                    break
        except StopIteration:
            self.done = True
        finally:
            return a

class VirtuosoResult(Result):
    """
    Subclass of Result to work better with EagerIterator.
    """
    _bindings_tuples = None
    _bindings_tuples_complete = False
    _bindings = None

    def __init__(self, inner_result):
        if type(inner_result) is EagerIterator:
            Result.__init__(self, "SELECT")
            self._eagerIterator = inner_result
            self.vars = inner_result.vars
        elif type(inner_result) is bool:
            Result.__init__(self, "ASK")
            self.askAnswer = inner_result
        elif type(inner_result) is Graph:
            Result.__init__(self, "CONSTRUCT")
            self.graph = inner_result
        else:
            raise ValueError("Unrecognized inner_result %r" % inner_result)

    @property
    def bindings(self):
        if self.type != "SELECT":
            return None
        if self._bindings is None:
            self_vars = self.vars
            self._bindings = [ dict(zip(self_vars, tpl))
                               for tpl in self ]
        return self._bindings

    def __iter__(self):
        """
        Iter over all bindings as tuples of rdflib Terms.
        """
        if self.type != "SELECT":
            return Result.__iter__(self)
        elif self._bindings_tuples is not None:
            if not self._bindings_tuples_complete:
                raise ValueError("Can not access bindings while iterating")
            return iter(self._bindings_tuples)
        else:
            self._bindings_tuples = []
            return self._iter_tuples()

    def __len__(self):
        try:
            return Result.__len__(self)
        except ValueError:
            return None # __len__ called during __iter__

    def _iter_tuples(self):
        self_bindings_tuples_append = self._bindings_tuples.append
        for i in self._eagerIterator:
            yield i
            self_bindings_tuples_append(i)
        self._bindings_tuples_complete = True

class Virtuoso(Store):
    """
    RDFLib Storage backed by Virtuoso

    .. automethod:: virtuoso.vstore.Virtuoso.cursor
    .. automethod:: virtuoso.vstore.Virtuoso.query
    .. automethod:: virtuoso.vstore.Virtuoso.sparql_query
    .. automethod:: virtuoso.vstore.Virtuoso.transaction
    .. automethod:: virtuoso.vstore.Virtuoso.commit
    .. automethod:: virtuoso.vstore.Virtuoso.rollback
    """
    context_aware = True
    transaction_aware = True
    formula_aware = True   # Not sure whether this is true; needed to read N3.

    def __init__(self, *av, **kw):
        self.long_iri = kw.pop('long_iri', False)
        self.inference = kw.pop('inference', None)
        self.quad_storage = kw.pop('quad_storage', None)
        self.signal_void = kw.pop('signal_void', None)
        connection = kw.pop('connection', None)
        if connection is not None:
            if not isinstance(connection, pyodbc.Connection):
                from sqlalchemy.engine.base import Connection
                if isinstance(connection, Connection):
                    # extract the pyodbc connection
                    connection = connection._Connection__connection.connection
            assert isinstance(connection, pyodbc.Connection)
            self._connection = connection
            self.__init_ns_decls__()
        super(Virtuoso, self).__init__(*av, **kw)
        self._transaction = None

    def open(self, dsn, **kwargs):
        self.__dsn = dsn
        establish = self.connection # ensures connection is established
        return VALID_STORE

    def __init_ns_decls__(self):
        self.__prefix = {}
        self.__namespace = {}
        q = u"DB.DBA.XML_SELECT_ALL_NS_DECLS()"
        with self.cursor() as cursor:
            for prefix, namespace in cursor.execute(q):
                namespace = URIRef(namespace)
                self.__prefix[namespace] = prefix
                self.__namespace[prefix] = namespace

    @property
    def connection(self):
        if not hasattr(self, "_connection"):
            try:
                self._connection = pyodbc.connect(self.__dsn)
                log.info("Virtuoso Store Connected: %s" % self.__dsn)
                self.__init_ns_decls__()
            except:
                log.error("Virtuoso Connection Failed")
                raise
        return self._connection

    def cursor(self, *av, **kw):
        """
        Acquire a cursor, setting the isolation level.
        """
        return Cursor(self.connection, *av, **kw)

    def close(self, commit_pending_transaction=False):
        if commit_pending_transaction:
            self.commit()
        else:
            self.rollback()
        self.connection.close()

    def clone(self):
        return Virtuoso(self.__dsn)

    def query(self, q, initNs={}, initBindings={}, queryGraph=None, **kwargs):
        """
        Run a SPARQL query on the connection. Returns a Graph in case of
        DESCRIBE or CONSTRUCT, a bool in case of Ask and a generator over
        the results otherwise.
        """
        prepared_base = None
        if hasattr(q, "original_args"):
            q, prepared_ns, prepared_base = q.original_args
            if not initNs:
                initNs = prepared_ns
            else:
                prepared_ns = dict(prepared_ns)
                prepared_ns.update(initNs)
                initNs = prepared_ns

        base = kwargs.pop("base", None) or prepared_base

        if initNs:
            splitpoint = _base_re.match(q).end()
            qleft, qright = q[:splitpoint], q[splitpoint:]
            q = "\n".join([ qleft ]
                          + [ "PREFIX %s: <%s>" % i for i in initNs.items() ]
                          + [ qright ])

        if initBindings:
            qleft, qright = q.rsplit("}", 1)
            q = "\n".join([ qleft, "#BEGIN of VALUES inserted by initBindings" ]
                          + [ "VALUES ?%s { %s }" % (var, val.n3())
                              for (var, val) in initBindings.items() ]
                          + [ "} # END of VALUES inserted by initBindings", qright ]
                          )

        if base is not None:
            splitpoint = _base_re.match(q).end()
            if splitpoint == 0:
                q = u'BASE <%s>\n%s' % (base, q)

        if queryGraph is not None and queryGraph is not '__UNION__':
            if isinstance(queryGraph, BNode):
                queryGraph = _bnode_to_nodeid(queryGraph)
            q = u'DEFINE input:default-graph-uri %s %s' % (queryGraph.n3(), q)
        return VirtuosoResult(self._query(q, **kwargs))

    def _query(self, q, cursor=None, commit=False):
        if self.quad_storage:
            q = u'DEFINE input:storage %s %s' % (self.quad_storage.n3(), q)
        if self.long_iri:
            q = u'DEFINE output:valmode "LONG" ' + q
        if self.inference:
            q = u'DEFINE input:inference %s %s' % (self.inference.n3(), q)
        if self.signal_void:
            q = u'define sql:signal-void-variables 1 ' + q
        q = u'SPARQL ' + q
        if cursor is None:
            if self._transaction is not None:
                cursor = self.transaction()
            else:
                cursor = self.cursor()

        try:
            log.debug("query: \n" + unicode(q))
            if _construct_re.match(q):
                return self._sparql_construct(q, cursor)
            elif _ask_re.match(q):
                return self._sparql_ask(q, cursor)
            elif _select_re.match(q):
                return self._sparql_select(q, cursor)
            else:
                return self._sparql_ul(q, cursor, commit=commit)
        except:
            log.error(u"Exception running: %s" % q.decode("utf-8"))
            raise

    def _sparql_construct(self, q, cursor):
        log.debug("_sparql_construct")
        g = Graph()
        with cursor:
            results = cursor.execute(q.encode("utf-8"))
            with cursor as resolver:
                for result in results:
                    g.add(resolve(resolver, x) for x in result)
        return g

    def _sparql_ask(self, q, cursor):
        log.debug("_sparql_ask")
        with cursor:
            # seems like ask -> false returns an empty result set
            # and ask -> true returns an single row
            results = cursor.execute(q.encode("utf-8"))
            return len(results.fetchall()) != 0
            # result = results.next()
            # result = resolve(None, result[0])
            # return result != 0

    def _sparql_select(self, q, cursor):
        log.debug("_sparql_select")
        with cursor:
            results = cursor.execute(q.encode("utf-8"))
            def f():
                with cursor:
                    for r in results:
                        yield [resolve(cursor, x) for x in r]
            e = EagerIterator(f())
            e.vars = [Variable(col[0]) for col in results.description]
            e.selectionF = e.vars
            return e

    def _sparql_ul(self, q, cursor, commit):
        log.debug("_sparql_ul")
        with cursor:
            cursor.execute(q.encode("utf-8"))
            if commit:
                cursor.commit()

    def transaction(self):
        """
        Return a long(er) life cursor associated with this store for
        bulk operations. The :meth:`commit` or :meth:`rollback`
        methods must be called
        """
        if self._transaction is not None and self._transaction.isOpen():
            #raise OperationalError("Transaction already in progress")
            return self._transaction
        self._transaction = self.cursor()
        return self._transaction

    def commit(self):
        """
        Commit any pending work. Also releases the cached cursor.
        """
        if self._transaction is not None:
            if self._transaction.isOpen():
                self._transaction.commit()
                self._transaction.close()
            self._transaction = None

    def rollback(self):
        """
        Roll back any pending work. Also releases the cached cursor.
        """
        if self._transaction is not None:
            if self._transaction.isOpen():
                self._transaction.rollback()
                self._transaction.close()
            self._transaction = None

    def contexts(self, statement=None):
        if statement is None and self.quad_storage is None:
            q = u'SELECT DISTINCT __ro2sq(G) FROM RDF_QUAD'
        else:
            statement = statement or (None, None, None)
            q = (u'SELECT DISTINCT ?g WHERE '
                 u'{ GRAPH ?g { %(S)s %(P)s %(O)s } }')
            q = q % _query_bindings(statement)
            if self.quad_storage:
                q = 'DEFINE input:storage %s %s' % (self.quad_storage.n3(), q)
            q = 'SPARQL '+q
        with self.cursor() as cursor:
            for uri, in cursor.execute(q):
                yield Graph(self, identifier=URIRef(uri))

    def triples(self, statement, context=None):
        s, p, o = statement
        if s is not None and p is not None and o is not None:
            # really we have an ASK query
            if self._triples_ask(statement, context):
                yield statement, context
        else:
            for x in self._triples_pattern(statement, context):
                yield x

    def _triples_ask(self, statement, context=None):
        query_bindings = _query_bindings(statement, context)
        q = (u'ASK WHERE { GRAPH %(G)s { %(S)s %(P)s %(O)s } }' % query_bindings)
        return self._query(q)

    def __contains__(self, statement, context=None):
        return self._triples_ask(statement, context)

    def _triples_pattern(self, statement, context=None):
        query_bindings_terms = _query_bindings(statement, context, False)
        query_bindings = {}
        query_constants = {}
        for k, v in query_bindings_terms.items():
            vn3 = v.n3()
            query_bindings[k] = vn3
            if type(v) is Variable:
                query_bindings[k + "v"] = vn3
            else:
                query_bindings[k + "v"] = ""
                query_constants[k] = query_bindings_terms[k]
        q = (u'SELECT %(Sv)s %(Pv)s %(Ov)s %(Gv)s '
             u'WHERE { GRAPH %(G)s { %(S)s %(P)s %(O)s } }')
        q = q % query_bindings

        for row in self._query(q):
            result, i = [], 0
            for column in "SPOG":
                if column in query_constants:
                    result.append(query_constants[column])
                else:
                    result.append(row[i])
                    i += 1
            yield tuple(result[:3]), result[3]

    def add(self, statement, context=None, quoted=False):
        assert not quoted, "No quoted graph support in Virtuoso store yet, sorry"
        query_bindings = _query_bindings(statement, context)
        q = u'INSERT '
        if context is not None:
            q += u'INTO GRAPH %(G)s ' % query_bindings
        q += u'{ %(S)s %(P)s %(O)s }' % query_bindings
        self._query(q, commit=self._transaction is None)
        super(Virtuoso, self).add(statement, context, quoted)

    def addN(self, quads):
        parts = [ u'INSERT {' ]
        evens = []
        old_g = None
        super_add = super(Virtuoso, self).add
        for s, p, o, g in quads:
            triple = (s, p, o)
            super_add(triple, g)
            query_bindings = _query_bindings(triple, g)
            gid = query_bindings['G']
            if gid != old_g:
                if old_g is not None:
                    parts.append(u'}')
                old_g = gid
                parts.append(u'GRAPH %s {' % gid)
            parts.append(u' %(S)s %(P)s %(O)s .' % query_bindings)
        if old_g is not None:
            parts.append("}}")
            q = "".join(parts)
            self._query(q, commit=self._transaction is None)



    def remove(self, statement, context=None):
        if statement == (None, None, None):
            if context is not None:
                q = u'CLEAR GRAPH %s' % context.identifier.n3()
            else:
                raise Exception("Clear all graphs???")
        else:
            query_bindings = _query_bindings(statement, context)
            if context is None:
                q = u'DELETE { GRAPH ?g { %(S)s %(P)s %(O)s }} WHERE { GRAPH ?g { %(S)s %(P)s %(O)s }}'
            # elif reduce(and_, [s is not None for s in statement]):
            #   Actually, due to virtuoso 7 bug, if they're URIRefs or bare literals without type and language...
            #     q = u'DELETE DATA FROM GRAPH %(G)s { %(S)s %(P)s %(O)s }'
            else:
                q = u'DELETE FROM GRAPH %(G)s { %(S)s %(P)s %(O)s } FROM %(G)s WHERE { %(S)s %(P)s %(O)s }'
            q = q % query_bindings
        self._query(q, commit=self._transaction is None)
        super(Virtuoso, self).remove(statement, context)

    def __len__(self, context=None):
        q = "{?s ?p ?o}"
        if context is not None:
            gid = context.identifier
            if isinstance(gid, BNode):
                gid = _bnode_to_nodeid(gid)
            q = "{GRAPH <%s>  %s }" % (gid, q)
        q = u"SELECT COUNT (*) WHERE " + q
        for count, in self._query(q):
            return int(count)
        return 0

    def bind(self, prefix, namespace, flags=1):
        q = u"DB.DBA.XML_SET_NS_DECL ('%s', '%s', %s)" % (prefix, namespace, flags)
        with self.cursor() as cursor:
            cursor.execute(q)
            cursor.commit()
        self.__prefix[namespace] = prefix
        self.__namespace[prefix] = namespace

    def namespace(self, prefix):
        return self.__namespace.get(prefix, None)

    def prefix(self, namespace):
        return self.__prefix.get(namespace, None)

    def namespaces(self):
        for prefix, namespace in self.__namespace.iteritems():
            yield prefix, namespace


def _bnode_to_nodeid(bnode):
    from string import ascii_letters
    iri = bnode
    for c in bnode[1:]:
        if c in ascii_letters:
            # from rdflib not virtuoso
            iri = "b" + "".join(str(ord(x) - 38) for x in bnode[:8])
            break
    return URIRef("nodeID://%s" % iri)


def _nodeid_to_bnode(iri):
    #from string import digits
    iri = iri[9:]  # strip off "nodeID://"
    bnode = iri
    if len(iri) == 17:
        # assume we made it...
        ones, tens = iri[1::2], iri[2::2]
        chars = [x + y for x, y in zip(ones, tens)]
        bnode = "".join(str(chr(int(x) + 38)) for x in chars)
    return BNode(bnode)


def resolve(resolver, args):
    """
    Takes the Virtuoso representation of an RDF node and returns
    an appropriate instance of :class:`rdflib.term.Node`.

    :param resolver: an cursor that can be used for database
        queries necessary to resolve the value
    :param args: the tuple returned
        by :mod:`pyodbc` in case of a SPASQL query.
    """
    if not isinstance(args, tuple):
        # Single number; convert to Literal
        return Literal(args)
    (value, dvtype, dttype, flag, lang, dtype) = args
#    if dvtype in (129, 211):
#        print "XXX", dvtype, value, dtype
    if dvtype == pyodbc.VIRTUOSO_DV_IRI_ID:
        q = (u'SELECT __ro2sq(%s)' % value)
        resolver.execute(str(q))
        iri, = resolver.fetchone()
        if iri[:9] == "nodeID://":
            return _nodeid_to_bnode(iri)
        return URIRef(iri)
    if dvtype == pyodbc.VIRTUOSO_DV_RDF:
        if dtype == XSD["gYear"].encode("ascii"):
            value = value[:4]
        elif dtype == XSD["gMonth"].encode("ascii"):
            value = value[:7]
        return Literal(value, lang=lang or None, datatype=dtype or None)
    if dvtype in (pyodbc.VIRTUOSO_DV_STRING, pyodbc.VIRTUOSO_DV_BLOB_WIDE_HANDLE,
                  pyodbc.VIRTUOSO_DV_WIDE):
        # TODO: HORRID temporary heuristics, but I get wrong flag values.
        if flag == 1 and ' ' not in value:
            if value[:9] == "nodeID://":
                return _nodeid_to_bnode(value)
            return URIRef(value)
        else:
            if dtype == XSD["gYear"].encode("ascii"):
                value = value[:4]
            elif dtype == XSD["gMonth"].encode("ascii"):
                value = value[:7]
            if not isinstance(value, unicode):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    value = value.decode('iso-8859-1')
            return Literal(value, lang=lang or None, datatype=dtype or None)
    if dvtype == pyodbc.VIRTUOSO_DV_LONG_INT:
        return Literal(int(value))
    if dvtype == pyodbc.VIRTUOSO_DV_SINGLE_FLOAT or dvtype == pyodbc.VIRTUOSO_DV_DOUBLE_FLOAT:
        return Literal(value, datatype=XSD["float"])
    if dvtype == pyodbc.VIRTUOSO_DV_NUMERIC:
        return Literal(value, datatype=XSD["decimal"])
    if dvtype == pyodbc.VIRTUOSO_DV_DATETIME or dvtype == pyodbc.VIRTUOSO_DV_TIMESTAMP:
        value = value.replace(" ", "T")
        if dttype == pyodbc.VIRTUOSO_DT_TYPE_DATE:
            return Literal(value[:10], datatype=XSD["date"])
        elif dttype == pyodbc.VIRTUOSO_DT_TYPE_TIME:
            return Literal(value, datatype=XSD["time"])
        elif dttype == pyodbc.VIRTUOSO_DT_TYPE_DATETIME:
            return Literal(value, datatype=XSD["dateTime"])
        log.warn("Unknown SPASQL DV DT type: %d for %s" % (dttype, value))
        return Literal(value)
    if dvtype == pyodbc.VIRTUOSO_DV_DATE:
        return Literal(value, datatype=XSD["date"])
    if dvtype == pyodbc.VIRTUOSO_DV_TIME:
        return Literal(value, datatype=XSD["time"])
    if dvtype == pyodbc.VIRTUOSO_DV_DB_NULL:
        return None
    log.warn("Unhandled SPASQL DV type: %d for %s" % (dvtype, value))
    return Literal(value)


def _query_bindings((s, p, o), g=None, to_n3=True):
    if isinstance(g, Graph):
        g = g.identifier
    if s is None: s = Variable("S")
    if p is None: p = Variable("P")
    if o is None: o = Variable("O")
    if g is None: g = Variable("G")
    if isinstance(s, BNode):
        s = _bnode_to_nodeid(s)
    if isinstance(p, BNode):
        p = _bnode_to_nodeid(p)
    if isinstance(o, BNode):
        o = _bnode_to_nodeid(o)
    if isinstance(g, BNode):
        g = _bnode_to_nodeid(g)
    if to_n3:
        return dict(
            zip("SPOG", [x.n3() for x in (s, p, o, g)])
        )
    else:
        return dict(
            zip("SPOG", [x for x in (s, p, o, g)])
        )


class VirtuosoNamespaceManager(NamespaceManager):
    def __init__(self, graph, session):
        super(VirtuosoNamespaceManager, self).__init__(graph)
        self.v_prefixes = {prefix: Namespace(uri) for (prefix, uri)
                          in session.execute('XML_SELECT_ALL_NS_DECLS()')}
        for prefix, namespace in self.v_prefixes.items():
            self.bind(prefix, namespace)

    def bind_virtuoso(self, session, prefix, namespace):
        self.bind(prefix, namespace)
        if self.v_prefixes.get(prefix, None) != Namespace(namespace):
            session.execute("XML_SET_NS_DECL('%s', '%s', 2)" % (
                prefix, namespace))
            self.v_prefixes[prefix] = namespace

    def bind_all_virtuoso(self, session):
        for prefix, ns in list(self.namespaces()):
            self.bind_virtuoso(session, prefix, ns)

def monkeypatch_prepare_query():
    """
    ensures that rdflib.plugins.sparql.processor is uptodate, else monkeypatch it.
    """
    # pylint: disable=invalid-name
    import rdflib.plugins.sparql.processor as sparql_processor
    _TEST_PREPARED_QUERY = sparql_processor.prepareQuery("ASK { ?s ?p ?o }")
    if not hasattr(_TEST_PREPARED_QUERY, "original_args"):
        # monkey-patch 'prepare'
        original_prepareQuery = sparql_processor.prepareQuery
        def monkeypatched_prepareQuery(queryString, initNS=None, base=None):
            """
            A monkey-patched version of the original prepareQuery,
            adding an attribute 'original_args' to the result.
            """
            if initNS is None:
                initNS = {}
            ret = original_prepareQuery(queryString, initNS, base)
            ret.original_args = (queryString, initNS, base)
            return ret
        sparql_processor.prepareQuery = monkeypatched_prepareQuery
        log.info("monkey-patched rdflib.plugins.sparql.processor.prepareQuery")
monkeypatch_prepare_query()
del monkeypatch_prepare_query
