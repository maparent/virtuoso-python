"""
"""
__dist__ = __import__("pkg_resources").get_distribution("rdflib")

from traceback import format_exc
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
try:
    from rdflib.graph import Graph
    from rdflib.term import URIRef, BNode, Literal, Variable
    from rdflib.namespace import XSD
except ImportError:
    from rdflib.Graph import Graph
    from rdflib import URIRef, BNode, Literal, Variable
from rdflib.store import Store, VALID_STORE, NO_STORE

if __dist__.version.startswith('3'):
    import vsparql

import pyodbc

__all__ = ['Virtuoso', 'resolve']

from virtuoso.common import READ_COMMITTED
log = __import__("logging").getLogger(__name__)

## hack to change BNode's random identifier generator to be
## compatible with Virtuoso's needs
from time import time
from random import choice, seed
from string import ascii_letters, digits
seed(time())

__bnode_old_new__ = BNode.__new__

@staticmethod
def __bnode_new__(cls, value=None, *av, **kw):
    if value is None:
        value = choice(ascii_letters) + \
            "".join(choice(ascii_letters+digits) for x in range(7))
    return __bnode_old_new__(cls, value, *av, **kw)
BNode.__new__ = __bnode_new__
## end hack

import re
_ask_re = re.compile('^([ \t\r\n]*DEFINE[ \t]+.*)*([ \t\r\n]*PREFIX[ \t]+[^ \t]*: <[^>]*>)*[ \t\r\n]*(ASK)[ \t\r\n]+WHERE', re.IGNORECASE)
_construct_re = re.compile('^([ \t\r\n]*DEFINE[ \t]+.*)*([ \t\r\n]*PREFIX[ \t]+[^ \t]*: <[^>]*>)*[ \t\r\n]*(CONSTRUCT|DESCRIBE)', re.IGNORECASE)

class Virtuoso(Store):
    """
    RDFLib Storage backed by Virtuoso

    .. automethod:: virtuoso.vstore.Virtuoso.cursor
    .. automethod:: virtuoso.vstore.Virtuoso.query
    .. automethod:: virtuoso.vstore.Virtuoso.sparql_query
    .. automethod:: virtuoso.vstore.Virtuoso.commit
    .. automethod:: virtuoso.vstore.Virtuoso.rollback
    """
    context_aware = True
    transaction_aware = True
    def __init__(self, *av, **kw):
        super(Virtuoso, self).__init__(*av, **kw)

    def open(self, dsn):
        self.__dsn = dsn
        try:
            self._connection = pyodbc.connect(dsn)
            log.info("Virtuoso Store Connected: %s" % dsn)
            self.__init_ns_decls__()
            self._cursor = None
            return VALID_STORE
        except:
            log.error("Virtuoso Connection Failed:\n%s" % format_exc())
            return NO_STORE

    def __init_ns_decls__(self):
        self.__prefix = {}
        self.__namespace = {}
        cursor = self.cursor()
        q = u"DB.DBA.XML_SELECT_ALL_NS_DECLS()"
        for prefix, namespace in cursor.execute(q):
            namespace = URIRef(namespace)
            self.__prefix[namespace] = prefix
            self.__namespace[prefix] = namespace
        cursor.close()

    def cursor(self, isolation=READ_COMMITTED):
        """
        Acquire a cursor, setting the isolation level.
        """
        cursor = self._connection.cursor()
        cursor.execute("SET TRANSACTION ISOLATION LEVEL %s" % isolation)
        return cursor

    def close(self, commit_pending_transaction=False):
        if commit_pending_transaction:
            self.commit()
        else:
            self.rollback()
        self._connection.close()

    def clone(self):
        return Virtuoso(self.__dsn)

    def query(self, q, cursor=None, *av, **kw):
        """
        Run a SQL query on the connection optionally with the supplied cursor.
        If the cursor is not specified one will be allocated and kept until
        :meth:`commit` or :meth:`rollback` is called.
        """
        log.debug(q)
        if not cursor:
            if not self._cursor:
                self._cursor = self.cursor(*av, **kw)
            cursor = self._cursor
        try:
            return cursor.execute(q.encode('utf-8')) #str(q))
        except:
            log.error(u"Error Executing: %s" % q)
            raise
        ## handle where e.args == ('S1TAT', 'Message...')

    def sparql_query(self, q, *av, **kw):
        """
        Run a SPARQL query on the connection. Returns a Graph in case of 
        DESCRIBE or CONSTRUCT, a bool in case of Ask and a generator over
        the results otherwise.
        """
        def _construct(results):
            # virtuoso handles construct by returning turtle
            for result, in results:
                g = Graph()
                turtle = result[0]
                g.parse(StringIO(turtle + "\n"), format="n3")
            return g

        def _bool(results):
            # seems like ask -> false returns an empty result set
            # and ask -> true returns an single row
            if list(results):
                return True
            return False

        def _cursor(results):
            resolver = self.cursor()
            for result in results:
                yield [resolve(resolver, x) for x in result]
            resolver.close()

        results = self.query(u'SPARQL define output:valmode "LONG" ' + q, *av, **kw)
        if _construct_re.match(q):
            return _construct(results)
        elif _ask_re.match(q):
            return _bool(results)
        else:
            return _cursor(results)
    
    def commit(self):
        """
        Commit any pending work. Also releases the cached cursor.
        """
        self.query("COMMIT WORK")
        self._cursor.close()
        self._cursor = None

    def rollback(self):
        """
        Roll back any pending work. Also releases the cached cursor.
        """
        self.query("ROLLBACK WORK")
        self._cursor.close()
        self._cursor = None

    def contexts(self, statement=None):
        if statement is None:
            q = (u'SELECT DISTINCT __ro2sq(G) FROM RDF_QUAD')
        else:
            q = (u'SELECT DISTINCT ?g WHERE '            
                 u'{ GRAPH ?g { %(S)s %(P)s %(O)s } }')
            q = _query_bindings(statement)
        for uri, in self.query(q):
            yield Graph(self, identifier=URIRef(uri))
            
    def triples(self, statement, context=None):
        query_bindings = _query_bindings(statement, context)
        for k,v in query_bindings.items():
            if v.startswith('?') or v.startswith('$'):
                query_bindings[k+"v"]=v
            else:
                query_bindings[k+"v"]="(%s) AS %s" % (v, Variable(k).n3())
        q = (u'SELECT DISTINCT %(Sv)s %(Pv)s %(Ov)s %(Gv)s '
             u'WHERE { GRAPH %(G)s { %(S)s %(P)s %(O)s } }')
        q = q % query_bindings

        resolver = self._connection.cursor()
        for s,p,o,g in self.sparql_query(q):
            yield (s,p,o), g
        resolver.close()

    def add(self, statement, context=None, quoted=False):
        assert not quoted, "No quoted graph support in Virtuoso store yet, sorry"
        query_bindings = _query_bindings(statement, context)
        q = u'SPARQL INSERT '
        if context is not None:
            q += u'INTO GRAPH %(G)s ' % query_bindings
        q += u'{ %(S)s %(P)s %(O)s }' % query_bindings
        self.query(q)
        super(Virtuoso, self).add(statement, context, quoted)

    def remove(self, statement, context=None):
        if statement == (None, None, None) and context is not None:
            q = u'SPARQL CLEAR GRAPH %s' % context.identifier.n3()
        else:
            query_bindings = _query_bindings(statement, context)
            q = u'SPARQL DELETE '
            if context is not None:
                q += u'FROM GRAPH %(G)s ' % query_bindings
            q += u'{ %(S)s %(P)s %(O)s } WHERE { %(S)s %(P)s %(O)s }' % query_bindings
        self.query(q)
        super(Virtuoso, self).remove(statement, context)

    def __len__(self, context=None):
        if isinstance(context, Graph):
            context = context.identifier
        if isinstance(context, BNode):
            context = _bnode_to_nodeid(context)
        q = u"SELECT COUNT (*) WHERE { "
        if context: q += "GRAPH %s { " % context.n3()
        q += "?s ?p ?o"
        if context: q += " }"
        q += " }"
        for count, in self.sparql_query(q):
            return count
            
    def bind(self, prefix, namespace, flags=1):
        q = u"DB.DBA.XML_SET_NS_DECL ('%s', '%s', %s)" % (prefix, namespace, flags)
        self.query(q)
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
            iri = "b" + "".join(str(ord(x)-38) for x in bnode[:8])
            break
    return URIRef("nodeID://%s" % iri)

def _nodeid_to_bnode(iri):
    from string import digits
    iri = iri[9:] # strip off "nodeID://"
    bnode = iri
    if len(iri) == 17:
        # assume we made it...
        ones, tens = iri[1::2], iri[2::2]
        chars = [x+y for x,y in zip(ones, tens)]
        bnode = "".join(str(chr(int(x)+38)) for x in chars)
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
    (value, dvtype, dttype, flag, lang, dtype) = args
#    if dvtype in (129, 211):
#        print "XXX", dvtype, value, dtype
    if dvtype == pyodbc.VIRTUOSO_DV_IRI_ID:
        q = (u'SELECT __ro2sq(%s)' % value)
        resolver.execute(str(q))
        iri, = resolver.fetchone()
        if iri.startswith("nodeID://"):
            return _nodeid_to_bnode(iri)
        return URIRef(iri)
    if dvtype == pyodbc.VIRTUOSO_DV_RDF:
        if dtype == XSD["gYear"].encode("ascii"):
             value = value[:4]
        elif dtype == XSD["gMonth"].encode("ascii"):
            value = value[:7]
        return Literal(value, lang=lang or None, datatype=dtype or None)
    if dvtype == pyodbc.VIRTUOSO_DV_STRING:
        return Literal(value)
    if dvtype == pyodbc.VIRTUOSO_DV_LONG_INT:
        return Literal(int(value))
    if dvtype == pyodbc.VIRTUOSO_DV_SINGLE_FLOAT:
        return Literal(value, datatype=XSD["float"])
    if dvtype == pyodbc.VIRTUOSO_DV_DATETIME:
        value = value.replace(" ", "T")
        if dttype == pyodbc.VIRTUOSO_DT_TYPE_DATETIME:
            return Literal(value, datatype=XSD["dateTime"])
        if dttype == pyodbc.VIRTUOSO_DT_TYPE_DATE:
            return Literal(value[:10], datatype=XSD["date"])
        if dttype == pyodbc.VIRTUOSO_DT_TYPE_TIME:
            return Literal(value, datatype=XSD["time"])
        log.warn("Unknown SPASQL DV DT type: %d for %s" % (dttype, value))
        return Literal(value)
    if dvtype == pyodbc.VIRTUOSO_DV_DATE:
        return Literal(value, datatype=URIRef("http://www.w3.org/2001/XMLSchema#date"))
    if dvtype == 204: ## XXX where is this const!?
        return None
    log.warn("Unhandled SPASQL DV type: %d for %s" % (dvtype, value))
    return Literal(value)

def _query_bindings((s,p,o), g=None):
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
    return dict(
        zip("SPOG", [x.n3() for x in (s,p,o,g)])
        )
