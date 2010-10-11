from rdflib.graph import Graph
from rdflib.store import Store, VALID_STORE, NO_STORE
from rdflib.term import URIRef, BNode, Literal, Variable
from urlparse import urlparse
from traceback import format_exc
import pyodbc

__all__ = ['Virtuoso']

log = __import__("logging").getLogger(__name__)

class Virtuoso(Store):
    context_aware = True
    transaction_aware = True
    def open(self, dsn):
        try:
            self._connection = pyodbc.connect(dsn)
            self._cursor = self._connection.cursor()
            log.info("Virtuoso Store Connected: %s" % dsn)
            return VALID_STORE
        except:
            log.error("Virtuoso Connection Failed:\n%s" % format_exc())
            return NO_STORE

    def close(self, commit_pending_transaction=False):
        if commit_pending_transaction:
            self.commit()
        else:
            self.rollback()
        self._connection.close()

    def query(self, q):
        return self._cursor.execute(str(q))
    def sparql_query(self, q):
        return self.query(u"SPARQL " + q)
    
    def commit(self):
        self.query("COMMIT WORK")
    def rollback(self):
        self.query("ROLLBACK WORK")

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
        q = (u'SPARQL define output:valmode "LONG" '
             u"SELECT DISTINCT %(Sv)s %(Pv)s %(Ov)s %(Gv)s "
             u"WHERE { GRAPH %(G)s { %(S)s %(P)s %(O)s } }")
        q = q % query_bindings
        log.debug(q)

        resolver = self._connection.cursor()
        try:
            for statement in self.query(q):
                s_id, p_id, o_id, g_id = statement
                s = _resolve_iri(resolver, s_id)
                p = _resolve_iri(resolver, p_id)
                o = _resolve_obj(resolver, o_id)
                if g_id is not None:
                    g = _resolve_iri(resolver, g_id)
                else:
                    g = None
                yield (s, p, o), g
        finally:
            resolver.close()

    def add(self, statement, context=None, quoted=False):
        assert not quoted, "No quoted graph support in Virtuoso store yet, sorry"
        query_bindings = _query_bindings(statement, context)
        q = u'SPARQL INSERT '
        if context is not None:
            q += u'INTO GRAPH %(G)s ' % query_bindings
        q += u'{ %(S)s %(P)s %(O)s }' % query_bindings
        log.debug(q)
        self.query(q)

    def remove(self, statement, context=None, quoted=False):
        assert not quoted, "No quoted graph support in Virtuoso store yet, sorry"
        query_bindings = _query_bindings(statement, context)
        q = u'SPARQL DELETE '
        if context is not None:
            q += u'FROM GRAPH %(G)s ' % query_bindings
        q += u'{ %(S)s %(P)s %(O)s }' % query_bindings
        log.debug(q)
        self.query(q)

def _bnode_to_nodeid(bnode):
    from string import ascii_letters
    iri = bnode
    for c in bnode[1:]:
        if c in ascii_letters:
            # from rdflib
            iri = "b" + "".join(str(ord(x)-38) for x in bnode)
    return URIRef("nodeID://%s" % iri)

def _nodeid_to_bnode(iri):
    from string import digits
    iri = iri[9:] # strip off "nodeID://"
    bnode = iri
    if len(iri) == 19:
        # assume we made it...
        ones, tens = iri[1::2], iri[2::2]
        chars = [x+y for x,y in zip(ones, tens)]
        bnode = "".join(str(chr(int(x)+38)) for x in chars)
    return BNode(bnode)

def _resolve_iri(resolver, iri):
    q = (u'SELECT __ro2sq(%s)' % iri)
    resolver.execute(str(q))
    iri, = resolver.fetchone()
    if iri.startswith("nodeID://"):
        return _nodeid_to_bnode(iri)
    return URIRef(iri)

def _resolve_obj(resolver, oid):
    if not isinstance(oid, basestring):
        return Literal(oid)
    if oid.startswith("#i"):
        q = (u'SELECT __ro2sq(%(O)s), '
             u'__ro2sq(DB.DBA.RDF_DATATYPE_OF_OBJ(%(O)s)), '
             u'DB.DBA.RDF_LANGUAGE_OF_OBJ(%(O)s)' % { "O": oid })
        resolver.execute(str(q))
        val, datatype, lang = resolver.fetchone()
        if datatype is None and lang is None:
            parsed = urlparse(val)
            if parsed.scheme and (parsed.netloc or parsed.path):
                if val.startswith("nodeID://"):
                    return _nodeid_to_bnode(val)
                return URIRef(val)
            return Literal(val, lang=lang, datatype=datatype)
    return Literal(oid)

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
