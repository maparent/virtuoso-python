from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.namespace import RDF, RDFS
from rdflib.term import URIRef, Literal, BNode
from datetime import datetime
from virtuoso.vstore import Virtuoso

class Test00Plugin(object):
    def test_get_plugin(self):
        V = plugin("Virtuoso", Store)
        assert V is Virtuoso

from math import pi
test_statements = (
    (URIRef("http://example.org/"), RDF["type"], RDFS["Resource"]),
    (BNode(), RDF["type"], RDFS["Resource"]),
    (URIRef("http://example.org/"), RDF["type"], BNode()),
    (URIRef("http://example.org/"), RDFS["label"], Literal("hello world")),
    (URIRef("http://example.org/"), RDFS["comment"],
     Literal("Here we have a long comment to purposely overflow the inline RDF_QUAD limit. "
             "We keep talking and talking, but what are we saying? Precisely nothing the "
             "whole idea is to have a bunch of characters here. Blah blah, yadda yadda, "
             "etc. This is probably enough. Hopefully. One more sentence to make certain.")),
    (URIRef("http://example.org/"), RDFS["label"], Literal(3)), # Fails because comes back untyped
    (URIRef("http://example.org/"), RDFS["comment"], Literal(datetime.now())),
    (URIRef("http://example.org/"), RDFS["comment"], Literal(datetime.now().date())),
#    (URIRef("http://example.org/"), RDFS["label"], Literal(pi)), # Fails because floats cannot be found?
    (URIRef("http://example.org/"), RDFS["label"], Literal("hello world", lang="en")), # Fails because comes back w/o language
    )

class Test01Store(object):
    @classmethod
    def setup_class(cls):
        cls.store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
        cls.graph = Graph(cls.store, identifier=URIRef("http://example.org/"))
            
    @classmethod
    def teardown_class(cls):
        cls.store.sparql_query("CLEAR GRAPH %s" % cls.graph.identifier.n3())
        cls.store.commit()
        cls.store.close()
        
    def test_01_query(self):
        g = ConjunctiveGraph(self.store)
        count = 0
        for statement in g.triples((None, None, None)):
            count += 1
            break
        assert count == 1, "Should have found at least one triple"

    def test_02_contexts(self):
        g = ConjunctiveGraph(self.store)
        for c in g.contexts():
            assert isinstance(c, Graph)
            
    def add_remove(self, statement):
        # add and check presence
        self.graph.add(statement)
        self.store.commit()
        
        assert statement in self.graph, "%s not found" % (statement,)

        # check that we really got back what we asked for
        for x in self.graph.triples(statement):
            assert statement == x, "Round-trip mismatch:\n\t%s\n\t%s" % (statement, x)

        # delete and check absence
        self.graph.remove(statement)
        self.store.commit()
        
        assert statement not in self.graph, "%s found" % (statement,)

# make separate tests for each of the test statements so that we don't
# get flooded with unreadable and irrelevant log messages if one fails
def _mk_add_remove(name, s):
    def _f(self):
        self.add_remove(s)
    _f.func_name = name
    return _f
for i in range(len(test_statements)):
    attr = "test_%02d_add_remove" % (i + 10)
    setattr(Test01Store, attr, _mk_add_remove(attr, test_statements[i]))

