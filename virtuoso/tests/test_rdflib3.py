from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.namespace import RDF, RDFS, XSD
from rdflib.term import URIRef, Literal, BNode
from datetime import datetime
from virtuoso.vstore import Virtuoso

from nose.plugins.skip import SkipTest

class Test00Plugin(object):
    def test_get_plugin(self):
        V = plugin("Virtuoso", Store)
        assert V is Virtuoso

from math import pi
test_statements = [
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
    (URIRef("http://example.org/"), RDFS["comment"], Literal(datetime.now().time())),
    (URIRef("http://example.org/"), RDFS["comment"], Literal("1970", datatype=XSD["gYear"])),
    (URIRef("http://example.org/"), RDFS["label"], Literal("hello world", lang="en")), # Fails because comes back w/o language
    ]

## special test that will induce a namespace creation for testing of serialisation
ns_test = (URIRef("http://bnb.bibliographica.org/entry/GB8102507"), RDFS["label"], Literal("foo"))
test_statements.append(ns_test)

float_test = (URIRef("http://example.org/"), RDFS["label"], Literal(pi))

class Test01Store(object):
    @classmethod
    def setup_class(cls):
        cls.store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y")
        cls.graph = Graph(cls.store, identifier=URIRef("http://example.org/"))
        cls.graph.remove((None, None, None))
        
    @classmethod
    def teardown_class(cls):
        cls.graph.remove((None, None, None))
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
            break
           
    def test_03_construct(self):
        self.graph.add(test_statements[0])
        q = "CONSTRUCT { ?s ?p ?o } WHERE { GRAPH %s { ?s ?p ?o } }" % self.graph.identifier.n3()
        result = self.store.query(q)
        assert isinstance(result, Graph)
        assert test_statements[0] in result
        self.graph.remove(test_statements[0])

    def test_04_ask(self):
        assert not self.graph.query("ASK WHERE { ?s ?p ?o }")
        self.graph.add(test_statements[0])
        assert self.graph.query("ASK WHERE { ?s ?p ?o }")
        self.graph.remove(test_statements[0])
        assert not self.graph.query("ASK WHERE { ?s ?p ?o }")

    def test_05_select(self):
        self.graph.add(test_statements[0])
        results = list(self.graph.query("SELECT DISTINCT ?s WHERE { ?s ?p ?o }"))
        assert results == [[test_statements[0][0]]], results
        self.graph.remove(test_statements[0])
        
    def test_06_construct(self):
        self.graph.add(test_statements[0])
        result = self.graph.query("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        assert result.construct is True
        assert isinstance(result.result, Graph)
        assert len(result.result) == 1
        self.graph.remove(test_statements[0])

    def test_07_float(self):
        raise SkipTest()
        self.add_remove(float_test)
        print 
        print repr(float_test[2])
        for x in self.graph.triples((None, None, None)):
            print repr(x[2])

    def test_08_serialize(self):
        self.graph.add(ns_test)
        self.graph.serialize(format="n3")

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

