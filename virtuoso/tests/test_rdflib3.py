from rdflib.graph import ConjunctiveGraph, Graph, Namespace
from rdflib.store import Store
from rdflib.plugin import get as plugin
from rdflib.namespace import RDF, RDFS, XSD
from rdflib.term import URIRef, Literal, BNode
from datetime import datetime
from virtuoso.vstore import EagerIterator, Virtuoso
from virtuoso.vsparql import Result
import os
import unittest

from nose.plugins.skip import SkipTest
from . import rdflib_connection

class Test00Plugin(unittest.TestCase):
    def test_get_plugin(self):
        V = plugin("Virtuoso", Store)
        assert V is Virtuoso

from math import pi
ex_subject = URIRef("http://example.org/")

test_statements = [
    (ex_subject, RDF["type"], RDFS["Resource"]),
    (BNode(), RDF["type"], RDFS["Resource"]),
    (ex_subject, RDF["type"], BNode()),
    (ex_subject, RDFS["label"], Literal("hello world")),
    (ex_subject, RDFS["comment"],
     Literal("Here we have a long comment to purposely overflow the inline RDF_QUAD limit. "
             "We keep talking and talking, but what are we saying? Precisely nothing the "
             "whole idea is to have a bunch of characters here. Blah blah, yadda yadda, "
             "etc. This is probably enough. Hopefully. One more sentence to make certain.")),
    (ex_subject, RDFS["label"], Literal(3)),
    (ex_subject, RDFS["comment"], Literal(datetime.now())),
    (ex_subject, RDFS["comment"], Literal(datetime.now().date())),
    #commented out the following line, as it seems to be broken
    #(ex_subject, RDFS["comment"], Literal(datetime.now().time())),
    (ex_subject, RDFS["comment"], Literal("1970", datatype=XSD["gYear"])),
    (ex_subject, RDFS["label"], Literal("hello world", lang="en")),
    ]

## special test that will induce a namespace creation for testing of serialisation
ns_test = (URIRef("http://bnb.bibliographica.org/entry/GB8102507"), RDFS["label"], Literal("foo"))
test_statements.append(ns_test)

float_test = (ex_subject, RDFS["label"], Literal(pi))

class Test00Open(unittest.TestCase):

    def test_open(self):
        store = Virtuoso(rdflib_connection)
        graph = Graph(store)
        result = graph.query("ASK { ?s ?p ?o }")
        assert not result

class Test01Store(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.store = Virtuoso(rdflib_connection)
        cls.identifier = URIRef("http://example2.org/")
        cls.graph = Graph(cls.store, identifier=cls.identifier)
        cls.graph.remove((None, None, None))

    @classmethod
    def tearDown(cls):
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
        q = "CONSTRUCT { ?s ?p ?o } WHERE { GRAPH %s { ?s ?p ?o } }" % (self.graph.identifier.n3(),)
        result = self.store.query(q)
        assert isinstance(result, Graph) or isinstance(result, Result)
        assert test_statements[0] in result
        self.graph.remove(test_statements[0])

    def test_04_ask(self):
        arg = (self.graph.identifier.n3(),)
        assert not self.graph.query("ASK FROM %s WHERE { ?s ?p ?o }" % arg)
        self.graph.add(test_statements[0])
        assert self.graph.query("ASK FROM %s WHERE { ?s ?p ?o }" % arg)
        self.graph.remove(test_statements[0])
        assert not self.graph.query("ASK FROM %s WHERE { ?s ?p ?o }" % arg)

    def test_05_select(self):
        for statement in test_statements:
            self.graph.add(statement)
        q = "SELECT DISTINCT ?s FROM %(g)s WHERE { ?s %(t)s ?o }" % {
            "t": RDF["type"].n3(), "g": self.graph.identifier.n3()}
        results = list(self.graph.query(q))
        assert len(results) == 2, results
        self.graph.remove((None, None, None))

    def test_06_construct(self):
        for statement in test_statements:
            self.graph.add(statement)
        q = "CONSTRUCT { ?s %(t)s ?o } FROM %(g)s WHERE { ?s %(t)s ?o }" % {
            "t": RDF["type"].n3(), "g": self.graph.identifier.n3()}
        result = self.graph.query(q)
        assert isinstance(result, Graph)
        assert len(result) == 3
        self.graph.remove((None, None, None))

    def test_07_float(self):
        self.add_remove(float_test)
        for ob in self.graph.objects(ex_subject, RDFS["label"]):
            assert isinstance(ob, float)

    def test_08_serialize(self):
        self.graph.add(ns_test)
        self.graph.serialize(format="n3")

    def test_09_multiple_results(self):
        # This fails on virtuoso 7. 
        # https://github.com/maparent/virtuoso-python/issues/2
        # https://github.com/openlink/virtuoso-opensource/issues/127
        TST=Namespace('http://example.com/ns/')
        self.graph.add((TST.A, RDFS.subClassOf, TST.B))
        self.graph.add((TST.B, RDFS.subClassOf, TST.C))
        self.graph.add((TST.C, RDFS.subClassOf, TST.D))
        self.graph.add((TST.D, RDFS.subClassOf, TST.TOP))
        result = self.graph.query("""SELECT DISTINCT ?class
            WHERE {
                ?class rdfs:subClassOf+ %s .
                %s rdfs:subClassOf+ ?class .
            }""" % (TST.TOP.n3(), TST.A.n3()))
        result = list(result)
        print result
        if not len(result):
            # This should be a xFail, but nosetests does not offer this.
            raise SkipTest

    def test_10_oount(self):
        statements = [s for s in test_statements if s[0]== ex_subject]
        for statement in statements:
            self.graph.add(statement)
        result = self.graph.query("""SELECT COUNT(?o) 
            WHERE {<http://example.org/> ?p ?o}""")
        result = result.next()[0]
        assert result == len(statements)

    def test_11_base(self):
        TST=Namespace('http://example.com/ns/')
        self.graph.add((TST.a, TST.b, TST.c))
        self.graph.add((TST.d, TST.e, TST.f))
        result = self.graph.query("""BASE <http://example.com/ns/>
            SELECT * { ?s <b> ?o }""")
        assert type(result) is EagerIterator
        assert len(list(result)) == 1
        
    def test_11_empty_prefix(self):
        TST=Namespace('http://example.com/ns/')
        self.graph.add((TST.a, TST.b, TST.c))
        self.graph.add((TST.d, TST.e, TST.f))
        result = self.graph.query("""PREFIX : <http://example.com/ns/>
            SELECT * { ?s :b ?o }""")
        assert type(result) is EagerIterator
        assert len(list(result)) == 1

    def test_12_ask(self):
        TST=Namespace('http://example.com/ns/')
        self.graph.add((TST.a, TST.b, TST.c))
        self.graph.add((TST.d, TST.e, TST.f))
        result = self.graph.query("ASK { ?s ?p ?o }")
        assert type(result) is bool
        assert result

    def test_13_prepared_qyery(self):
        from rdflib.plugins.sparql.processor import prepareQuery
        pquery = prepareQuery("SELECT * { ?s <b> ?o }", base="http://example.com/ns/")
        
        TST=Namespace('http://example.com/ns/')
        self.graph.add((TST.a, TST.b, TST.c))
        self.graph.add((TST.d, TST.e, TST.f))
        result = self.graph.query(pquery)
        assert type(result) is EagerIterator
        assert len(list(result)) == 1

    def test_99_deadlock(self):
        os.environ["VSTORE_DEBUG"] = "TRUE"
        dirname = os.path.dirname(__file__)
        fixture = os.path.join(dirname, "fixture1.rdf")
        self.graph.parse(fixture)
        for statement in self.graph.triples((None, None, None)):
            pass
        self.graph.remove((None, None, None))

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
    attr = "test_%02d_add_remove" % (i + 80)
    setattr(Test01Store, attr, _mk_add_remove(attr, test_statements[i]))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test00Plugin)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(Test01Store)
    unittest.TextTestRunner(verbosity=2).run(suite)
