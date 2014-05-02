from nose.plugins.skip import SkipTest

from sqlalchemy.engine import create_engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker, mapper, relation
from sqlalchemy.ext.declarative import as_declarative
from sqlalchemy.sql import text, bindparam
from sqlalchemy.inspection import inspect
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

from rdflib import URIRef, Graph
from rdflib.namespace import Namespace, NamespaceManager

from virtuoso.vmapping import *
from virtuoso.vstore import Virtuoso

engine = create_engine("virtuoso://dba:dba@VOSAS2")
Session = sessionmaker(bind=engine)
session = Session(autocommit=False)
metadata = MetaData(schema="test.DBA")

TST = Namespace('http://example.com/test#')
nsm = NamespaceManager(Graph())
nsm.bind('tst', TST)
nsm.bind('virtrdf', VirtRDF)

ta_iri = PatternIriClass(
    TST.ta_iri, 'http://example.com/test#tA/%d', ('id', Integer, False))
tb_iri = PatternIriClass(
    TST.tb_iri, 'http://example.com/test#tB/%d', ('id', Integer, False))


@as_declarative(bind=engine, metadata=metadata)
class Base(object):
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class A(Base):
    __tablename__ = "test_a"
    id = Column(Integer, primary_key=True)
    name = Column(
        String, info={'rdf': QuadMapPattern(None, TST.name, None)})


inspect(A).local_table.info = {
    "rdf_subject_pattern": ta_iri.apply('id'),
    "rdf_patterns": [QuadMapPattern(None, RDF.type, TST.tA)]
}


class B(Base):
    __tablename__ = "test_b"
    id = Column(Integer, primary_key=True)
    name = Column(String, info={'rdf': QuadMapPattern(None, TST.name, None)})
    type = Column(String(20))
    a_id = Column(Integer, ForeignKey(A.id), info={
        'rdf': QuadMapPattern(None, TST.alink, ta_iri.apply())})
    a = relation(A)


inspect(B).local_table.info = {
    "rdf_subject_pattern": tb_iri.apply('id'),
    "rdf_patterns": [QuadMapPattern(None, RDF.type, TST.tB)]
}


def clean():
    for table in ("test_table", "test_b", "test_a"):
        conn = engine.connect()
        result = conn.execute(
            text("SELECT TABLE_CATALOG FROM TABLES WHERE "
                 "lower(TABLE_NAME) = '%s'" % table)
        )
        for s in result.fetchall():
            conn.execute(text("DROP TABLE %s..%s" % (s[0], table)))
            session.commit()


class TestMapping(object):
    @classmethod
    def setup_class(cls):
        clean()
        metadata.create_all(engine)

    @classmethod
    def teardown_class(self):
        clean()

    def test_05_declare_quads(self):
        alias_manager = ClassAliasManager()
        g = GraphQuadMapPattern(TST.g, None, None)
        qs = QuadStorage(TST.qs, [g])
        cpe = ClassPatternExtractor(alias_manager, TST.g, TST.qs)
        g.add_patterns(cpe.extract_info(A))
        g.add_patterns(cpe.extract_info(B))
        defn = qs.definition_statement(nsm, alias_manager, engine)
        print defn
        r = session.execute('sparql '+defn)
        #for x in r.fetchall(): print x
        a = A()
        b = B()
        b.a = a
        session.add(b)
        session.commit()
        store = Virtuoso(connection=session.bind.connect(),
                         quad_storage=qs.name)

        graph = Graph(store, identifier=TST.g)
        assert list(graph.triples((None, TST.alink, None)))
