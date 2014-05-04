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

from . import sqla_connection

engine = create_engine(sqla_connection)
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
    __mapper_args__ = {
        'polymorphic_identity': 'B',
        'polymorphic_on': type,
        'with_polymorphic': '*'
    }


inspect(B).local_table.info = {
    "rdf_subject_pattern": tb_iri.apply('id'),
    "rdf_patterns": [QuadMapPattern(None, RDF.type, TST.tB)]
}

class C(B):
    __tablename__ = "test_c"
    id = Column(Integer, ForeignKey(
            B.id, ondelete='CASCADE', onupdate='CASCADE'
        ), primary_key=True)

    __mapper_args__ = {
        'polymorphic_identity': 'C',
    }


def clean():
    for table in ("test_table", "test_c", "test_b", "test_a"):
        conn = engine.connect()
        result = conn.execute(
            text("SELECT TABLE_CATALOG FROM TABLES WHERE "
                 "lower(TABLE_NAME) = '%s'" % table)
        )
        for s in result.fetchall():
            conn.execute(text("DROP TABLE %s..%s" % (s[0], table)))
            session.commit()


class TestMapping(object):
    qsname = TST.qs
    graphname = TST.g

    @classmethod
    def setup_class(cls):
        clean()
        metadata.create_all(engine)
        cls.store = Virtuoso(connection=session.bind.connect(),
            quad_storage=cls.qsname)


    @classmethod
    def teardown_class(cls):
        clean()

    def teardown(self):
        qs = QuadStorage(self.qsname, ())
        session.execute('sparql '+qs.drop(nsm))

    def create_qs_graph(self):
        g = GraphQuadMapPattern(self.graphname, None, None)
        qs = QuadStorage(self.qsname, [g])
        cpe = ClassPatternExtractor(qs.alias_manager, self.graphname, self.qsname)
        g.add_patterns(cpe.extract_info(A))
        g.add_patterns(cpe.extract_info(B))
        g.add_patterns(cpe.extract_info(C))
        return qs, g

    def declare_qs_graph(self, qs):
        defn = qs.definition_statement(nsm, engine=engine)
        print defn
        return list(session.execute('sparql '+defn))

    def test_05_declare_quads_and_link(self):
        qs, g = self.create_qs_graph()
        print self.declare_qs_graph(qs)
        a = A()
        b = B()
        b.a = a
        session.add(b)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert list(graph.triples((None, TST.alink, None)))

    def test_06_conditional_prop(self):
        raise SkipTest()
        qs, g = self.create_qs_graph()
        g.add_patterns([
            QuadMapPattern(
                tb_iri.apply(B.id),
                TST.safe_name,
                B.name,
                conditions=[B.name != None]
                )
            ])
        print self.declare_qs_graph(qs)
        b = B(name='name')
        b2 = B()
        session.add(b)
        session.add(b2)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.safe_name, None))))
        assert 2 == len(list(graph.triples((None, TST.name, None))))

    def test_06_conditional_link(self):
        raise SkipTest()
        qs, g = self.create_qs_graph()
        g.add_patterns([
            QuadMapPattern(
                tb_iri.apply(B.id),
                TST.safe_alink,
                ta_iri.apply(B.a_id),
                conditions=[B.a_id != None]
                )
            ])
        print self.declare_qs_graph(qs)
        a = A()
        b = B(a=a)
        b2 = B()
        session.add(b)
        session.add(b2)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.safe_alink, None))))
        assert 2 == len(list(graph.triples((None, TST.alink, None))))
