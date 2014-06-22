from nose.plugins.skip import SkipTest

from sqlalchemy.engine import create_engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker, mapper, relation
from sqlalchemy.ext.declarative import as_declarative
from sqlalchemy.sql import text, bindparam
from sqlalchemy.inspection import inspect
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

from rdflib import URIRef, Graph
from rdflib.namespace import Namespace, NamespaceManager, RDF

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
    TST.ta_iri, 'http://example.com/test#tA/%d', None, ('id', Integer, False))
tb_iri = PatternIriClass(
    TST.tb_iri, 'http://example.com/test#tB/%d', None, ('id', Integer, False))


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

    def tearDown(self):
        qs = QuadStorage(self.qsname, nsm=nsm)
        try:
            print qs.drop(session, True)
            for table in ("test_c", "test_b", "test_a"):
                session.execute('delete from test..'+table)
            session.commit()
        except Exception as e:
            print e
            session.rollback()

    def create_qs_graph(self):
        qs = QuadStorage(
            self.qsname, alias_manager=ClassAliasManager(
                Base._decl_class_registry), nsm=nsm)
        g = GraphQuadMapPattern(self.graphname, qs, None, None)
        cpe = ClassPatternExtractor(qs.alias_manager, g)
        g.add_patterns(cpe.extract_info(A))
        g.add_patterns(cpe.extract_info(B))
        g.add_patterns(cpe.extract_info(C))
        return qs, g

    def declare_qs_graph(self, qs):
        # defn = qs.definition_statement(engine=engine)
        # print "old:", defn
        defn = qs.full_declaration_clause()
        print defn.compile(engine)
        result = list(session.execute(defn))
        print result
        return result

    def test_05_declare_quads_and_link(self):
        qs, g = self.create_qs_graph()
        print self.declare_qs_graph(qs)
        a = A()
        b = B()
        b.a = a
        session.add(b)
        session.add(a)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert list(graph.triples((None, TST.alink, None)))

    def test_06_conditional_prop(self):
        qs, g = self.create_qs_graph()
        g.add_patterns([
            QuadMapPattern(
                tb_iri.apply(B.id),
                TST.safe_name,
                B.name,
                condition=(B.name != None))
        ])
        print self.declare_qs_graph(qs)
        b = B(name='name')
        b2 = B()
        session.add(b)
        session.add(b2)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.safe_name, None))))
        assert 1 == len(list(graph.triples((None, TST.name, None))))

    def test_07_conditional_link(self):
        qs, g = self.create_qs_graph()
        g.add_patterns([
            QuadMapPattern(
                tb_iri.apply(B.id),
                TST.safe_alink,
                ta_iri.apply(B.a_id),
                condition=(B.a_id != None))
        ])
        print self.declare_qs_graph(qs)
        a = A()
        b = B(a=a)
        b2 = B()
        session.add(b)
        session.add(b2)
        session.add(a)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.safe_alink, None))))
        assert 2 == len(list(graph.triples((None, TST.alink, None))))

    def test_08_subclassing(self):
        qs, g = self.create_qs_graph()
        g.add_patterns([
            QuadMapPattern(
                tb_iri.apply(C.id),
                TST.cname,
                C.name)
        ])
        print self.declare_qs_graph(qs)
        b = B(name='b1')
        c = C(name='c1')
        session.add(b)
        session.add(c)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.cname, None))))
