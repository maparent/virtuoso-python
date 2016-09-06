from __future__ import print_function
from nose.tools import assert_raises
from sqlalchemy import Integer, String, MetaData, ForeignKey, Column
from sqlalchemy.engine import create_engine
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import sessionmaker, relation
from sqlalchemy.ext.declarative.api import as_declarative
from sqlalchemy.sql import text
from rdflib import Graph, Namespace
from rdflib.namespace import RDF
from virtuoso.quadextractor import ClassPatternExtractor

from virtuoso.vmapping import (
    VirtuosoQuadMapPattern, VirtuosoPatternIriClass, QuadStorage, VirtuosoGraphQuadMapPattern)
from virtuoso.vstore import Virtuoso, VirtuosoNamespaceManager
from . import sqla_connection

engine = create_engine(sqla_connection)
Session = sessionmaker(bind=engine)
session = Session(autocommit=False)
metadata = MetaData(schema="test.DBA")

TST = Namespace('http://example.com/test#')
nsm = VirtuosoNamespaceManager(Graph(), session)
nsm.bind('tst', TST)


class MyClassPatternExtractor(ClassPatternExtractor):
    def iri_accessor(self, sqla_cls):
        return super(MyClassPatternExtractor, self).iri_accessor(sqla_cls)

    def get_base_conditions(self, alias_maker, cls, for_graph):
        return super(MyClassPatternExtractor, self).get_base_conditions(
            alias_maker, cls, for_graph)

    def make_column_name(self, cls, column, for_graph):
        return getattr(TST, 'col_pattern_%s_%s' % (
                       cls.__name__, column.key))

    def class_pattern_name(self, cls, for_graph):
        return getattr(TST, 'class_pattern_' + cls.__name__)


@as_declarative(bind=engine, metadata=metadata)
class Base(object):
    def __init__(self, **kw):
        for k, v in kw.iteritems():
            setattr(self, k, v)


class A(Base):
    __tablename__ = "test_a"
    id = Column(Integer, primary_key=True)
    name = Column(
        String, info={'rdf': VirtuosoQuadMapPattern(None, TST.name, None)})


inspect(A).local_table.info = {
    "rdf_iri": VirtuosoPatternIriClass(
        TST.ta_iri, 'http://example.com/test#tA/%d', None,
        ('id', Integer, False)),
    "rdf_patterns": [VirtuosoQuadMapPattern(None, RDF.type, TST.tA)]
}


class B(Base):
    __tablename__ = "test_b"
    id = Column(Integer, primary_key=True)
    name = Column(String, info={'rdf': VirtuosoQuadMapPattern(None, TST.name, None)})
    type = Column(String(20))
    a_id = Column(Integer, ForeignKey(A.id), info={
        'rdf': VirtuosoQuadMapPattern(None, TST.alink)})
    a = relation(A)
    __mapper_args__ = {
        'polymorphic_identity': 'B',
        'polymorphic_on': type,
        'with_polymorphic': '*'
    }


inspect(B).local_table.info = {
    "rdf_iri": VirtuosoPatternIriClass(
        TST.tb_iri, 'http://example.com/test#tB/%d', None,
        ('id', Integer, False)),
    "rdf_patterns": [VirtuosoQuadMapPattern(None, RDF.type, TST.tB)]
}


class C(B):
    __tablename__ = "test_c"
    id = Column(Integer, ForeignKey(
        B.id, ondelete='CASCADE', onupdate='CASCADE'
    ), primary_key=True)

    __mapper_args__ = {
        'polymorphic_identity': 'C',
    }


class D(Base):
    __tablename__ = "test_d"
    id = Column(Integer, primary_key=True)
    name = Column(String, info={'rdf': VirtuosoQuadMapPattern(None, TST.name, None)})
    a_id = Column(Integer, ForeignKey(A.id))
    a = relation(A, info={
        'rdf': VirtuosoQuadMapPattern(None, TST.alink)})


inspect(D).local_table.info = {
    "rdf_iri": VirtuosoPatternIriClass(
        TST.td_iri, 'http://example.com/test#tD/%d', None,
        ('id', Integer, False)),
    "rdf_patterns": [VirtuosoQuadMapPattern(None, RDF.type, TST.tD)]
}


def clean():
    for table in ("test_table", "test_d",  "test_c", "test_b", "test_a"):
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
        qs = QuadStorage(self.qsname, None, nsm=nsm)
        try:
            print(qs.drop(session, True))
            for table in ("test_d", "test_c", "test_b", "test_a"):
                session.execute('delete from test..'+table)
            session.commit()
        except Exception as e:
            print(e)
            session.rollback()

    def create_qs_graph(self):
        cpe = MyClassPatternExtractor(Base._decl_class_registry)
        qs = QuadStorage(self.qsname, cpe, nsm=nsm, add_default=False)
        g = VirtuosoGraphQuadMapPattern(self.graphname, qs, None, None)
        qs.alias_manager = cpe  # Hack
        cpe.add_class(A, g)
        cpe.add_class(B, g)
        cpe.add_class(C, g)
        cpe.add_class(D, g)
        return qs, g, cpe

    def declare_qs_graph(self, qs):
        defn = qs.full_declaration_clause()
        print(defn.compile(engine))
        result = list(session.execute(defn))
        print(result)
        return result

    def test_05_declare_quads_and_link(self):
        qs, g, cpe = self.create_qs_graph()
        print(self.declare_qs_graph(qs))
        a = A()
        b = B()
        b.a = a
        session.add(b)
        session.add(a)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert list(graph.triples((None, TST.alink, None)))

    def test_05b_declare_quads_and_link(self):
        qs, g, cpe = self.create_qs_graph()
        td_iri = cpe.iri_accessor(D)
        print(self.declare_qs_graph(qs))
        a = A()
        d = D()
        d.a = a
        session.add(d)
        session.add(a)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert list(graph.triples((None, TST.alink, None)))

    def test_06_conditional_prop(self):
        qs, g, cpe = self.create_qs_graph()
        tb_iri = cpe.iri_accessor(B)
        cpe.add_pattern(
            B, VirtuosoQuadMapPattern(
                tb_iri.apply(B.id),
                TST.safe_name,
                B.name,
                conditions=(B.name != None,)),
            g)
        print(self.declare_qs_graph(qs))
        b = B(name='name')
        b2 = B()
        session.add(b)
        session.add(b2)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.safe_name, None))))
        assert 1 == len(list(graph.triples((None, TST.name, None))))

    def test_07_conditional_link(self):
        qs, g, cpe = self.create_qs_graph()
        ta_iri = cpe.iri_accessor(A)
        tb_iri = cpe.iri_accessor(B)
        cpe.add_pattern(
            B, VirtuosoQuadMapPattern(
                tb_iri.apply(B.id),
                TST.safe_alink,
                ta_iri.apply(B.a_id),
                conditions=(B.a_id != None,)),
            g)
        print(self.declare_qs_graph(qs))
        a = A()
        b = B(a=a)
        b2 = B()
        session.add(b)
        session.add(b2)
        session.add(a)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.safe_alink, None))))
        q = graph.triples((None, TST.alink, None))
        from pyodbc import DataError
        assert_raises(DataError, list, q)

    def test_08_subclassing(self):
        qs, g, cpe = self.create_qs_graph()
        tb_iri = cpe.iri_accessor(B)
        cpe.add_pattern(
            C, VirtuosoQuadMapPattern(
                tb_iri.apply(C.id),
                TST.cname,
                C.name),
            g)
        print(self.declare_qs_graph(qs))
        b = B(name='b1')
        c = C(name='c1')
        session.add(b)
        session.add(c)
        session.commit()
        graph = Graph(self.store, identifier=self.graphname)
        assert 1 == len(list(graph.triples((None, TST.cname, None))))
