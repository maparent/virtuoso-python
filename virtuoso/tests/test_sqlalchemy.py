from nose.plugins.skip import SkipTest

from sqlalchemy.engine import create_engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker, mapper, relation
from sqlalchemy.sql import text, bindparam
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

from rdflib import URIRef, Graph
from rdflib.namespace import Namespace, NamespaceManager

from virtuoso.vmapping import *

engine = create_engine("virtuoso://dba:dba@VOS")
Session = sessionmaker(bind=engine)
session = Session(autocommit=False)
metadata = MetaData()

TST = Namespace('http://example.com/test#')
nsm = NamespaceManager(Graph())
nsm.bind('tst', TST)
nsm.bind('virt', VirtRDF)

ta_iri = PatternIriClass(TST.ta_iri,'http://example.com/test#tA/%d', None, ('id', Integer, False))
tb_iri = PatternIriClass(TST.tb_iri,'http://example.com/test#tB/%d', None, ('id', Integer, False))

test_table = Table('test_table', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('name', String),
                   schema="test.DBA",
                   )


class Object(object):
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

test_table_a = Table("test_a", metadata,
                     Column("id", Integer, primary_key=True,
                            info={'rdf': IriSubjectQuadMapPattern(ta_iri)}),
                     Column('name', String,
                            info={'rdf': LiteralQuadMapPattern(TST.name)}),
                     schema="test.DBA",
                     info={"rdf_class":TST.tA})
test_table_b = Table("test_b", metadata,
                     Column("id", Integer, primary_key=True,
                            info={'rdf': IriSubjectQuadMapPattern(tb_iri)}),
                     Column('name', String,
                            info={'rdf': LiteralQuadMapPattern(TST.name)}),
                     Column("a_id", Integer, ForeignKey(test_table_a.c.id),
                            info={'rdf': IriQuadMapPattern(ta_iri, TST.alink)}),
                     schema="test.DBA",
                     info={"rdf_class":TST.tB})
test_table_c = Table("test_c", metadata,
                     Column("id", Integer, primary_key=True, autoincrement=False),
                     Column('name', String),
                     schema="test.DBA")


class A(Object):
    pass


class B(Object):
    pass


def table_exists(table):
    conn = engine.connect()
    catalog, schema = table.schema.split('.', 1) if table.schema else (None, None)
    result = conn.execute(
        text("SELECT TABLE_NAME FROM TABLES WHERE "
             "lower(TABLE_CATALOG) = lower(:catname) AND "
             "lower(TABLE_SCHEMA) = lower(:schemaname) AND "
             "lower(TABLE_NAME) = lower(:tablename)"),
        tablename = table.name, schemaname=schema, catname=catalog)
    return result.scalar() is not None


def clean():
    for table in ("test_table", "test_b", "test_a", "test_c"):
        conn = engine.connect()
        result = conn.execute(
            text("SELECT TABLE_CATALOG FROM TABLES WHERE "
                 "lower(TABLE_NAME) = '%s'" % table)
        )
        for s in result.fetchall():
            conn.execute(text("DROP TABLE %s..%s" % (s[0], table)))
            session.commit()


class Test01Basic(object):
    @classmethod
    def setup_class(self):
        clean()

    @classmethod
    def teardown_class(self):
        clean()

    def test_01_table(self):
        test_table.create(engine)
        try:
            assert table_exists(test_table)
        finally:
            test_table.drop(engine)
        assert not table_exists(test_table)

    def test_02_fkey(self):
        test_table_a.create(engine)
        test_table_b.create(engine)
        try:
            test_table_a.drop(engine)
            assert False, "Should not be able to drop %s because of FKEY" % test_table_a
        except DBAPIError:
            pass
        test_table_b.drop(engine)
        test_table_a.drop(engine)

    def test_03_rollback(self):
        test_table_a.create(engine)
        session.rollback()
        assert not table_exists(test_table)

    def test_04_rollback_after_error(self):
        test_table_c.create(engine)
        session.commit()
        assert table_exists(test_table_c)
        session.execute(text("insert into test..test_c values (1, 'a')"))
        ex = False
        try:
            session.execute(text("insert into test..test_c (name) values ('b')"))
        except DBAPIError:
            ex = True
            session.rollback()
        assert ex, "The invalid insert did not throw an exception???"
        r = session.execute(text("select count(id) from test..test_c where name='a'"))
        assert r.scalar() == 0


class Test02Object(object):
    @classmethod
    def setup_class(cls):
        clean()
        test_table.create(engine)
        mapper(Object, test_table)

    def teardown(self):
        session.rollback()

    @classmethod
    def teardown_class(self):
        clean()

    def test_01_insert(self):
        o1 = Object(name="foo")
        session.add(o1)
        o2 = Object(name="bar")
        session.add(o2)
        session.commit()

        o = session.query(Object).get(1)
        assert o
        assert o.name == "foo"
        o = session.query(Object).get(2)
        assert o
        assert o.name == "bar"

    def test_02_update(self):
        assert session.query(Object).filter(Object.name == "foo").count() == 1

        o = session.query(Object).filter(Object.name == "foo").all()[0]
        _oid = o.id
        o.name = "baz"
        session.add(o)
        session.commit()

        assert session.query(Object).filter(Object.name == "foo").count() == 0
        assert session.query(Object).filter(Object.name == "baz").count() == 1

        o = session.query(Object).filter(Object.name == "baz").all()[0]
        assert o.id == _oid

    def test_03_delete(self):
        [session.delete(o) for o in
         session.query(Object).filter(Object.name == "baz").all()]
        session.commit()

        assert session.query(Object).filter(Object.name == "baz").count() == 0
        assert session.query(Object).count() > 0

        [session.delete(o) for o in
         session.query(Object).all()]
        session.commit()

        assert session.query(Object).count() == 0

    def test_04_identity(self):
        o1 = Object(name="foo")
        session.add(o1)
        session.flush()
        id = o1.id
        assert id
        session.commit()
        o1 = session.query(Object).filter(Object.name == "foo").one()
        assert o1.id == id


class Test03Relation(object):
    @classmethod
    def setup_class(cls):
        clean()
        test_table_a.create(engine)
        test_table_b.create(engine)
        A.__mapper__=mapper(A, test_table_a)
        B.__mapper__=mapper(B, test_table_b, properties={'a': relation(A)})

    @classmethod
    def teardown_class(self):
        clean()

    def test_01_create(self):
        a = A()
        b = B()
        b.a = a
        # NB, a gets implicitly added
        session.add(b)
        session.commit()

        b = session.query(B).get(1)
        assert b
        assert isinstance(b.a, A)

    def test_02_update(self):
        c = A()
        b = session.query(B).get(1)
        assert b
        _oldid = b.a.id
        b.a = c
        session.add(b)
        session.commit()

        b = session.query(B).get(1)
        assert b
        assert isinstance(b.a, A)
        assert b.a.id != _oldid

    def test_03_fkey_violation(self):
        b = session.query(B).get(1)
        assert b
        # delete out from under, should raise a foreign key
        # constraint because we haven't set cascade on the
        # relation
        try:
            session.delete(b.a)
            session.commit()
            raise ValueError("Should have had an exception because of FK")
        except:
            session.rollback()

    def test_04_delete(self):
        b = session.query(B).get(1)
        # delete out from under, should raise a foreign key
        # constraint because we haven't set cascade on the
        # relation
        session.delete(b)
        session.delete(b.a)
        session.commit()

        assert session.query(A).count() == 1
        assert session.query(B).count() == 0

    def test_05_declare_quads(self):
        ap=ClassQuadMapPattern(A)
        bp=ClassQuadMapPattern(B)
        g=GraphQuadMapPattern(TST.g, None, None, None, ap, bp)
        qs = QuadStorage(TST.qs, [g])
        defn = qs.definition_statement(nsm)
        print defn
        r = session.execute('sparql '+defn)
        #for x in r.fetchall(): print x
        a = A()
        b = B()
        b.a = a
        session.add(b)
        session.commit()
        from virtuoso.vstore import Virtuoso
        store = Virtuoso("DSN=VOS;UID=dba;PWD=dba;WideAsUTF16=Y", quad_storage=qs.name)
        graph = Graph(store, identifier=TST.g)
        assert list(graph.triples((None, TST.alink, None)))
