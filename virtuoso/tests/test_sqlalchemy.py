from sqlalchemy.engine import create_engine
from sqlalchemy.exc import ProgrammingError, DBAPIError
from sqlalchemy.orm import sessionmaker, mapper, relation, backref
from sqlalchemy.sql import text, expression, bindparam
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

engine = create_engine("virtuoso://dba:dba@VOS")
Session = sessionmaker(bind=engine)
session = Session(autocommit=False)
metadata = MetaData()

test_table = Table('test_table', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('name', String),
                   schema="test"
                   )
class Object(object):
    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

test_table_a = Table("test_a", metadata,
                     Column("id", Integer, primary_key=True),
                     Column('name', String),
                     schema="test")
test_table_b = Table("test_b", metadata,
                     Column("id", Integer, primary_key=True),
                     Column('name', String),
                     Column("a_id", Integer, ForeignKey(test_table_a.c.id)),
                     schema="test")
class A(Object): pass
class B(Object): pass

def table_exists(table):
    conn = engine.connect()
    result = conn.execute(
        text("SELECT TABLE_NAME FROM TABLES WHERE "
             "lower(TABLE_SCHEMA) = :schemaname AND "
             "lower(TABLE_NAME) = :tablename",
             bindparams=[
                 bindparam("tablename", table.name),
                 bindparam("schemaname", table.schema if table.schema else "DBA")
                 ])
        )
    return result.scalar() is not None

def clean():
    for table in ("test_table", "test_b", "test_a"):
        conn = engine.connect()
        result = conn.execute(
            text("SELECT 1 FROM TABLES WHERE "
                 "lower(TABLE_NAME) = '%s'" % table)
            )
        if result.scalar():
            conn.execute(text("DROP TABLE %s" % table))

class Test01Basic(object):
    @classmethod
    def setup_class(self):
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

class Test02Object(object):
    @classmethod
    def setup_class(cls):
        clean()
        test_table.create(engine)
        mapper(Object, test_table)

    def test_01_insert(self):
        o1 = Object(name = "foo")
        session.add(o1)
        o2 = Object(name = "bar")
        session.add(o2)
        session.commit()

        o = session.query(Object).get(1)
        assert o.name == "foo"
        o = session.query(Object).get(2)
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

class Test03Relation(object):
    @classmethod
    def setup_class(cls):
        clean()
        test_table_a.create(engine)
        test_table_b.create(engine)
        mapper(A, test_table_a)
        mapper(B, test_table_b, properties={ 'a': relation(A) })

    def test_01_create(self):
        a = A()
        b = B()
        b.a = a
        # NB, a gets implicitly added
        session.add(b)
        session.commit()
    
        b = session.query(B).get(1)
        assert isinstance(b.a, A)

    def test_02_update(self):
        c = A()
        b = session.query(B).get(1)
        _oldid = b.a.id
        b.a = c
        session.add(b)
        session.commit()

        b = session.query(B).get(1)
        assert isinstance(b.a, A)
        assert b.a.id != _oldid

    def test_03_fkey_violation(self):
        b = session.query(B).get(1)
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
