from sqlalchemy.engine import create_engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import sessionmaker, mapper
from sqlalchemy.sql import text, expression, bindparam
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

engine = create_engine("virtuoso://dba:dba@VOS")
Session = sessionmaker(bind=engine)
session = Session(autocommit=False)
metadata = MetaData()

test_table = Table('test_table', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('name', String)
                   )

class Object(object):
    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

def table_exists(table):
    conn = engine.connect()
    result = conn.execute(
        text("SELECT TABLE_NAME FROM TABLES WHERE "
             "TABLE_SCHEMA=:schemaname AND "
             "TABLE_NAME=:tablename",
             bindparams=[
                 bindparam("tablename", table.name.upper()),
                 bindparam("schemaname", table.schema.upper() if table.schema else "DBA")
                 ])
        )
    return result.scalar() is not None

def clean():
    for table in ("TEST_TABLE", "TEST_A", "TEST_B"):
        conn = engine.connect()
        result = conn.execute(
            text("SELECT 1 FROM TABLES WHERE "
                 "TABLE_NAME='%s'" % table)
            )
        if result.scalar():
            conn.execute(text("DROP TABLE %s" % table))

class Test01Basic(object):
    @classmethod
    def setup_class(self):
        clean()
    @classmethod
    def teardown_class(self):
        clean()

    def test_01_table(self):
        test_table.create(engine)
        assert table_exists(test_table)
        test_table.drop(engine)
        assert not table_exists(test_table)

    def test_02_table_schema(self):
        test_table.schema = "TEST_SCHEMA"
        test_table.create(engine)
        assert table_exists(test_table)
        test_table.drop(engine)
        assert not table_exists(test_table)

    def test_03_fkey(self):
        a = Table("test_a", metadata,
                  Column("id", Integer, primary_key=True))
        b = Table("test_b", metadata,
                  Column("id", Integer, primary_key=True),
                  Column("a", Integer, ForeignKey("test_a.id")))
        a.create(engine)
        b.create(engine)
        try:
            a.drop(engine)
            assert False, "Should not be able to drop %s because of FKEY" % a
        except ProgrammingError:
            pass
        b.drop(engine)
        a.drop(engine)

class Test02Object(object):
    @classmethod
    def setup_class(cls):
        clean()
        test_table.create(engine)
        mapper(Object, test_table)

    @classmethod
    def teardown_class(cls):
        clean()

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

    def test_02_delete(self):
        o = Object(name = "delete")
        session.add(o)
        session.commit()
        
        o = session.query(Object).filter(Object.name == "delete").all()[0]
        session.delete(o)
        session.commit()

        assert session.query(Object).filter(Object.name == "delete").count() == 0
