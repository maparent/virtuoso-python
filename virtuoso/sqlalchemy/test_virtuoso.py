from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text, expression, bindparam
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey

engine = create_engine("virtuoso://dba:dba@VOS")
Session = sessionmaker(bind=engine)
metadata = MetaData()

test_table = Table('test_table', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('name', String)
                   )

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

def test_01_table():
    test_table.create(engine)
    assert table_exists(test_table)
    test_table.drop(engine)
    assert not table_exists(test_table)

def test_02_table_schema():
    test_table.schema = "TEST_SCHEMA"
    test_table.create(engine)
    assert table_exists(test_table)
    test_table.drop(engine)
    assert not table_exists(test_table)
