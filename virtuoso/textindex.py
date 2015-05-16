from sqlalchemy import Column, Integer
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.schema import _CreateDropBase, Table, Index
from sqlalchemy.sql.expression import (func, literal_column)
from sqlalchemy.sql import ddl
from sqlalchemy.sql.base import _bind_or_error


class TextIndex(Index):
    __visit_name__ = 'text_index'

    def __init__(
            self, column, clusters=None, key=None, language=None,
            encoding=None, do_insert=True, transform=None):
        column = self.normalize_column(column)
        self.column = column
        self.table = None
        super(TextIndex, self).__init__(None, column)
        self.clusters = [self.normalize_column(c) for c in (clusters or ())]
        self.key = self.normalize_column(key) if key else None
        self.language = language
        self.encoding = encoding
        self.do_insert = do_insert
        self.transform = transform

    def _set_parent(self, table):
        super(TextIndex, self)._set_parent(table)
        self.name = "{table}_{column}_WORDS".format(
            table=table.name,
            column=self.column.name)

    def create(self, bind=None):
        if bind is None:
            bind = _bind_or_error(self)
        bind._run_visitor(SchemaGeneratorWithTextIndex, self)
        return self

    def drop(self, bind=None, checkfirst=False):
        if bind is None:
            bind = _bind_or_error(self)
        bind._run_visitor(SchemaDropperWithTextIndex,
                          self,
                          checkfirst=checkfirst)

    @staticmethod
    def normalize_column(column):
        if isinstance(column, str):
            pass  # convert to column
        if isinstance(column, InstrumentedAttribute):
            mapper = column.parent
            column = mapper.c[column.name]
        assert isinstance(column, Column)
        return column

    def contains(self, query_str, ranges=None, offband=None, descending=False, score_limit=None,
                 start_id=None, end_id=None):
        """Creates a clause with contains arguments"""
        args = [self.column, query_str]
        if descending:
            args.append(literal_column('DESCENDING'))
        if start_id:
            args.extend((literal_column('START_ID'), start_id))
        if end_id:
            args.extend((literal_column('END_ID'), end_id))
        if score_limit:
            args.extend((literal_column('SCORE_LIMIT'), score_limit))
        if ranges:
            # Should be an alias
            args.extend((literal_column('RANGES'), ranges))
        if offband is None:
            offband = self.clusters
        else:
            offband = [self.normalize_column(c) for c in offband]
        for c in offband:
            args.extend((literal_column('OFFBAND'), c))
        return func.contains(*args)

    score_name = literal_column('SCORE', type_=Integer)


class CreateTextIndex(_CreateDropBase):
    """Represent a CREATE TEXT INDEX statement."""

    __visit_name__ = "create_text_index"


class DropTextIndex(_CreateDropBase):
    """Represent a DROP TEXT INDEX statement."""

    __visit_name__ = "drop_text_index"


class SchemaGeneratorWithTextIndex(ddl.SchemaGenerator):
    def visit_text_index(self, index):
        self.connection.execute(CreateTextIndex(index))


class SchemaDropperWithTextIndex(ddl.SchemaDropper):
    def visit_table(self, table, drop_ok=False, _is_metadata_operation=False):
        if not drop_ok and not self._can_drop_table(table):
            return
        # Ideally should come before the hook, but this will do
        if hasattr(table, 'indexes'):
            for index in table.indexes:
                self.traverse_single(index)
        super(SchemaDropperWithTextIndex, self).visit_table(
            table, drop_ok, _is_metadata_operation)


    def visit_text_index(self, index):
        self.connection.execute(DropTextIndex(index))


class TableWithTextIndex(Table):
    def create(self, bind=None, checkfirst=False):
        if bind is None:
            bind = _bind_or_error(self)
        bind._run_visitor(SchemaGeneratorWithTextIndex,
                          self,
                          checkfirst=checkfirst)

    def drop(self, bind=None, checkfirst=False):
        if bind is None:
            bind = _bind_or_error(self)
        bind._run_visitor(SchemaDropperWithTextIndex,
                          self,
                          checkfirst=checkfirst)

