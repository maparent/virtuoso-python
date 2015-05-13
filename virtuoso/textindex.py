from sqlalchemy import Column
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.expression import TextClause, func, literal_column
from sqlalchemy.types import (
    CHAR, VARCHAR, NCHAR, NVARCHAR, String, UnicodeText, Unicode, Text)

from .alchemy import XML, LONGXML, LONGVARCHAR, LONGNVARCHAR, CoerceUnicode

TEXT_TYPES = (CHAR, VARCHAR, NCHAR, NVARCHAR, String, UnicodeText,
              Unicode, Text, LONGVARCHAR, LONGNVARCHAR, CoerceUnicode)

class TextIndex(object):
    def __init__(
            self, column, clusters=None, key=None, language=None,
            encoding=None, do_insert=True, transform=None):
        self.column = self.normalize_column(column)
        self.clusters = [self.normalize_column(c) for c in (clusters or ())]
        self.key = self.normalize_column(key) if key else None
        self.language = language
        self.encoding = encoding
        self.do_insert = do_insert
        self.transform = transform

    @staticmethod
    def normalize_column(column):
        if isinstance(column, str):
            pass  # convert to column
        if isinstance(column, InstrumentedAttribute):
            mapper = column.parent
            column = mapper.c[column.name]
        assert isinstance(column, Column)
        return column

    def create_statement(self):
        column = self.column
        params = dict(table=column.table.name, column=column.name)
        for x in ('xml','clusters','key','with_insert','transform','language','encoding'):
            params[x] =''
        if isinstance(column.type, (XML, LONGXML)):
            params['xml'] = 'XML'
        else:
            assert isinstance(column.type, TEXT_TYPES)
        if self.clusters:
            params['clusters'] = 'CLUSTERED WITH (' + ','.join((
                '"%s"' % (c.name,) for c in self.clusters)) + ')'
        if self.key:
            params['key'] = 'WITH KEY "' + self.key.name + '"'
        if not self.do_insert:
            params['with_insert'] = 'NO INSERT'
        if self.transform:
            params['transform'] = 'USING ' + self.transform
        if self.language:
            params['language'] = 'LANGUAGE ' + self.language
        if self.encoding:
            params['encoding'] = 'ENCODING ' + self.encoding
        return TextClause(
            'CREATE TEXT {xml} INDEX ON "{table}" ( "{column}" ) {key} '
            '{with_insert} {clusters} {transform} {language} {encoding}'.format(**params))

    def drop_statement(self):
        column = self.column
        column = self.column
        return TextClause(
            'DROP TABLE {table}_{column}_WORDS'.format(
                table=column.table.name, column=column.name))

    def contains(self, query_str, ranges=None, offband=None, descending=False, score_limit=None,
                 start_id=None, end_id=None):
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
