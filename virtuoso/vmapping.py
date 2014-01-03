import re
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from sqlalchemy.types import TypeEngine
from rdflib import Namespace

VirtRDF = Namespace('http://www.openlinksw.com/schemas/virtrdf#')


class IriClass(object):
    __metaclass__ = ABCMeta


class PatternIriClass(IriClass):
    #parse_pattern = re.compile(r'(%(?:\{\w+\})?[dsU])')
    parse_pattern = re.compile(r'(%[dsU])')

    def __init__(self, name, pattern, *args):
        """args must be triples of (name, sql type, and nullable(bool))
        sql type must be a sqlalchemy type or sqlalchemy type instance
        """
        self.name = name
        self.pattern = pattern
        self.varnames = [arg[0] for arg in args]
        self.vars = OrderedDict((arg[0:2] for arg in args))
        for k, v in self.vars.items():
            if not isinstance(v, TypeEngine):
                assert isinstance(v, type) and TypeEngine in v.mro()
                self.vars[k] = v()

        self.nullable = dict((arg[0::2] for arg in args))
        pieces = self.parse_pattern.split(pattern)
        self.is_int = [x[1] == 'd' for x in pieces[1::2]]
        assert len(pieces) // 2 == len(self.varnames),\
            "number of pieces and variables must agree"

        def _re_pattern_builder(pos, pattern):
            if pos % 2 == 0:
                return re.escape(pattern)
            pos = pos // 2
            varname = self.varnames[pos]
            star = '*' if self.nullable[varname] else '+'
            code = pattern[1]
            if code == 'd':
                cclass = '[0-9]'
            elif code == 'U':
                # Url-valid chars
                cclass = r"[-A-Za-z0-9_\.~]"
            else:
                cclass = '.'
            return '(%s%s)' % (cclass, star)

        self.pattern_matcher = re.compile(''.join((
            _re_pattern_builder(pos, pat)
            for (pos, pat) in enumerate(pieces))))

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == len(self.varnames):
            args = [kwargs[x] for x in self.varnames]
        elif len(kwargs) != 0 or len(args) != len(self.varnames):
            raise ArgumentError()
        return self.pattern % args

    def parse(self, iri):
        r = self.pattern_matcher.match(iri)
        assert r, "The iri does not match " + self.pattern
        assert len(r.group(0)) == len(iri),\
            "The iri does not match " + self.pattern
        vals = [int(v) if self.is_int[p] else v
                for p, v in enumerate(r.groups())]
        return dict(zip(self.varnames, vals))

    def virt_def(self, engine=None):
        dialect = engine.dialect if engine else None
        return 'create iri class %s "%s" (%s) .' % (
            self.name, self.pattern, ','.join(["in %s %s %s" % (
                vname, vtype.compile(dialect),
                '' if self.nullable[vname] else 'not null')
                for vname, vtype in self.vars.items()]))

    def drop(self, ns=None):
        return "drop iri class %s ." % (self.name.n3(ns),)


class QuadMapPattern(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        self.name = name

    def drop(self, ns=None):
        if self.name:
            return "drop quad map %s ." % (self.name.n3(ns),)

    def set_col(self, column):
        pass

    @abstractmethod
    def virt_def(self, ns=None):
        pass


def _qual_name(col):
    if col.table.schema:
        return "%s..%s.%s" % (col.table.schema, col.table.name, col.name)
    else:
        return "%s.%s" % (col.table.name, col.name)


class LiteralQuadMapPattern(QuadMapPattern):
    def __init__(self, prop, col=None, name=None):
        super(LiteralQuadMapPattern, self).__init__(name)
        self.property = prop
        self.column = col

    def set_col(self, column):
        self.column = column

    def virt_def(self, ns=None):
        stmt = "%s %s " % (self.property.n3(ns), _qual_name(self.column))
        if self.name:
            stmt += " as %s " % (self.name.n3(ns),)
        return stmt


class IriSubjectQuadMapPattern(QuadMapPattern):
    def __init__(self, iri_class, name=None, *cols):
        super(IriSubjectQuadMapPattern, self).__init__(name)
        self.iri_class = iri_class
        self.columns = cols

    def set_col(self, column):
        self.columns = [column]

    def virt_def(self, ns=None):
        return "%s (%s) " % (self.iri_class.name, ', '.join(
            _qual_name(c) for c in self.columns))


class IriQuadMapPattern(QuadMapPattern):
    def __init__(self, iri_class, prop, name=None, *cols):
        super(IriQuadMapPattern, self).__init__(name)
        self.iri_class = iri_class
        self.property = prop
        self.columns = cols

    def set_col(self, column):
        self.columns = [column]

    def virt_def(self, ns=None):
        stmt = "%s %s (%s) " % (
            self.property.n3(ns), self.iri_class.name, ', '.join(
                _qual_name(c) for c in self.columns))
        if self.name:
            stmt += " as %s " % (self.name.n3(ns),)
        return stmt


class RdfClassPattern(QuadMapPattern):
    def __init__(self, rdf_class, name=None):
        super(RdfClassPattern, self).__init__(name)
        self.rdf_class = rdf_class

    def virt_def(self, ns=None):
        return "a %s" % (self.rdf_class.n3(ns))


class ClassQuadMapPattern(QuadMapPattern):
    def __init__(self, sqla_cls, rdf_class=None, subject_pattern=None,
                 name=None, *patterns):
        super(ClassQuadMapPattern, self).__init__(None)
        self.patterns = list(patterns)
        self.subject_pattern = subject_pattern
        self.sqla_cls = sqla_cls

        mapper = sqla_cls.__mapper__
        rdf_class = rdf_class or mapper.mapped_table.info['rdf_class']
        for c in mapper.columns:
            if 'rdf' in c.info:
                qmp = c.info['rdf']
                qmp.set_col(c)
                if not subject_pattern and isinstance(
                        qmp, IriSubjectQuadMapPattern) and c.primary_key:
                    self.subject_pattern = qmp
                else:
                    self.patterns.append(qmp)
        assert self.subject_pattern
        self.patterns.insert(0, RdfClassPattern(rdf_class, name))

    def virt_def(self, ns=None):
        return self.subject_pattern.virt_def(ns) + ' ; '.join(
            qmp.virt_def(ns) for qmp in self.patterns) + ' . '


class GraphQuadMapPattern(QuadMapPattern):
    def __init__(self, graph_iri, name=None, option=None, *qmps):
        super(GraphQuadMapPattern, self).__init__(name)
        self.iri = graph_iri
        self.qmps = qmps
        self.option = option

    def virt_def(self, ns=None):
        inner = ''.join((qmp.virt_def(ns) for qmp in self.qmps))
        stmt = 'graph %s %s {%s}' % (
            self.iri.n3(ns),
            'option(%s)' % (self.option) if self.option else '',
            inner)
        if self.name:
            stmt = 'create %s as %s' (self.name.n3(ns), stmt)
        return stmt


class QuadStorage(object):
    def __init__(self, name, graphqmps):
        self.name = name
        self.graphqms = graphqms

    def virt_def(self, ns=None):
        stmt = 'create quad storage %s { %s } .' % (
            self.name.n3(ns), '\n'.join(
                gqm.virt_def(ns) for gqm in self.graphqms))

    def drop(self, ns=None):
        if self.name:
            return "drop quad storage %s ." % (self.name.n3(ns),)

