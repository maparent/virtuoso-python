import re
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict

from sqlalchemy.types import TypeEngine
from rdflib import Namespace

VirtRDF = Namespace('http://www.openlinksw.com/schemas/virtrdf#')


class Mapping(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, nsm=None):
        self.name = name
        self.nsm = nsm

    @abstractproperty
    def mapping_name(self):
        pass

    def drop(self, nsm=None):
        nsm = nsm or self.nsm
        return "drop %s %s ." % (
            self.mapping_name, self.name.n3(nsm))

    def patterns_iter(self):
        return ()

    @abstractmethod
    def virt_def(self, nsm=None):
        pass

    def definition_statement(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        prefixes = "\n".join("PREFIX %s: %s " % (
            p, ns.n3()) for (p, ns) in nsm.namespaces()) if nsm else ''
        patterns = set(self.patterns_iter())
        patterns = '\n'.join((p.virt_def(nsm, engine) for p in patterns))
        return '%s\n%s\n%s\n' % (
            prefixes, patterns, self.virt_def(nsm))


class IriClass(Mapping):
    @property
    def mapping_name(self):
        return "iri class"


class PatternIriClass(IriClass):
    #parse_pattern = re.compile(r'(%(?:\{\w+\})?[dsU])')
    parse_pattern = re.compile(r'(%[dsU])')

    def __init__(self, name, pattern, nsm=None, *args):
        """args must be triples of (name, sql type, and nullable(bool))
        sql type must be a sqlalchemy type or sqlalchemy type instance
        """
        super(PatternIriClass, self).__init__(name, nsm)
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

    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        dialect = engine.dialect if engine else None
        return 'create %s %s "%s" (%s) . ' % (
            self.mapping_name, self.name.n3(nsm), self.pattern,
            ','.join(["in %s %s %s" % (
                vname, vtype.compile(dialect),
                '' if self.nullable[vname] else 'not null')
                for vname, vtype in self.vars.items()]))

    def patterns_iter(self):
        yield self


class QuadMapPattern(Mapping):
    __metaclass__ = ABCMeta

    def __init__(self, name=None, storage=None, nsm=None):
        super(QuadMapPattern, self).__init__(name, nsm)
        self.storage = storage

    @property
    def mapping_name(self):
        return "quad map"

    def set_col(self, column):
        pass

    @abstractmethod
    def virt_def(self, nsm=None):
        pass

    def import_stmt(self, nsm=None):
        nsm = nsm or self.nsm
        assert self.name and self.storage and self.storage.name
        return "create %s using storage %s . " % (
            self.name.n3(nsm), self.storage.name.n3(nsm))


def _qual_name(col):
    assert col.table.schema
    return "%s.%s.%s" % (col.table.schema, col.table.name, col.name)


class LiteralQuadMapPattern(QuadMapPattern):
    def __init__(self, prop, col=None, name=None, nsm=None):
        super(LiteralQuadMapPattern, self).__init__(name, nsm)
        self.property = prop
        self.column = col

    def set_col(self, column):
        self.column = column

    def virt_def(self, nsm=None):
        nsm = nsm or self.nsm
        stmt = "%s %s " % (self.property.n3(nsm), _qual_name(self.column))
        if self.name:
            stmt += " as %s " % (self.name.n3(nsm),)
        return stmt


class IriSubjectQuadMapPattern(QuadMapPattern):
    def __init__(self, iri_class, name=None, nsm=None, *cols):
        super(IriSubjectQuadMapPattern, self).__init__(name, nsm)
        self.iri_class = iri_class
        self.columns = cols

    def set_col(self, column):
        self.columns = [column]

    def virt_def(self, nsm=None):
        nsm = nsm or self.nsm
        return "%s (%s) " % (self.iri_class.name.n3(nsm), ', '.join(
            _qual_name(c) for c in self.columns))

    def patterns_iter(self):
        for pat in self.iri_class.patterns_iter():
            yield pat


class IriQuadMapPattern(QuadMapPattern):
    def __init__(self, iri_class, prop, name=None, nsm=None, *cols):
        super(IriQuadMapPattern, self).__init__(name, nsm)
        self.iri_class = iri_class
        self.property = prop
        self.columns = cols

    def set_col(self, column):
        self.columns = [column]

    def virt_def(self, nsm=None):
        nsm = nsm or self.nsm
        stmt = "%s %s (%s) " % (
            self.property.n3(nsm), self.iri_class.name.n3(nsm), ', '.join(
                _qual_name(c) for c in self.columns))
        if self.name:
            stmt += " as %s " % (self.name.n3(nsm),)
        return stmt

    def patterns_iter(self):
        for pat in self.iri_class.patterns_iter():
            yield pat


class RdfClassPattern(QuadMapPattern):
    def __init__(self, rdf_class, name=None, nsm=None):
        super(RdfClassPattern, self).__init__(name, nsm)
        self.rdf_class = rdf_class

    def virt_def(self, nsm=None):
        nsm = nsm or self.nsm
        return "a %s" % (self.rdf_class.n3(nsm))


class ClassQuadMapPattern(QuadMapPattern):
    def __init__(self, sqla_cls, rdf_class=None, subject_pattern=None,
                 name=None, nsm=None, *patterns):
        super(ClassQuadMapPattern, self).__init__(None, nsm)
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
        self.patterns.insert(0, RdfClassPattern(rdf_class, name, nsm))

    def virt_def(self, nsm=None):
        nsm = nsm or self.nsm
        return self.subject_pattern.virt_def(nsm) + ' ;\n'.join(
            qmp.virt_def(nsm) for qmp in self.patterns) + ' .\n'

    def patterns_iter(self):
        for c in self.patterns:
            for pat in c.patterns_iter():
                yield pat
        for pat in self.subject_pattern.patterns_iter():
            yield pat


class GraphQuadMapPattern(QuadMapPattern):
    def __init__(self, graph_iri, name=None, nsm=None, option=None, *qmps):
        super(GraphQuadMapPattern, self).__init__(name, nsm)
        self.iri = graph_iri
        self.qmps = qmps
        self.option = option

    def virt_def(self, nsm=None):
        nsm = nsm or self.nsm
        inner = ''.join((qmp.virt_def(nsm) for qmp in self.qmps))
        stmt = 'graph %s %s {\n%s\n}' % (
            self.iri.n3(nsm),
            'option(%s)' % (self.option) if self.option else '',
            inner)
        if self.name:
            stmt = 'create %s as %s . ' (self.name.n3(nsm), stmt)
        return stmt

    def patterns_iter(self):
        for qmp in self.qmps:
            for pat in qmp.patterns_iter():
                yield pat


class QuadStorage(Mapping):
    def __init__(self, name, native_graphmaps, imported_graphmaps=None,
                 add_default=True, nsm=None):
        super(QuadStorage, self).__init__(name, nsm)
        self.native_graphmaps = native_graphmaps
        self.imported_graphmaps = imported_graphmaps or []
        self.add_default = add_default
        for gmap in native_graphmaps:
            gmap.storage = self

    @property
    def mapping_name(self):
        return "quad storage"

    def virt_def(self, nsm=None):
        nsm = nsm or self.nsm
        native = '\n'.join(gqm.virt_def(nsm) for gqm in self.native_graphmaps)
        imported = '\n'.join(gqm.import_stmt(nsm)
                             for gqm in self.imported_graphmaps)
        if self.add_default:
            imported += '.' + DefaultQuadMap.import_stmt(nsm)
        return 'create %s %s {\n %s \n}' % (
            self.mapping_name, self.name.n3(nsm),
            '\n'.join((native, imported)))

    def add_imported(self, qmap, nsm=None):
        return 'alter %s %s {\n %s \n}' % (
            self.mapping_name, self.name.n3(nsm), qmap.import_stmt(nsm))

    def patterns_iter(self):
        for qmp in self.native_graphmaps:
            for pat in qmp.patterns_iter():
                yield pat
        for qmp in self.imported_graphmaps:
            for pat in qmp.patterns_iter():
                yield pat

DefaultQuadMap = GraphQuadMapPattern(None, VirtRDF.DefaultQuadMap)
DefaultQuadStorage = QuadStorage(VirtRDF.DefaultQuadStorage,
                                 [DefaultQuadMap], add_default=False)
