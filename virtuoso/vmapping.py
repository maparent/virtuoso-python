import re
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict

from sqlalchemy.types import TypeEngine
from rdflib import Namespace, RDF

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
    def virt_def(self, nsm=None, engine=None):
        pass

    def definition_statement(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        prefixes = "\n".join("PREFIX %s: %s " % (
            p, ns.n3()) for (p, ns) in nsm.namespaces()) if nsm else ''
        patterns = set(self.patterns_iter())
        patterns = '\n'.join((p.virt_def(nsm, engine) for p in patterns))
        return '%s\n%s\n%s\n' % (
            prefixes, patterns, self.virt_def(nsm, engine))


class IriClass(Mapping):
    @property
    def mapping_name(self):
        return "iri class"

    def virt_def(self, nsm=None, engine=None):
        return ''

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

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if self.name != other.name and (
            self.name is not None or other.name is not None):
            return False
        if self.pattern != other.pattern:
            return False
        return True

    def __hash__(self):
        return hash(self.name) if self.name else hash(self.pattern)


class QuadMapPattern(Mapping):
    __metaclass__ = ABCMeta

    def __init__(self, name=None, storage=None, nsm=None):
        super(QuadMapPattern, self).__init__(name, nsm)
        self.storage = storage

    @property
    def mapping_name(self):
        return "quad map"

    def set_columns(self, *columns):
        pass

    @abstractmethod
    def virt_def(self, nsm=None, engine=None):
        pass

    def import_stmt(self, nsm=None):
        nsm = nsm or self.nsm
        assert self.name and self.storage and self.storage.name
        return "create %s using storage %s . " % (
            self.name.n3(nsm), self.storage.name.n3(nsm))

    def resolve(self, sqla_cls):
        pass

def _qual_name(col, engine):
    assert col.table.schema
    if engine:
        prep = engine.dialect.identifier_preparer
        return '%s.%s' % (prep.format_table(col.table, True),
                          prep.format_column(col))
    else:
        return "%s.%s.%s" % (col.table.schema, col.table.name, col.name)


class ApplyIriClass(Mapping):
    def __init__(self, iri_class, *arguments):
        super(ApplyIriClass, self).__init__(None, iri_class.nsm)
        self.iri_class = iri_class
        self.arguments = list(arguments)

    def resolve(self, sqla_cls):
        columns = sqla_cls.__mapper__.mapped_table.columns
        for i, arg in enumerate(self.arguments):
            if isinstance(arg, str) and arg in columns:
                self.arguments[i] = getattr(sqla_cls, arg)

    def set_columns(self, *columns):
        self.arguments = columns

    @staticmethod
    def _argument(arg, nsm=None, engine=None):
        if getattr(arg, 'n3', None) is not None:
            return arg.n3(nsm)
        elif getattr(arg, 'table', None) is not None:
            return _qual_name(arg, engine)
        raise ArgumentError()

    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        return "%s (%s) " % (
            self.iri_class.name.n3(nsm), ', '.join(
                ApplyIriClass._argument(arg, nsm, engine) for arg in self.arguments))

    @property
    def mapping_name(self):
        return None

class ConstantQuadMapPattern(QuadMapPattern):
    def __init__(self, prop, object, name=None, nsm=None):
        super(ConstantQuadMapPattern, self).__init__(name, nsm)
        self.property = prop
        self.object = object

    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        stmt = "%s %s " % (self.property.n3(nsm), self.object.n3(nsm))
        if self.name:
            stmt += " as %s " % (self.name.n3(nsm),)
        return stmt


# convenience
class RdfClassQuadMapPattern(ConstantQuadMapPattern):
    def __init__(self, rdf_class, name=None, nsm=None):
        super(RdfClassQuadMapPattern, self).__init__(RDF.type, rdf_class, name, nsm)


class LiteralQuadMapPattern(QuadMapPattern):
    def __init__(self, prop, column=None, name=None, nsm=None):
        super(LiteralQuadMapPattern, self).__init__(name, nsm)
        self.property = prop
        self.column = column

    def set_columns(self, *columns):
        self.column = columns[0]

    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        stmt = "%s %s " % (self.property.n3(nsm), _qual_name(self.column, engine))
        if self.name:
            stmt += " as %s " % (self.name.n3(nsm),)
        return stmt

    def resolve(self, sqla_cls):
        columns = sqla_cls.__mapper__.mapped_table.columns
        if isinstance(self.column, str) and self.column in columns:
            self.column = getattr(sqla_cls, self.column)


class IriQuadMapPattern(QuadMapPattern):
    def __init__(self, prop, apply_iri_class, name=None, nsm=None):
        super(IriQuadMapPattern, self).__init__(name, nsm)
        self.apply_iri_class = apply_iri_class
        self.property = prop

    def set_columns(self, *columns):
        self.apply_iri_class.set_columns(*columns)

    def resolve(self, sqla_cls):
        self.apply_iri_class.resolve(sqla_cls)


    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        stmt = "%s %s" % (
            self.property.n3(nsm), self.apply_iri_class.virt_def(nsm, engine))
        if self.name:
            stmt += " as %s " % (self.name.n3(nsm),)
        return stmt

    def patterns_iter(self):
        for pat in self.apply_iri_class.patterns_iter():
            yield pat



class ClassQuadMapPattern(QuadMapPattern):
    def __init__(self, sqla_cls, subject_pattern=None,
                 name=None, nsm=None, *patterns):
        super(ClassQuadMapPattern, self).__init__(None, nsm)
        self.sqla_cls = sqla_cls
        mapper = sqla_cls.__mapper__
        info = mapper.mapped_table.info
        if 'rdf_subject_pattern' in info:
            subject_pattern = subject_pattern or info['rdf_subject_pattern']
        subject_pattern.resolve(sqla_cls)
        self.subject_pattern = subject_pattern
        patterns = list(patterns)
        if 'rdf_patterns' in info:
            patterns.extend(info['rdf_patterns'])
        self.patterns = patterns
        for p in patterns:
            p.resolve(sqla_cls)

        for c in mapper.columns:
            if 'rdf' in c.info:
                qmp = c.info['rdf']
                qmp.set_columns(c)
                self.patterns.append(qmp)

    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        return self.subject_pattern.virt_def(nsm, engine) + ' ;\n'.join(
            qmp.virt_def(nsm, engine) for qmp in self.patterns) + ' .\n'

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

    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        inner = ''.join((qmp.virt_def(nsm, engine) for qmp in self.qmps))
        stmt = 'graph %s %s {\n%s\n}' % (
            self.iri.n3(nsm),
            'option(%s)' % (self.option) if self.option else '',
            inner)
        if self.name:
            stmt = 'create %s as %s . ' % (self.name.n3(nsm), stmt)
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

    def virt_def(self, nsm=None, engine=None):
        nsm = nsm or self.nsm
        native = '\n'.join(gqm.virt_def(nsm, engine) for gqm in self.native_graphmaps)
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
