import re
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
from itertools import groupby
from types import StringTypes

from sqlalchemy.types import TypeEngine
from sqlalchemy.sql.expression import ClauseElement
from sqlalchemy.orm.attributes import InstrumentedAttribute
from rdflib import Namespace, RDF

VirtRDF = Namespace('http://www.openlinksw.com/schemas/virtrdf#')


class Mapping(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractproperty
    def mapping_name(self):
        pass

    def drop(self, nsm):
        return "drop %s %s ." % (
            self.mapping_name, self.name.n3(nsm))

    def patterns_iter(self):
        return ()

    @abstractmethod
    def virt_def(self, nsm, engine=None):
        pass

    def definition_statement(self, nsm, engine=None):
        prefixes = "\n".join("PREFIX %s: %s " % (
            p, ns.n3()) for (p, ns) in nsm.namespaces()) if nsm else ''
        patterns = set(self.patterns_iter())
        patterns = '\n'.join((p.virt_def(nsm, engine) for p in patterns))
        return '%s\n%s\n%s\n' % (
            prefixes, patterns, self.virt_def(nsm, engine))

    @staticmethod
    def resolve_argument(arg, classes):
        if isinstance(arg, (InstrumentedAttribute, ClauseElement)):
            return arg
        if isinstance(classes, (list, tuple)):
            classes = {cls.name: cls for cls in classes}
        if isinstance(arg, StringTypes):
            if '.' in arg:
                cls, arg = arg.split('.', 1)
                if cls not in classes:
                    raise ArgumentError("Please provide class: "+cls)
                arg = getattr(cls, arg, None)
                if arg is None:
                    raise ArgumentError(
                        "Class <{0}> does not have a column <{1}> ".format(
                            cls, arg))
                if not isinstance(arg, InstrumentedAttribute):
                    raise ArgumentError(
                        "{0}.{1} is not a column".format(cls, arg))
                return arg
            included = [cls for cls in classes.itervalues()
                        if getattr(cls, arg, None)]
            if not len(included):
                raise ArgumentError(
                    "Argument <{0}> not found in provided classes.".format(
                        arg))
            if len(included) > 1:
                raise ArgumentError(
                    "Argument <{0}> found in many classes: {1}.".format(
                        arg, ','.join(cls.__name__ for cls in included)))
            return getattr(included[0], arg)

    @staticmethod
    def format_arg(arg, nsm, engine=None):
        if getattr(arg, 'n3', None) is not None:
            return arg.n3(nsm)
        elif getattr(arg, 'compile', None) is not None:
            return unicode(arg.compile(engine))
        elif isinstance(arg, Mapping):
            return arg.virt_def(nsm, engine)
        raise ArgumentError()



class ApplyFunction(Mapping):
    def __init__(self, fndef, *arguments):
        super(ApplyFunction, self).__init__(None)
        self.fndef = fndef
        self.arguments = tuple(arguments)

    def resolve(self, *classes):
        self.arguments = tuple((
            self.resolve_argument(arg, classes) for arg in self.arguments))

    def virt_def(self, nsm, engine=None):
        return "%s (%s) " % (
            self.fndef.name.n3(nsm), ', '.join([
                self.format_arg(arg, nsm, engine) for arg in self.arguments]))

    @property
    def mapping_name(self):
        return None

    def patterns_iter(self):
        for pat in self.fndef.patterns_iter():
            yield pat

    def set_arguments(self, *arguments):
        self.arguments = arguments

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.fndef == other.fndef and \
            self.arguments == other.arguments


class VirtuosoAbstractFunction(Mapping):
    __metaclass__ = ABCMeta

    def __init__(self, name, *arguments):
        super(VirtuosoAbstractFunction, self).__init__(name)
        self.arguments = tuple(arguments)

    def patterns_iter(self):
        yield self

    def apply(self, *arguments):
        return ApplyFunction(self, *arguments)


class IriClass(VirtuosoAbstractFunction):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        super(IriClass, self).__init__(name)

    @property
    def mapping_name(self):
        return "iri class"

    def virt_def(self, nsm, engine=None):
        return ''


class PatternIriClass(IriClass):
    #parse_pattern = re.compile(r'(%(?:\{\w+\})?[dsU])')
    parse_pattern = re.compile(r'(%[dsU])')

    def __init__(self, name, pattern, *args):
        """args must be triples of (name, sql type, and nullable(bool))
        sql type must be a sqlalchemy type or sqlalchemy type instance
        """
        super(PatternIriClass, self).__init__(name)
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

    def virt_def(self, nsm, engine=None):
        dialect = engine.dialect if engine else None
        return 'create %s %s "%s" (%s) . ' % (
            self.mapping_name, self.name.n3(nsm), self.pattern,
            ','.join(["in %s %s %s" % (
                vname, vtype.compile(dialect),
                '' if self.nullable[vname] else 'not null')
                for vname, vtype in self.vars.items()]))

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

    def __init__(self, subject=None, predicate=None, obj=None,
                 graph=None, name=None, storage=None):
        super(QuadMapPattern, self).__init__(name)
        self.storage = storage
        self.graph = graph
        self.subject = subject
        self.predicate = predicate
        self.object = obj

    @property
    def mapping_name(self):
        return "quad map"

    def import_stmt(self, storage_name, nsm):
        assert self.name
        return "create %s using storage %s . " % (
            self.name.n3(nsm), storage_name.n3(nsm))

    def resolve(self, *classes):
        if isinstance(self.subject, ApplyFunction):
            self.subject.resolve(*classes)
        if isinstance(self.object, ApplyFunction):
            self.object.resolve(*classes)

    def set_defaults(self, subject=None, obj=None, graph=None, storage=None, name=None):
        self.storage = self.storage or storage
        self.subject = self.subject or subject
        self.name = self.name or name
        if self.object is not None:
            if isinstance(self.object, ApplyFunction):
                self.object.set_arguments(obj)
        else:
            self.object = obj
        self.graph = self.graph or graph

    def virt_def(self, nsm, engine=None):
        stmt = "%s %s" % (
            self.format_arg(self.predicate, nsm), self.format_arg(self.object, nsm, engine))
        if self.name:
            stmt += "\n    as %s " % (self.name.n3(nsm),)
        return stmt

    def patterns_iter(self):
        if isinstance(self.subject, Mapping):
            for p in self.subject.patterns_iter():
                yield p
        if isinstance(self.object, Mapping):
            for p in self.object.patterns_iter():
                yield p

class ClassPatternExtractor(object):
    def __init__(self, graph=None, storage=None):
        self.graph = graph
        self.storage = storage

    def extract_subject_pattern(self, sqla_cls):
        try:
            mapper = sqla_cls.__mapper__
            info = mapper.local_table.info
            return info.get('rdf_subject_pattern', None)
        except AttributeError:
            return None

    def make_column_name(self, cls, column):
        pass

    def extract_column_info(self, sqla_cls, subject_pattern):
        mapper = sqla_cls.__mapper__
        info = mapper.local_table.info
        for c in mapper.columns:
            if 'rdf' in c.info:
                qmp = c.info['rdf']
                if isinstance(qmp, QuadMapPattern):
                    qmp.set_defaults(subject_pattern, c, self.graph, self.storage,
                        self.make_column_name(sqla_cls, c))
                    if qmp.graph == self.graph and qmp.storage == self.storage:
                        qmp.resolve(sqla_cls)
                        yield qmp

    def extract_info(self, sqla_cls, subject_pattern=None):
        subject_pattern = subject_pattern or \
            self.extract_subject_pattern(sqla_cls)
        if not subject_pattern:
            return
        subject_pattern.resolve(sqla_cls)
        mapper = sqla_cls.__mapper__
        info = mapper.local_table.info
        for c in self.extract_column_info(sqla_cls, subject_pattern):
            yield c


class GraphQuadMapPattern(Mapping):
    def __init__(self, graph_iri, name=None, option=None, *qmps):
        super(GraphQuadMapPattern, self).__init__(name)
        self.iri = graph_iri
        self.qmps = list(qmps)
        self.option = option

    def virt_def(self, nsm, engine=None):
        inner = '.\n'.join([
            self.format_arg(subject, nsm, engine) + ' ' + ';\n'.join([
                    pattern.virt_def(nsm, engine) for pattern in sgroups
                ]) for subject, sgroups in groupby(self.qmps, lambda p: p.subject)
        ])
        stmt = 'graph %s%s {\n%s.\n}' % (
            self.iri.n3(nsm),
            ' option(%s)' % (self.option) if self.option else '',
            inner)
        if self.name:
            stmt = 'create %s as %s.' % (self.name.n3(nsm), stmt)
        return stmt

    def patterns_iter(self):
        for qmp in self.qmps:
            for pat in qmp.patterns_iter():
                yield pat

    @property
    def mapping_name(self):
        return None

    def import_stmt(self, storage_name, nsm):
        assert self.name and self.storage and self.storage.name
        return " create %s using storage %s . " % (
            self.name.n3(nsm), storage_name.n3(nsm))

    def add_patterns(self, patterns):
        for pattern in patterns:
            assert isinstance(pattern, QuadMapPattern)
            self.qmps.append(pattern)


class QuadStorage(Mapping):
    def __init__(self, name, native_graphmaps, imported_graphmaps=None,
                 add_default=True):
        super(QuadStorage, self).__init__(name)
        self.native_graphmaps = native_graphmaps
        self.imported_graphmaps = imported_graphmaps or []
        self.add_default = add_default
        for gmap in native_graphmaps:
            gmap.storage = self

    @property
    def mapping_name(self):
        return "quad storage"

    def virt_def(self, nsm, engine=None):
        native = '\n'.join(gqm.virt_def(nsm, engine) for gqm in self.native_graphmaps)
        imported = '\n'.join(gqm.import_stmt(self.name, nsm)
                             for gqm in self.imported_graphmaps)
        if self.add_default:
            imported += '.' + DefaultQuadMap.import_stmt(self.name, nsm)
        return 'create %s %s {\n %s \n}' % (
            self.mapping_name, self.name.n3(nsm),
            '\n'.join((native, imported)))

    def add_imported(self, qmap, nsm):
        return 'alter %s %s {\n %s \n}' % (
            self.mapping_name, self.name.n3(nsm), qmap.import_stmt(self.name, nsm))

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
