import re

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, defaultdict
from itertools import chain
from types import StringTypes

from sqlalchemy import create_engine
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.util import ORMAdapter
from sqlalchemy.schema import Column
from sqlalchemy.sql.expression import ClauseElement
from sqlalchemy.sql.selectable import Alias
from sqlalchemy.sql.visitors import ClauseVisitor, Visitable
from sqlalchemy.types import TypeEngine
from sqlalchemy.util import memoized_property

from rdflib import Namespace, URIRef, Graph
from rdflib.namespace import NamespaceManager
from rdflib.term import Identifier

VirtRDF = Namespace('http://www.openlinksw.com/schemas/virtrdf#')


class Mapping(ClauseElement):
    def __init__(self, name, nsm=None):
        self.name = name
        self.nsm = nsm

    #@abstractproperty
    def mapping_name(self):
        raise NotImplemented()

    def drop(self, session, force=False):
        errors = []
        # first drop submaps I know about
        for submap in self.known_submaps():
            errors.extend(submap.drop(session, force))
        # then drop submaps I don't know about
        for submap in self.effective_submaps(session):
            errors.extend(submap.drop(session, force))
        # It may have been ineffective. Abort.
        remaining = self.effective_submaps(session)
        remaining = list(remaining)
        if remaining:
            errors.append("Remaining in %s: " % (self.representation(),)
                + ', '.join((x.representation() for x in remaining)))
            if not force:
                return errors
        stmt = self.drop_statement()
        if stmt:
            print stmt
            errors.extend(session.execute(
                'sparql ' + self.prefixes() + "\n" + stmt))
        return errors

    def drop_statement(self):
        if not self.name:
            return ''
        return "drop %s %s ." % (
            self.mapping_name, self.name.n3(self.nsm))

    def known_submaps(self):
        return ()

    def effective_submaps(self, session):
        return ()

    def patterns_iter(self):
        return ()

    #@abstractmethod
    def virt_def(self, alias_set, engine=None):
        raise NotImplemented()

    def prefixes(self):
        assert self.nsm
        return "\n".join("PREFIX %s: %s " % (
            p, ns.n3()) for (p, ns) in self.nsm.namespaces())

    def definition_statement(self, alias_manager=None, engine=None):
        prefixes = self.prefixes()
        patterns = set(self.patterns_iter())
        patterns = '\n'.join((p.virt_def(None, engine)
                              for p in patterns))
        return '%s\n%s\n%s\n' % (
            prefixes, patterns, self.virt_def(None, engine))

    @staticmethod
    def resolve_argument(arg, classes):
        if isinstance(arg, (InstrumentedAttribute, ClauseElement)):
            return arg
        if isinstance(classes, (list, tuple)):
            classes = {cls.__name__: cls for cls in classes}
        if isinstance(arg, StringTypes):
            if '.' in arg:
                cls, arg = arg.split('.', 1)
                if cls not in classes:
                    raise ValueError("Please provide class: "+cls)
                arg = getattr(cls, arg, None)
                if arg is None:
                    raise AttributeError(
                        "Class <{0}> does not have a column <{1}> ".format(
                            cls, arg))
                if not isinstance(arg, InstrumentedAttribute):
                    raise TypeError(
                        "{0}.{1} is not a column".format(cls, arg))
                return arg
            included = [cls for cls in classes.itervalues()
                        if getattr(cls, arg, None)]
            if not len(included):
                raise AttributeError(
                    "Argument <{0}> not found in provided classes.".format(
                        arg))
            if len(included) > 1:
                raise ValueError(
                    "Argument <{0}> found in many classes: {1}.".format(
                        arg, ','.join(cls.__name__ for cls in included)))
            return getattr(included[0], arg)

    def format_arg(self, arg, alias_set, engine=None):
        if isinstance(arg, Mapping):
            return arg.virt_def(alias_set, engine)
        elif getattr(arg, 'n3', None) is not None:
            assert self.nsm
            return arg.n3(self.nsm)
        elif getattr(arg, 'compile', None) is not None:
            return str(alias_set.aliased_term(arg).compile(engine))
        elif isinstance(arg, (str, unicode, int)):
            return unicode(arg)
        raise TypeError()

    def representation(self):
        if self.name:
            return "<%s %s>" % (
                self.__class__.__name__, self.name.n3(self.nsm))
        else:
            return "<%s <?>>" % (self.__class__.__name__, )

    def set_namespace_manager(self, nsm):
        self.nsm = nsm

    def __repr__(self):
        return self.representation()


class ApplyFunction(Mapping):
    def __init__(self, fndef, nsm=None, *arguments):
        super(ApplyFunction, self).__init__(None, nsm)
        self.fndef = fndef
        self.arguments = tuple(arguments)

    def resolve(self, *classes):
        self.arguments = tuple((
            self.resolve_argument(arg, classes) for arg in self.arguments))

    def virt_def(self, alias_set, engine=None):
        assert self.nsm
        return "%s (%s) " % (
            self.fndef.name.n3(self.nsm), ', '.join([
                self.format_arg(arg, alias_set, engine)
                for arg in self.arguments]))

    @property
    def mapping_name(self):
        raise NotImplemented()

    def patterns_iter(self):
        for pat in self.fndef.patterns_iter():
            yield pat

    def set_arguments(self, *arguments):
        self.arguments = tuple(arguments)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.fndef == other.fndef and \
            self.arguments == other.arguments

    def set_namespace_manager(self, nsm):
        super(ApplyFunction, self).set_namespace_manager(nsm)
        self.fndef.set_namespace_manager(nsm)
        for arg in self.arguments:
            if isinstance(arg, Mapping):
                arg.set_namespace_manager(nsm)


class VirtuosoAbstractFunction(Mapping):
    def __init__(self, name, nsm=None, *arguments):
        super(VirtuosoAbstractFunction, self).__init__(name, nsm)
        self.arguments = tuple(arguments)

    def patterns_iter(self):
        yield self

    def apply(self, *arguments):
        return ApplyFunction(self, self.nsm, *arguments)


class IriClass(VirtuosoAbstractFunction):
    def __init__(self, name, nsm=None):
        super(IriClass, self).__init__(name, nsm)

    @property
    def mapping_name(self):
        return "iri class"

    def virt_def(self, alias_set, engine=None):
        assert self.nsm
        return ''


class PatternIriClass(IriClass):
    #parse_pattern = re.compile(r'(%(?:\{\w+\})?[dsU])')
    parse_pattern = re.compile(r'(%[dsU])')

    def __init__(self, name, pattern, nsm=None, *args):
        """args must be triples of (name, sql type, and nullable(bool))
        sql type must be a sqlalchemy type or sqlalchemy type instance
        """
        super(PatternIriClass, self).__init__(name, nsm=nsm)
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
            raise ValueError()
        return self.pattern % args

    def parse(self, iri):
        r = self.pattern_matcher.match(iri)
        assert r, "The iri does not match " + self.pattern
        assert len(r.group(0)) == len(iri),\
            "The iri does not match " + self.pattern
        vals = [int(v) if self.is_int[p] else v
                for p, v in enumerate(r.groups())]
        return dict(zip(self.varnames, vals))

    def virt_def(self, alias_set, engine=None):
        assert self.nsm
        dialect = engine.dialect if engine else None
        return 'create %s %s "%s" (%s) . ' % (
            self.mapping_name, self.name.n3(self.nsm), self.pattern,
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

    def effective_submaps(self, session):
        assert self.nsm
        if not self.name:
            return
        prefixes = self.prefixes()
        res = list(session.execute("""SPARQL %s\n SELECT ?map ?submap WHERE {graph virtrdf: {
            ?map virtrdf:qmSubjectMap ?qmsm . ?qmsm virtrdf:qmvIriClass %s
            OPTIONAL {?map virtrdf:qmUserSubMaps ?o . ?o ?p ?submap . ?submap a virtrdf:QuadMap}
            }}""" % (prefixes, self.name.n3(self.nsm))))
        # Or is it ?qmsm virtrdf:qmvFormat ? Seems broader
        for (mapname, submapname) in res:
            if submapname:
                yield GraphQuadMapPattern(None, None, name=URIRef(mapname), nsm=self.nsm)
            else:
                yield QuadMapPattern(name=URIRef(mapname), nsm=self.nsm)


class ClauseEqWrapper(object):
    __slots__ = ('clause',)

    def __init__(self, clause):
        self.clause = clause

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            return False
        return self.clause.compare(other.clause)

    def __ne__(self, other):
        return not self.__eq__(other)


class QuadMapPattern(Mapping):
    def __init__(self, subject=None, predicate=None, obj=None,
                 graph_name=None, name=None, condition=None, nsm=None):
        super(QuadMapPattern, self).__init__(name, nsm)
        self.graph_name = graph_name
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.condition = condition
        self.conditionc_set = set()
        if condition is not None:
            self.conditionc_set.add(_sig(condition))

    @property
    def mapping_name(self):
        return "quad map"

    def and_condition(self, condition):
        condition_c = _sig(condition)
        if self.condition is None:
            self.condition = condition
        elif condition_c not in self.conditionc_set:
            self.condition = self.condition & condition
        self.conditionc_set.add(condition_c)

    def and_conditions(self, conditions):
        for condition in conditions:
            self.and_condition(condition)

    def import_stmt(self, storage_name):
        assert self.name
        assert self.nsm
        return "create %s using storage %s . " % (
            self.name.n3(self.nsm), storage_name.n3(self.nsm))

    def resolve(self, *classes):
        if isinstance(self.subject, ApplyFunction):
            self.subject.resolve(*classes)
        if isinstance(self.object, ApplyFunction):
            self.object.resolve(*classes)

    def set_defaults(self, subject=None, obj=None, graph_name=None,
                     name=None, condition=None):
        self.subject = subject if self.subject is None else self.subject
        self.name = self.name or name
        if self.condition is None:
            self.condition = condition

        if obj is not None:
            if self.object is not None:
                if isinstance(self.object, ApplyFunction):
                    self.object.set_arguments(obj)
            else:
                self.object = obj
        self.graph_name = self.graph_name or graph_name

    def aliased_classes(self, alias_set, as_alias=True):
        v = GatherColumnsVisitor(alias_set.mgr.class_reg)

        def add_term(t):
            if isinstance(t, ApplyFunction):
                for sub in t.arguments:
                    add_term(sub)
            elif isinstance(t, Visitable):
                v.traverse(t)
            elif isinstance(t, (Column, InstrumentedAttribute)):
                v.columns.add(t)
        for term in self.terms():
            add_term(term)
        classes = v.get_classes()
        if as_alias:
            alias_of = {inspect(alias)._target: alias for alias in alias_set.aliases}
            classes = {alias_of[cls] for cls in classes}
        return classes

    def missing_aliases(self, alias_set):
        print "unaliasing: alias_set", [(BaseAliasSet.alias_name(alias), alias) for alias in alias_set.aliases]
        term_aliases = self.aliased_classes(alias_set)
        print "            term_aliases", [(BaseAliasSet.alias_name(alias), alias) for alias in term_aliases]
        missing_aliases = set(alias_set.aliases) - term_aliases
        print "            missing_aliases", [(BaseAliasSet.alias_name(alias), alias) for alias in missing_aliases]
        return missing_aliases

    def virt_def(self, alias_set, engine=None):
        assert self.nsm
        missing_aliases = self.missing_aliases(alias_set)
        options = ''
        if missing_aliases:
            options = ' option(%s)' % ', '.join((
                'using '+BaseAliasSet.alias_name(alias)
                for alias in missing_aliases))
        stmt = "%s %s%s" % (
            self.format_arg(self.predicate, alias_set),
            self.format_arg(self.object, alias_set, engine),
            options)
        if self.name:
            stmt += "\n    as %s " % (self.name.n3(self.nsm),)
        return stmt

    def terms(self):
        return (self.subject, self.predicate, self.object, self.graph_name)

    def patterns_iter(self):
        if isinstance(self.subject, Mapping):
            for p in self.subject.patterns_iter():
                yield p
        if isinstance(self.object, Mapping):
            for p in self.object.patterns_iter():
                yield p

    def set_namespace_manager(self, nsm):
        super(QuadMapPattern, self).set_namespace_manager(nsm)
        for term in self.terms():
            if isinstance(term, Mapping):
                term.set_namespace_manager(nsm)


class DebugClauseVisitor(ClauseVisitor):
    def visit_binary(self, binary):
        print "visit_binary", repr(binary)

    def visit_column(self, column):
        print "visit_column", repr(column)

    def visit_bindparam(self, bind):
        print "visit_bindparam", repr(bind)


def _get_column_class(col, class_registry=None, use_annotations=True):
    col = inspect(col)
    cls = getattr(col, 'class_', None)
    if cls:
        return cls
    if use_annotations:
        ann = getattr(col, '_annotations', None)
        if ann:
            mapper = getattr(ann, 'parententity', getattr(ann, 'parentmapper', None))
            if mapper:
                cls = getattr(mapper, 'class_', None)
                if cls:
                    return cls
    if class_registry:
        table = col.table
        if isinstance(table, Alias):
            #import pdb; pdb.set_trace()
            table = table.original
        for cls in class_registry.itervalues():
            if isinstance(cls, type) and inspect(cls).local_table == table:
                # return highest such class.
                for supercls in cls.mro():
                    if not getattr(supercls, '__mapper__', None):
                        return cls
                    if inspect(supercls).local_table != table:
                        return cls
                    cls = supercls
    assert False, "Cannot obtain the class from the column " + repr(col)


class GatherColumnsVisitor(ClauseVisitor):
    def __init__(self, class_reg=None):
        super(GatherColumnsVisitor, self).__init__()
        self.columns = set()
        self.class_reg = class_reg

    def visit_column(self, column):
        self.columns.add(column)

    def get_classes(self):
        return {_get_column_class(col, self.class_reg) for col in self.columns}


def _sig(condition):
    return unicode(condition.compile(
        compile_kwargs={"literal_binds": True}))


_camel2underscore_re = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
def camel2underscore(camel):
    return _camel2underscore_re.sub(r'_\1', camel).lower()


class BaseAliasSet(object):
    __metaclass__ = ABCMeta

    def __init__(self, mgr, id, term):
        self.mgr = mgr
        self.id = id
        self.term = term

    def __hash__(self):
        return hash(self.term)

    def __eq__(self, other):
        return (
            other.__class__ == self.__class__
            and hash(self) == hash(other)
            and _sig(self.term) == _sig(other.term))

    def _alias_name(self, cls):
        return "alias_%s_%s" % (camel2underscore(cls.__name__), self.id)

    def adapter(self):
        adapter = None
        for alias in self.aliases:
            adapter = ORMAdapter(alias).chain(adapter)
        return adapter

    @abstractproperty
    def aliases(self):
        pass

    def get_column_alias(self, column):
        if isinstance(column, Column):
            for alias in self.aliases:
                # TODO: What if there's many?
                if inspect(alias).mapper.local_table == column.table:
                    return getattr(alias, column.key)
        else:
            for alias in self.aliases:
                if inspect(alias).mapper.class_ == column.class_:
                    return getattr(alias, column.key)
        assert False, "column %s not in known aliases" % column

    def aliased_term(self, term=None):
        term = term if term is not None else self.term
        if isinstance(term, Visitable):
            return self.adapter().traverse(
                term if term is not None else self.term)
        elif isinstance(term, (Column, InstrumentedAttribute)):
            return self.get_column_alias(term)
        else:
            assert False, term

    @staticmethod
    def alias_name(alias):
        return inspect(alias).selectable.name

    def alias(self, cls):
        name = self._alias_name(cls)
        table = inspect(cls).local_table
        return aliased(cls, table.alias(name=name))

    def full_table_name(self, cls, engine):
        # There must be a better way...
        column = inspect(cls).local_table.columns.values()[0]
        return str(column.compile(engine)).rsplit('.', 1)[0]


class ClassAlias(BaseAliasSet):
    def __init__(self, mgr, column):
        super(ClassAlias, self).__init__(mgr, "0", column)

    @memoized_property
    def aliases(self):
        return [self.alias(self.term)]

    def virt_def(self, engine=None):
        # TODO: This needs to be made a clause... or to generate them.
        return "FROM %s AS %s" % (
            self.full_table_name(self.term, engine),
            self._alias_name(self.term))

    def get_column_alias(self, column):
        if isinstance(column, Column):
            assert column.table == inspect(self.term).local_table
        else:
            assert column.class_ == self.term
        return getattr(self.aliases[0], column.key)


class ConditionAliasSet(BaseAliasSet):
    """A coherent set of class alias that are used in a condition's instance"""
    def __init__(self, mgr, id, condition):
        super(ConditionAliasSet, self).__init__(mgr, id, condition)
        self.extra_classes = set()

    def add_extra_class(self, cls):
        self.extra_classes.add(cls)

    @memoized_property
    def aliases(self):
        g = GatherColumnsVisitor(self.mgr.class_reg)
        g.traverse(self.term)
        classes = g.get_classes()
        for cls in self.extra_classes:
            assert isinstance(cls, type)
            classes.add(cls)
        return [self.alias(cls) for cls in classes]

    def virt_def(self, engine=None):
        # TODO: This needs to be made a clause... or to generate them.
        return "\n".join([
            "FROM %s AS %s" % (
                self.full_table_name(inspect(alias).mapper, engine),
                self.alias_name(alias))
            for alias in self.aliases])

    def where_clause(self, engine=None):
        condition = self.aliased_term()
        return "WHERE (%s)\n" % str(condition.compile(engine))


class ClassAliasManager(object):
    def __init__(self, class_reg=None):
        self.alias_sets = {}
        self.base_aliases = {}
        self.class_reg = class_reg

    def superclass_conditions(self, column):
        """Columns defined on superclass may come from another table.
        Here we calculate the necessary joins.
        Also class identity conditions for single-table inheritance
        """
        conditions = {}
        if isinstance(column, (int, str, unicode, Identifier)):
            return {}, column
        cls = _get_column_class(column, self.class_reg)
        condition = inspect(cls)._single_table_criterion
        if condition is not None:
            conditions[_sig(condition)] = condition
        local_keys = {c.key for c in inspect(cls).local_table.columns}
        if (getattr(cls, column.key, None) is not None
                and column.key not in local_keys):
            for sup in cls.mro()[1:]:
                condition = inspect(cls).inherit_condition
                conditions[_sig(condition)] = condition
                cls = sup
                local_keys = {c.key for c in inspect(cls).local_table.columns}
                if column.key in local_keys:
                    column = getattr(cls, column.key)
                    break
            else:
                assert False, "The column is found in the "\
                    "class and not in superclasses?"
        return conditions, column

    def superclass_conditions_multiple(self, columns):
        conditions = {}
        newcols = []
        for column in columns:
            conds, newcol = self.superclass_conditions(column)
            conditions.update(conds)
            newcols.append(newcol)
        return conditions, newcols

    def add_quadmap(self, quadmap):
        conditions = {}
        all_args = []
        for term_index in ('subject', 'predicate', 'object', 'graph_name'):
            term = getattr(quadmap, term_index)
            if isinstance(term, ApplyFunction):
                tconditions, args = self.superclass_conditions_multiple(
                    term.arguments)
                conditions.update(tconditions)
                term.set_arguments(*args)
                # Another assumption
                all_args.extend(args)
            elif isinstance(term, (InstrumentedAttribute, Column)):
                tconditions, arg = self.superclass_conditions(term)
                conditions.update(tconditions)
                all_args.append(arg)
                setattr(quadmap, term_index, arg)
        term_classes = {
            _get_column_class(col, self.class_reg) for col in all_args}
        if quadmap.condition is not None:
            # in some cases, sqla transforms condition terms to
            # use the superclass. Lots of silly gymnastics to invert that.
            g = GatherColumnsVisitor(self.class_reg)
            g.traverse(quadmap.condition)
            for col in g.columns:
                sub = _get_column_class(col, self.class_reg)
                cls = _get_column_class(col, self.class_reg, False)
                if cls not in term_classes and sub in term_classes:
                    col = getattr(sub, col.key, None)
                    tconditions, arg = self.superclass_conditions(col)
                    conditions.update(tconditions)
                    # TODO: Can I change terms in the condition?
                    # I think taken care of downstream
        quadmap.and_conditions(conditions.values())
        for arg in all_args:
            if isinstance(arg, (Column, InstrumentedAttribute)):
                self.add_class(arg, quadmap.condition)

    def get_alias_set(self, quadmap):
        # TODO: Horrible!
        # Maybe quadmap should have ref class?
        if quadmap.condition is not None:
            condition_c = _sig(quadmap.condition)
            return self.alias_sets[condition_c]
        else:
            subject = quadmap.subject
            # TODO: Abstract those assumptions
            assert isinstance(subject, ApplyFunction)
            id_column = subject.arguments[0]
            cls = _get_column_class(id_column, self.class_reg)
            return self.base_aliases[cls]

    def add_class(self, column_or_class, condition=None):
        if isinstance(column_or_class, type):
            cls = column_or_class
        elif isinstance(column_or_class, (Column, InstrumentedAttribute)):
            cls = _get_column_class(column_or_class, self.class_reg)
        else:
            assert False
        if condition is not None:
            id = str(len(self.alias_sets) + 1)
            condition_c = _sig(condition)
            cas = self.alias_sets.setdefault(
                condition_c, ConditionAliasSet(self, id, condition))
            cas.add_extra_class(cls)
        else:
            self.base_aliases[cls] = ClassAlias(self, cls)

    def remove_class(self, cls, condition=None):
        if condition is None:
            del self.base_aliases[cls]
        else:
            pass  # TODO

    def virt_def(self, engine=None):
        # TODO: This needs to be made a clause... or to generate them.
        from_clauses = "\n".join(
            [ca.virt_def(engine)
             for ca in chain(self.alias_sets.itervalues(),
                             self.base_aliases.itervalues())])
        alias_engine = create_engine('virtuoso_alias://')
        where_clauses = "".join([
            ca.where_clause(alias_engine)
            for ca in self.alias_sets.itervalues()])
        return from_clauses + '\n' + where_clauses

    def get_column_alias(self, column, condition=None):
        if condition is not None:
            alias_set = self.alias_sets[_sig(condition)]
        else:
            cls = _get_column_class(column, self.class_reg)
            if cls not in self.base_aliases:
                self.base_aliases[cls] = ClassAlias(self, cls)
            alias_set = self.base_aliases[cls]
        return alias_set.get_column_alias(column)


class ClassPatternExtractor(object):
    def __init__(self, alias_manager, graph=None):
        self.graph = graph
        self.alias_manager = alias_manager

    def extract_subject_pattern(self, sqla_cls):
        try:
            mapper = inspect(sqla_cls)
            info = mapper.local_table.info
            return info.get('rdf_subject_pattern', None)
        except NoInspectionAvailable as err:
            return None

    def make_column_name(self, cls, column):
        pass

    def set_defaults(self, qmp, subject_pattern, sqla_cls, column=None):
        name = self.make_column_name(sqla_cls, column) if (
            column is not None) else None
        qmp.set_defaults(subject_pattern, column, self.graph.name, name)

    def extract_column_info(self, sqla_cls, subject_pattern):
        mapper = inspect(sqla_cls)
        info = mapper.local_table.info
        for c in mapper.columns:
            # Local columns only to avoid duplication
            if c.table != mapper.local_table:
                continue
            if 'rdf' in c.info:
                qmp = c.info['rdf']
                if isinstance(qmp, QuadMapPattern):
                    self.set_defaults(qmp, subject_pattern, sqla_cls, c)
                    if qmp.graph_name == self.graph.name:
                        qmp.resolve(sqla_cls)
                        yield qmp

    def extract_info(self, sqla_cls, subject_pattern=None):
        subject_pattern = subject_pattern or \
            self.extract_subject_pattern(sqla_cls)
        if subject_pattern is None:
            return
        subject_pattern.resolve(sqla_cls)
        col = subject_pattern.arguments[0]
        assert isinstance(col, (InstrumentedAttribute, Column))
        condition = inspect(sqla_cls)._single_table_criterion
        self.alias_manager.add_class(col, condition)
        found = False
        for c in self.extract_column_info(sqla_cls, subject_pattern):
            found = True
            yield c
        if not found:
            self.alias_manager.remove_class(sqla_cls, condition)


class GraphQuadMapPattern(Mapping):
    def __init__(self, graph_iri, storage, name=None, option=None, nsm=None, *qmps):
        super(GraphQuadMapPattern, self).__init__(name, nsm)
        self.iri = graph_iri
        self.qmps = list(qmps)
        self.option = option
        self.storage = storage

    def graph_name_def(self, engine=None):
        assert self.nsm
        return self.iri.n3(self.nsm)

    def known_submaps(self):
        return self.qmps

    def effective_submaps(self, session):
        assert self.nsm
        if not self.name:
            return
        prefixes = self.prefixes()
        res = list(session.execute("""SPARQL %s SELECT ?map WHERE {graph virtrdf: {
            %s virtrdf:qmUserSubMaps ?o . ?o ?p ?map . ?map a virtrdf:QuadMap
            }}""" % (prefixes, self.name.n3(self.nsm))))
        for (mapname, ) in res:
            yield QuadMapPattern(name=URIRef(mapname), graph_name=self.name, nsm=self.nsm)

    def virt_def(self, alias_manager, engine=None):
        assert self.nsm
        arguments = defaultdict(list)
        for qmp in self.qmps:
            alias_set = alias_manager.get_alias_set(qmp)
            subject = self.format_arg(qmp.subject, alias_set, engine)
            arguments[subject].append(qmp.virt_def(alias_set, engine))
        inner = '.\n'.join((
            subject + ' ' + ';\n'.join(args)
            for subject, args in arguments.iteritems()))
        stmt = 'graph %s%s {\n%s . \n}' % (
            self.graph_name_def(engine),
            ' option(%s)' % (self.option) if self.option else '',
            inner)
        if self.name:
            print stmt
            stmt = 'create %s as %s ' % (self.name.n3(self.nsm), stmt)
        return stmt

    def patterns_iter(self):
        for qmp in self.qmps:
            for pat in qmp.patterns_iter():
                yield pat

    @property
    def mapping_name(self):
        return "quad map"

    def import_stmt(self, storage_name):
        assert (self.name and self.storage is not None 
                and self.storage.name and self.nsm)
        return " create %s using storage %s . " % (
            self.name.n3(self.nsm), storage_name.n3(self.nsm))

    def add_patterns(self, patterns):
        assert self.nsm
        for pattern in patterns:
            assert isinstance(pattern, QuadMapPattern)
            pattern.set_namespace_manager(self.nsm)
            self.qmps.append(pattern)


class PatternGraphQuadMapPattern(GraphQuadMapPattern):
    "Reprensents a graph where the graph name is an IRI. Not functional."
    def __init__(self, graph_iri_pattern, storage, alias_set,
                 name=None, option=None, nsm=None, *qmps):
        super(PatternGraphQuadMapPattern, self).__init__(
            graph_iri_pattern, storage, name, option, nsm, *qmps)
        self.alias_set = alias_set

    def graph_name_def(self, engine=None):
        return self.iri.virt_def(self.alias_set, engine)


class QuadStorage(Mapping):
    def __init__(self, name, imported_graphmaps=None,
                 alias_manager=None, add_default=True, nsm=None):
        super(QuadStorage, self).__init__(name, nsm)
        self.alias_manager = alias_manager or ClassAliasManager()
        self.native_graphmaps = []
        self.imported_graphmaps = imported_graphmaps or []
        self.add_default = add_default

    @property
    def mapping_name(self):
        return "quad storage"

    def known_submaps(self):
        return self.native_graphmaps

    def effective_submaps(self, session):
        assert self.nsm
        if not self.name:
            return
        prefixes = self.prefixes()
        res = list(session.execute("""SPARQL %s SELECT ?map WHERE {graph virtrdf: {
            %s virtrdf:qsUserMaps ?o . ?o ?p ?map . ?map a virtrdf:QuadMap
            }}""" % (prefixes, self.name.n3(self.nsm))))
        for (mapname, ) in res:
            yield GraphQuadMapPattern(None, self, name=URIRef(mapname), nsm=self.nsm)

    def virt_def(self, alias_manager=None, engine=None):
        assert self.nsm
        alias_manager = alias_manager or self.alias_manager
        # TODO: Make sure this is only done once.
        for gqm in self.native_graphmaps:
            for qmp in gqm.qmps:
                alias_manager.add_quadmap(qmp)
        stmt = '.\n'.join(gqm.virt_def(alias_manager, engine)
                          for gqm in self.native_graphmaps)
        if self.imported_graphmaps:
            stmt += '.\n' + '.\n'.join(gqm.import_stmt(self.name)
                                       for gqm in self.imported_graphmaps)
        if self.add_default:
            stmt += '.\n' + DefaultQuadMap.import_stmt(self.name)
        return 'create %s %s \n%s{\n %s \n}' % (
            self.mapping_name, self.name.n3(self.nsm),
            alias_manager.virt_def(engine), stmt)

    def add_imported(self, qmap, alias_manager, engine=None):
        return 'alter %s %s \n%s\n{\n %s \n}' % (
            self.mapping_name, self.name.n3(self.nsm),
            alias_manager.virt_def(engine),
            qmap.import_stmt(self.name))

    def alter_native(self, gqm, alias_manager, engine=None):
        assert self.nsm
        prefixes = self.prefixes()
        patterns = set(gqm.patterns_iter())
        patterns = '\n'.join((p.virt_def(None, engine)
                              for p in patterns))
        # TODO: Make sure this is only done once.
        for qmp in gqm.qmps:
            alias_manager.add_quadmap(qmp)
        return '%s\n%s\nalter %s %s \n%s\n{\n %s \n}' % (
            prefixes, patterns,
            self.mapping_name, self.name.n3(self.nsm),
            alias_manager.virt_def(engine),
            gqm.virt_def(alias_manager, engine))

    def patterns_iter(self):
        for qmp in self.native_graphmaps:
            for pat in qmp.patterns_iter():
                yield pat
        for qmp in self.imported_graphmaps:
            for pat in qmp.patterns_iter():
                yield pat

    def add_graphmap(self, graphmap):
        self.native_graphmaps.append(graphmap)
        graphmap.set_namespace_manager(self.nsm)

DefaultNSM = NamespaceManager(Graph())
DefaultNSM.bind('virtrdf', VirtRDF)
DefaultQuadStorage = QuadStorage(VirtRDF.DefaultQuadStorage, add_default=False, nsm=DefaultNSM)
DefaultQuadMap = GraphQuadMapPattern(
    None, DefaultQuadStorage, VirtRDF.DefaultQuadMap)
DefaultQuadStorage.add_graphmap(DefaultQuadMap)
