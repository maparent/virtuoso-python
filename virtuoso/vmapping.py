import re

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, defaultdict
from itertools import chain
from types import StringTypes
from inspect import isabstract

from sqlalchemy import create_engine
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.util import ORMAdapter
from sqlalchemy.schema import Column
from sqlalchemy.sql.expression import (
    ClauseElement, Executable, FunctionElement)
from sqlalchemy.sql.selectable import Alias
from sqlalchemy.sql.visitors import ClauseVisitor, Visitable
from sqlalchemy.types import TypeEngine
from sqlalchemy.util import memoized_property

from rdflib import Namespace, URIRef, Graph
from rdflib.namespace import NamespaceManager
from rdflib.term import Identifier

from .vstore import VirtuosoNamespaceManager

VirtRDF = Namespace('http://www.openlinksw.com/schemas/virtrdf#')


class SparqlStatement(ClauseElement):
    def __init__(self, nsm):
        self.nsm = nsm

    def __bool__(self):
        # Avoid many tedious "is not None"
        return True


class SparqlMappingStatement(SparqlStatement):
    def __init__(self, mapping):
        super(SparqlMappingStatement, self).__init__(mapping.nsm)
        self.mapping = mapping

    def as_clause(self, arg):
        return self.mapping.as_clause(arg)


class CompoundSparqlStatement(SparqlMappingStatement):
    def __init__(self, clauses):
        super(CompoundSparqlStatement, self).__init__(clauses[0])
        self.clauses = clauses
        self.supports_execution = all((c.supports_execution for c in clauses))
        if self.supports_execution:
            self._execution_options = clauses[0]._execution_options

    def _compile(self, compiler, **kwargs):
        return "\n".join((compiler.process(clause, **kwargs)
                          for clause in self.clauses))

compiles(CompoundSparqlStatement)(CompoundSparqlStatement._compile)


class WrapSparqlStatement(SparqlStatement):
    def __init__(self, statement):
        super(WrapSparqlStatement, self).__init__(statement.nsm)
        self.statement = statement
        self.supports_execution = statement.supports_execution
        if self.supports_execution:
            self._execution_options = statement._execution_options

    def _compile(self, compiler, **kwargs):
        known_prefixes = set()
        if isinstance(self.nsm, VirtuosoNamespaceManager):
            known_prefixes = set(self.nsm.v_prefixes.values())
        prefixes = [DeclarePrefixStmt(p, ns)
                    for (p, ns) in self.nsm.namespaces()
                    if Namespace(ns) not in known_prefixes]
        return "sparql %s\n%s" % (
            "\n".join((compiler.process(prefix, **kwargs)
                       for prefix in prefixes)),
            compiler.process(self.statement, **kwargs))

compiles(WrapSparqlStatement)(WrapSparqlStatement._compile)


class CreateIriClassStmt(Executable, SparqlMappingStatement):
    def _compile(self, compiler, **kwargs):
        kwargs['literal_binds'] = True
        mapping = self.mapping
        name = compiler.process(self.as_clause(mapping.name), **kwargs)
        args = ((compiler.preparer.quote(vname),
                 vtype.compile(compiler.dialect),
                 '' if mapping.nullable[vname] else 'NOT NULL')
                for vname, vtype in mapping.vars.items())
        return 'create %s %s "%s" (%s) . ' % (
            mapping.mapping_name, name, mapping.pattern,
            ','.join(("in %s %s %s" % argv for argv in args)))

compiles(CreateIriClassStmt)(CreateIriClassStmt._compile)


class DropMappingStmt(Executable, SparqlMappingStatement):
    def _compile(self, compiler, **kwargs):
        kwargs['literal_binds'] = True
        mapping = self.mapping
        name = compiler.process(
            self.as_clause(self.mapping.name), **kwargs)
        return "drop %s %s" % (mapping.mapping_name, name)

compiles(DropMappingStmt)(DropMappingStmt._compile)


class DeclarePrefixStmt(ClauseElement):
    def __init__(self, prefix, uri):
        super(DeclarePrefixStmt, self).__init__()
        self.prefix = prefix
        self.uri = uri

    def _compile(self, compiler, **kwargs):
        return "PREFIX %s: %s" % (
            self.prefix, self.uri.n3())

compiles(DeclarePrefixStmt)(DeclarePrefixStmt._compile)


class CreateQuadStorageStmt(Executable, SparqlMappingStatement):
    def __init__(self, mapping, graphs, alias_defs, imported_graphs=None):
        super(CreateQuadStorageStmt, self).__init__(mapping)
        self.graphs = graphs
        self.imported_graphs = imported_graphs or ()
        self.alias_defs = alias_defs

    def _compile(self, compiler, **kwargs):
        kwargs['literal_binds'] = True
        graphs = chain(self.graphs, self.imported_graphs)
        stmt = '.\n'.join((compiler.process(graph, **kwargs)
                           for graph in graphs))
        name = compiler.process(
            self.as_clause(self.mapping.name), **kwargs)
        alias_def = '\n'.join((compiler.process(alias_def, **kwargs)
                               for alias_def in self.alias_defs))
        return 'create quad storage %s \n%s {\n%s.\n}' % (
            name, alias_def, stmt)

compiles(CreateQuadStorageStmt)(CreateQuadStorageStmt._compile)


class AlterQuadStorageStmt(Executable, SparqlMappingStatement):
    def __init__(self, mapping, clause, alias_defs):
        super(AlterQuadStorageStmt, self).__init__(mapping)
        self.clause = clause
        self.alias_defs = alias_defs

    def _compile(self, compiler, **kwargs):
        kwargs['literal_binds'] = True
        stmt = compiler.process(self.clause, **kwargs)
        name = compiler.process(
            self.as_clause(self.mapping.name), **kwargs)
        alias_def = '\n'.join((compiler.process(alias_def, **kwargs)
                               for alias_def in self.alias_defs))
        return 'alter quad storage %s \n%s {\n%s.\n}' % (
            name, alias_def, stmt)

compiles(AlterQuadStorageStmt)(AlterQuadStorageStmt._compile)


class DeclareAliasStmt(ClauseElement):
    def __init__(self, table, name):
        super(DeclareAliasStmt, self).__init__()
        self.table = table
        self.name = name

    def _compile(self, compiler, **kwargs):
        # There must be a better way...
        column = self.table.columns.values()[0]
        table_name = compiler.process(
            column, **kwargs).rsplit('.', 1)[0]
        return "FROM %s AS %s" % (
            table_name, self.name)

compiles(DeclareAliasStmt)(DeclareAliasStmt._compile)


class AliasConditionStmt(SparqlStatement):
    def __init__(self, nsm, c_alias_set):
        super(AliasConditionStmt, self).__init__(nsm)
        self.c_alias_set = c_alias_set

    def _compile(self, compiler, **kwargs):
        # Horrid monkey patching. Better than before, though.
        old_quote = compiler.preparer.quote
        c_alias_set = self.c_alias_set
        alias_names = {
            c_alias_set.alias_name(a) for a in self.c_alias_set.aliases}

        def quote(value, force=None):
            if value in alias_names:
                return "^{%s.}^" % value
            return old_quote(value, force)
        compiler.preparer.quote = quote
        condition = compiler.process(
            self.c_alias_set.aliased_term(), **kwargs)
        compiler.preparer.quote = old_quote
        return "WHERE (%s)" % (condition, )

compiles(AliasConditionStmt)(AliasConditionStmt._compile)


class CreateGraphStmt(SparqlMappingStatement):
    def __init__(self, mapping, clauses):
        super(CreateGraphStmt, self).__init__(mapping)
        self.clauses = clauses

    def _compile(self, compiler, **kwargs):
        graph_map = self.mapping
        inner = ''.join((compiler.process(clause, **kwargs)
                         for clause in self.clauses))
        stmt = 'graph %s%s {\n%s.\n}' % (
            compiler.process(self.as_clause(graph_map.iri), **kwargs),
            ' option(%s)' % (graph_map.option) if graph_map.option else '',
            inner)
        if graph_map.name:
            name = compiler.process(self.as_clause(graph_map.name), **kwargs)
            stmt = 'create %s as %s ' % (name, stmt)
        return stmt

compiles(CreateGraphStmt)(CreateGraphStmt._compile)


class ImportGraphStmt(SparqlMappingStatement):
    def __init__(self, mapping, storage):
        super(ImportGraphStmt, self).__init__(mapping)
        self.storage = storage

    def _compile(self, compiler, **kwargs):
        graphmap = self.mapping
        graphname = compiler.process(
            self.as_clause(graphmap.name), **kwargs)
        storage_name = compiler.process(
            self.as_clause(self.storage.name), **kwargs)
        return "create %s using storage %s" % (
            graphname, storage_name)

compiles(ImportGraphStmt)(ImportGraphStmt._compile)


class DeclareQuadMapStmt(SparqlMappingStatement):
    def __init__(self, mapping, subject, predicate, object_, initial=True):
        super(DeclareQuadMapStmt, self).__init__(mapping)
        self.subject = subject
        self.predicate = predicate
        self.object = object_
        self.initial = initial

    def _compile(self, compiler, **kwargs):
        # TODO: Can I introspect the compiler to see where I stand?
        clause = "" if self.initial else ".\n"
        subject = compiler.process(self.subject, **kwargs)\
            if self.subject is not None else None
        predicate = compiler.process(self.predicate, **kwargs)\
            if self.predicate is not None else None
        object_ = compiler.process(self.object, **kwargs)
        if predicate is None:
            clause = ", %s" % (object_)
        elif subject is None:
            clause = ";\n\t%s %s" % (predicate, object_)
        else:
            clause += "%s %s %s" % (
                subject, predicate, object_)
        missing_aliases = self.mapping.missing_aliases()
        if missing_aliases:
            missing_aliases = ['using '+inspect(alias).selectable.name
                               for alias in missing_aliases]
            clause += " option(%s)" % ', '.join(missing_aliases)
        if self.mapping.name:
            name = compiler.process(
                self.as_clause(self.mapping.name), **kwargs)
            clause += " as "+name
        return clause

compiles(DeclareQuadMapStmt)(DeclareQuadMapStmt._compile)


class RdfLiteralStmt(SparqlStatement):
    def __init__(self, literal, nsm=None):
        super(RdfLiteralStmt, self).__init__(nsm)
        self.literal = literal

    def _compile(self, compiler, **kwargs):
        nsm = kwargs.get('nsm', self.nsm)
        return self.literal.n3(nsm)

compiles(RdfLiteralStmt)(RdfLiteralStmt._compile)


# str(statement.compile(compile_kwargs=dict(nsm=nsm)))

class Mapping(object):
    def __init__(self, name, nsm=None):
        self.name = name
        self.nsm = nsm

    #@abstractproperty
    def mapping_name(self):
        raise NotImplemented()

    def drop(self, session, force=False, in_storage=None):
        errors = []
        # first drop submaps I know about
        for submap in self.known_submaps():
            errors.extend(submap.drop(session, force, in_storage))
        # then drop submaps I don't know about
        for submap in self.effective_submaps(session):
            errors.extend(submap.drop(session, force, in_storage))
        # It may have been ineffective. Abort.
        remaining = self.effective_submaps(session)
        remaining = list(remaining)
        if remaining:
            errors.append("Remaining in %s: " % (repr(self),)
                          + ', '.join((repr(x) for x in remaining)))
            if not force:
                return errors
        stmt = self.drop_statement()
        if stmt is not None:
            # This should work in theory. It also works in the CLI.
            # if in_storage is not None and in_storage != self:
            #     ctx_stmt = in_storage.alter_clause(stmt)
            errors.extend(session.execute(WrapSparqlStatement(stmt)))
        return errors

    def drop_statement(self):
        if self.name is not None:
            return DropMappingStmt(self)

    def known_submaps(self):
        return ()

    def effective_submaps(self, session):
        return ()

    def patterns_iter(self):
        return ()

    def prefixes(self):
        assert self.nsm
        return "\n".join("PREFIX %s: %s " % (
            p, ns.n3()) for (p, ns) in self.nsm.namespaces())

    def prefix_clauses(self):
        assert self.nsm
        return [DeclarePrefixStmt(p, ns)
                for (p, ns) in self.nsm.namespaces()]

    def iri_definition_clauses(self):
        return filter(
            lambda x: x is not None,
            (iri.definition_statement()
             for iri in set(self.patterns_iter())))

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

    def as_clause(self, arg):
        if isinstance(arg, (Column, InstrumentedAttribute)):
            assert self.alias_set
            return self.alias_set.aliased_term(arg)
        elif isinstance(arg, ClauseElement):
            return arg
        if isinstance(arg, Mapping):
            assert False
        elif getattr(arg, 'n3', None) is not None:
            return RdfLiteralStmt(arg, self.nsm)
        elif isinstance(arg, (str, unicode, int)):
            return arg
        raise TypeError()

    def __repr__(self):
        if self.name:
            return "<%s %s>" % (
                self.__class__.__name__, self.name.n3(self.nsm))
        else:
            return "<%s <?>>" % (self.__class__.__name__, )

    def set_namespace_manager(self, nsm):
        self.nsm = nsm

    def set_alias_set(self, alias_set):
        self.alias_set = alias_set

    def known_prefix_uris(self, session):
        return {uri for (prefix, uri)
                in session.execute('XML_SELECT_ALL_NS_DECLS()')}


class ApplyFunction(Mapping, SparqlMappingStatement, FunctionElement):
    def __init__(self, fndef, nsm=None, *arguments):
        super(ApplyFunction, self).__init__(None, nsm)
        self.fndef = fndef
        self.name = fndef.name
        self.arguments = tuple(arguments)

    def resolve(self, *classes):
        self.arguments = tuple((
            self.resolve_argument(arg, classes) for arg in self.arguments))

    def clone(self):
        return ApplyFunction(self.fndef, self.nsm, *self.arguments)

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
            self.alias_set == other.alias_set and \
            self.arguments == other.arguments

    def set_namespace_manager(self, nsm):
        super(ApplyFunction, self).set_namespace_manager(nsm)
        self.fndef.set_namespace_manager(nsm)
        for arg in self.arguments:
            if isinstance(arg, Mapping):
                arg.set_namespace_manager(nsm)

    def set_alias_set(self, alias_set):
        super(ApplyFunction, self).set_alias_set(alias_set)
        #self.fndef.set_alias_set(alias_set)
        for arg in self.arguments:
            if isinstance(arg, Mapping):
                arg.set_alias_set(alias_set)

    def _compile(self, compiler, **kwargs):
        nsm = kwargs.get('nsm', self.nsm or self.fndef.nsm)
        args = [compiler.process(self.as_clause(arg), **kwargs)
                for arg in self.arguments]
        return "%s (%s) " % (
            self.name.n3(nsm), ', '.join(args))

compiles(ApplyFunction)(ApplyFunction._compile)


class VirtuosoAbstractFunction(Mapping):
    def __init__(self, name, nsm=None, *arguments):
        super(VirtuosoAbstractFunction, self).__init__(name, nsm)
        self.arguments = tuple(arguments)

    def patterns_iter(self):
        yield self

    def apply(self, *arguments):
        return ApplyFunction(self, self.nsm, *arguments)

    def definition_statement(self):
        return None


class IriClass(VirtuosoAbstractFunction):
    def __init__(self, name, nsm=None):
        super(IriClass, self).__init__(name, nsm)

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

    def definition_statement(self):
        return CreateIriClassStmt(self)

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
        res = list(session.execute("""SPARQL %s\n SELECT ?map ?submap
            WHERE {graph virtrdf: {
            ?map virtrdf:qmSubjectMap ?qmsm . ?qmsm virtrdf:qmvIriClass %s
            OPTIONAL {?map virtrdf:qmUserSubMaps ?o
                      . ?o ?p ?submap . ?submap a virtrdf:QuadMap}
            }}""" % (prefixes, self.name.n3(self.nsm))))
        # Or is it ?qmsm virtrdf:qmvFormat ? Seems broader
        for (mapname, submapname) in res:
            if submapname:
                yield GraphQuadMapPattern(
                    None, None, name=URIRef(mapname), nsm=self.nsm)
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
        self.conditionc_set = set()  # The signatures of condition clauses
        self.alias_set = None
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

    def clone_with_defaults(
            self, subject=None, obj=None, graph_name=None,
            name=None, condition=None):
        subject = subject if self.subject is None else self.subject
        if (subject is not None and isinstance(subject, ApplyFunction)):
            subject = subject.clone()
        new_obj = obj if self.object is None else self.object
        if (new_obj is not None and isinstance(new_obj, ApplyFunction)):
            new_obj = new_obj.clone()
        if (obj is not None and self.object is not None
                and isinstance(self.object, ApplyFunction)):
            if isinstance(obj, ApplyFunction):
                new_obj.set_arguments(*obj.arguments)
            else:
                new_obj.set_arguments(obj)
        graph_name = graph_name if self.graph_name is None else self.graph_name
        name = name if self.name is None else self.name
        condition = condition if self.condition is None else self.condition
        return self.__class__(subject, self.predicate, new_obj, graph_name,
                              name, condition, self.nsm)

    def aliased_classes(self, as_alias=True):
        v = GatherColumnsVisitor(self.alias_set.mgr.class_reg)

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
            alias_of = {inspect(alias)._target: alias
                        for alias in self.alias_set.aliases}
            classes = {alias_of[cls] for cls in classes}
        return classes

    def missing_aliases(self):
        term_aliases = self.aliased_classes(self.alias_set)
        return set(self.alias_set.aliases) - term_aliases

    def declaration_clause(
            self, share_subject=False, share_predicate=False, initial=True):
        return DeclareQuadMapStmt(
            self,
            self.as_clause(self.subject)
                if not share_subject else None,
            self.as_clause(self.predicate)
                if not share_predicate else None,
            self.as_clause(self.object),
            initial)

    def terms(self):
        return (self.subject, self.predicate, self.object, self.graph_name)

    def patterns_iter(self):
        if isinstance(self.subject, Mapping):
            for p in self.subject.patterns_iter():
                yield p
        if isinstance(self.object, Mapping):
            for p in self.object.patterns_iter():
                yield p

    def __repr__(self):
        elements = [self.name or "?"] + [repr(t) for t in self.terms()]
        return "<QuadMapPattern %s: %s %s %s %s>" % tuple(elements)

    def term_representations(self):
        representations = [repr(t) for t in self.terms()]
        if self.alias_set is not None:
            representations.insert(0, self.alias_set.id)
        return representations

    def set_namespace_manager(self, nsm):
        super(QuadMapPattern, self).set_namespace_manager(nsm)
        for term in self.terms():
            if isinstance(term, Mapping):
                term.set_namespace_manager(nsm)

    def set_alias_set(self, alias_set):
        super(QuadMapPattern, self).set_alias_set(alias_set)
        for t in self.terms():
            if isinstance(t, Mapping):
                t.set_alias_set(alias_set)


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
            mapper = getattr(ann, 'parententity',
                             getattr(ann, 'parentmapper', None))
            if mapper:
                cls = getattr(mapper, 'class_', None)
                if cls:
                    return cls
    if class_registry:
        table = col.table
        if isinstance(table, Alias):
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


class ClassAlias(BaseAliasSet):
    def __init__(self, mgr, column):
        super(ClassAlias, self).__init__(mgr, "0", column)

    @memoized_property
    def aliases(self):
        return [self.alias(self.term)]

    def get_column_alias(self, column):
        if isinstance(column, Column):
            assert column.table == inspect(self.term).local_table
        else:
            assert column.class_ == self.term
        return getattr(self.aliases[0], column.key)

    def alias_statements(self):
        return (DeclareAliasStmt(
            inspect(self.term).local_table,
            self._alias_name(self.term)), )

    def __eq__(self, other):
        return (
            other.__class__ == self.__class__
            and hash(self) == hash(other)
            and self.term == other.term)


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

    def alias_statements(self):
        return (DeclareAliasStmt(
                inspect(alias).mapper.local_table,
                self.alias_name(alias))
                for alias in self.aliases)

    def where_statement(self, nsm):
        return AliasConditionStmt(nsm, self)

    def __eq__(self, other):
        return (
            other.__class__ == self.__class__
            and hash(self) == hash(other)
            and _sig(self.term) == _sig(other.term))

    def remove_class(self, cls):
        self.extra_classes.remove(cls)
        return len(self.extra_classes) != 0


def issuperclass(cls, classdef):
    "Reverse of issubclass"
    if isinstance(classdef, type):
        return issubclass(classdef, cls)
    for cls2 in classdef:
        if issubclass(cls2, cls):
            return True


class ClassAliasManager(object):
    def __init__(self, class_reg=None):
        self.alias_sets = {}
        self.base_aliases = {}
        self.class_reg = class_reg
        self.id_counter = 1

    def get_column_class(self, col, use_annotations=True):
        return _get_column_class(col, self.class_reg, use_annotations)

    def superclass_conditions(self, column):
        """Columns defined on superclass may come from another table.
        Here we calculate the necessary joins.
        Also class identity conditions for single-table inheritance
        """
        conditions = {}
        if isinstance(column, (int, str, unicode, Identifier)):
            return {}, column
        cls = self.get_column_class(column)
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
        foreign_keys = set()  # From foreign keys: Not arguments but conditions
        for column in columns:
            conds, newcol = self.superclass_conditions(column)
            conditions.update(conds)
            newcols.append(newcol)
            foreign_keys = getattr(column, 'foreign_keys', ())
            for foreign_key in foreign_keys:
                # Do not bother with inheritance here
                cls1 = self.get_column_class(foreign_key.parent)
                cls2 = self.get_column_class(foreign_key.column)
                if issubclass(cls1, cls2) or issubclass(cls2, cls1):
                    continue
                foreign_keys.add(foreign_key)
        return conditions, newcols, foreign_keys

    def add_quadmap(self, quadmap):
        conditions = {}
        all_args = []
        all_foreign_keys = set()
        for term_index in ('subject', 'predicate', 'object', 'graph_name'):
            term = getattr(quadmap, term_index)
            if isinstance(term, ApplyFunction):
                tconditions, args, foreign_keys = \
                    self.superclass_conditions_multiple(
                        term.arguments)
                conditions.update(tconditions)
                all_foreign_keys.update(foreign_keys)
                term.set_arguments(*args)
                # Another assumption
                all_args.extend(args)
            elif isinstance(term, (InstrumentedAttribute, Column)):
                tconditions, arg = self.superclass_conditions(term)
                conditions.update(tconditions)
                all_args.append(arg)
                setattr(quadmap, term_index, arg)
        term_classes = {
            self.get_column_class(col) for col in all_args}
        foreign_classes = {
            self.get_column_class(fk.column) for fk in all_foreign_keys
        } - term_classes
        if quadmap.condition is not None:
            # in some cases, sqla transforms condition terms to
            # use the superclass. Lots of silly gymnastics to invert that.
            g = GatherColumnsVisitor(self.class_reg)
            g.traverse(quadmap.condition)
            for col in g.columns:
                cls = self.get_column_class(col, False)
                if cls in term_classes:
                    continue
                sub = self.get_column_class(col)
                if sub in term_classes:
                    continue
                subs = [c for c in term_classes if issubclass(c, cls)]
                if subs:
                    assert len(subs) == 1
                    sub = subs[0]
                    col = getattr(sub, col.key, None)
                    tconditions, arg = self.superclass_conditions(col)
                    conditions.update(tconditions)
                    continue
                # TODO: Can I change terms in the condition?
                # I think taken care of downstream
                # Next step: the column may be a reference to a foreign key that was mentioned
                # TODO: Maybe even look for an arbitrary join with with one of the arg classes.
                subs = [c for c in foreign_classes if issubclass(c, cls)]
                if subs or cls in foreign_classes or sub in foreign_classes:
                    # We have a condition term based on a foreign column. 
                    # Now find which foreign column to use for the join
                    for fkey in foreign_keys:
                        foreign_class = self.get_column_class(fkey.column)
                        if issuperclass(self.get_column_class(fkey.parent, False), tuple(term_classes))\
                                and issuperclass(foreign_class, tuple(foreign_classes)):
                            condition = (fkey.parent == fkey.column)
                            conditions[_sig(condition)] = condition
                            if cls != foreign_class:
                                col = getattr(foreign_class, col.key)
                                tconditions, arg = self.superclass_conditions(col)
                                conditions.update(tconditions)
                            break
        quadmap.and_conditions(conditions.values())
        for arg in all_args:
            if isinstance(arg, (Column, InstrumentedAttribute)):
                self.add_class(arg, quadmap.condition)
        # TODO: Horrible!
        # Maybe quadmap should have ref class?
        if quadmap.condition is not None:
            condition_c = _sig(quadmap.condition)
            alias_set = self.alias_sets[condition_c]
        else:
            subject = quadmap.subject
            # TODO: Abstract those assumptions
            assert isinstance(subject, ApplyFunction)
            id_column = subject.arguments[0]
            cls = self.get_column_class(id_column)
            alias_set = self.base_aliases[cls]
        quadmap.set_alias_set(alias_set)
        return alias_set

    def add_class(self, column_or_class, condition=None):
        if isinstance(column_or_class, type):
            cls = column_or_class
        elif isinstance(column_or_class, (Column, InstrumentedAttribute)):
            cls = self.get_column_class(column_or_class)
        else:
            assert False
        if condition is not None:
            id = str(self.id_counter)
            self.id_counter += 1
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
            condition_c = _sig(condition)
            cas = self.alias_sets[condition_c]
            remaining = cas.remove_class(cls)
            if not remaining:
                del self.alias_sets[condition_c]

    def alias_statements(self):
        return chain(*(alias_set.alias_statements() for alias_set
                       in chain(self.base_aliases.itervalues(),
                                self.alias_sets.itervalues())))

    def where_statements(self, nsm):
        return [alias_set.where_statement(nsm)
                for alias_set in self.alias_sets.itervalues()]

    def get_column_alias(self, column, condition=None):
        if condition is not None:
            alias_set = self.alias_sets[_sig(condition)]
        else:
            cls = self.get_column_class(column)
            if cls not in self.base_aliases:
                self.base_aliases[cls] = ClassAlias(self, cls)
            alias_set = self.base_aliases[cls]
        return alias_set.get_column_alias(column)


def simple_iri_accessor(sqla_cls):
    """A function that extracts the IRIClass from a SQLAlchemy ORM class.
    This is an example, but different extractors will use different iri_accessors"""
    try:
        mapper = inspect(sqla_cls)
        info = mapper.local_table.info
        return info.get('rdf_iri', None)
    except NoInspectionAvailable as err:
        return None


class ClassPatternExtractor(object):
    "Obtains RDF quad definitions from a SQLAlchemy ORM class."
    def __init__(
            self, alias_manager, iri_accessor=simple_iri_accessor, graph=None):
        self.graph = graph
        self.iri_accessor = iri_accessor
        self.alias_manager = alias_manager

    def get_subject_pattern(self, sqla_cls):
        try:
            iri = self.iri_accessor(sqla_cls)
            mapper = inspect(sqla_cls)
            keys = [getattr(sqla_cls, key.key) for key in mapper.primary_key]
            return iri.apply(*keys)
        except Exception as err:
            pass

    def make_column_name(self, cls, column):
        pass

    def column_as_reference(self, column):
        # Replace with reference
        fks = list(column.foreign_keys)
        assert len(fks) == 1
        fk = fks[0]
        target = _get_column_class(
            fk.column, self.alias_manager.class_reg)
        iri = self.iri_accessor(target)
        return iri.apply(column)

    def qmp_with_defaults(self, qmp, subject_pattern, sqla_cls, column=None):
        name = None
        if column is not None:
            name = self.make_column_name(sqla_cls, column)
            if column.foreign_keys:
                column = self.column_as_reference(column)
        return qmp.clone_with_defaults(subject_pattern, column, self.graph.name, name)

    def extract_column_info(self, sqla_cls, subject_pattern):
        mapper = inspect(sqla_cls)
        info = mapper.local_table.info
        supercls = sqla_cls.mro()[1]
        for c in mapper.columns:
            # Local columns only to avoid duplication
            # exception for abstract superclass
            if (getattr(supercls, c.key, None) is not None
                and not isabstract(supercls)):
                    continue
            if 'rdf' in c.info:
                qmp = c.info['rdf']
                if isinstance(qmp, QuadMapPattern):
                    qmp = self.qmp_with_defaults(qmp, subject_pattern, sqla_cls, c)
                    if qmp.graph_name == self.graph.name:
                        qmp.resolve(sqla_cls)
                        yield qmp

    def extract_info(self, sqla_cls, subject_pattern=None):
        if isabstract(sqla_cls):
            return
        subject_pattern = subject_pattern or \
            self.get_subject_pattern(sqla_cls)
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
    def __init__(self, graph_iri, storage, name=None, option=None, nsm=None):
        super(GraphQuadMapPattern, self).__init__(name, nsm)
        self.iri = graph_iri
        self.qmps = set()
        self.option = option
        if storage is not None:
            storage.add_graphmap(self)

    def known_submaps(self):
        return self.qmps

    def effective_submaps(self, session):
        assert self.nsm
        if not self.name:
            return
        prefixes = self.prefixes()
        res = list(session.execute("""SPARQL %s SELECT ?map
            WHERE {graph virtrdf: {
            %s virtrdf:qmUserSubMaps ?o . ?o ?p ?map . ?map a virtrdf:QuadMap
            }}""" % (prefixes, self.name.n3(self.nsm))))
        for (mapname, ) in res:
            yield QuadMapPattern(
                name=URIRef(mapname), graph_name=self.name, nsm=self.nsm)

    def declaration_clause(self):
        assert self.nsm
        assert self.alias_manager
        qmps = list(self.qmps)
        qmps.sort(key=QuadMapPattern.term_representations)
        clauses = []
        initial = True
        subject = None
        predicate = None
        for qmp in qmps:
            clauses.append(qmp.declaration_clause(
                subject == qmp.subject,
                predicate == qmp.predicate,
                initial))
            subject = qmp.subject
            predicate = qmp.predicate
            initial = False
        return CreateGraphStmt(self, clauses)

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

    def import_clause(self, storage):
        return ImportGraphStmt(self, storage)

    def add_patterns(self, patterns):
        assert self.nsm
        assert self.alias_manager
        for pattern in patterns:
            assert isinstance(pattern, QuadMapPattern)
            if pattern not in self.qmps:
                pattern.set_namespace_manager(self.nsm)
                self.qmps.add(pattern)
                self.alias_manager.add_quadmap(pattern)

    def set_storage(self, storage):
        self.storage = storage
        self.alias_manager = storage.alias_manager
        self.set_namespace_manager(storage.nsm)


class PatternGraphQuadMapPattern(GraphQuadMapPattern):
    "Reprensents a graph where the graph name is an IRI. Not functional."
    def __init__(self, graph_iri_pattern, storage, alias_set,
                 name=None, option=None, nsm=None):
        super(PatternGraphQuadMapPattern, self).__init__(
            graph_iri_pattern, storage, name, option, nsm)
        self.alias_set = alias_set


class QuadStorage(Mapping):
    def __init__(self, name, imported_graphmaps=None,
                 alias_manager=None, add_default=True, nsm=None):
        super(QuadStorage, self).__init__(name, nsm)
        self.alias_manager = alias_manager or ClassAliasManager()
        self.native_graphmaps = []
        self.imported_graphmaps = imported_graphmaps or []
        if add_default:
            self.imported_graphmaps.append(DefaultQuadMap)

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
        res = list(session.execute("""SPARQL %s SELECT ?map
            WHERE {graph virtrdf: {
            %s virtrdf:qsUserMaps ?o . ?o ?p ?map . ?map a virtrdf:QuadMap
            }}""" % (prefixes, self.name.n3(self.nsm))))
        for (mapname, ) in res:
            yield GraphQuadMapPattern(
                None, self, name=URIRef(mapname), nsm=self.nsm)

    def declaration_clause(self):
        graph_statements = [
            graph.declaration_clause() for graph in self.native_graphmaps]
        import_clauses = [
            graph.import_clause(self) for graph in self.imported_graphmaps]
        alias_defs = chain(self.alias_manager.alias_statements(),
                           self.alias_manager.where_statements(self.nsm))
        return CreateQuadStorageStmt(
            self, graph_statements, list(alias_defs), import_clauses)

    def full_declaration_clause(self):
        return WrapSparqlStatement(
            CompoundSparqlStatement(list(chain(
                self.iri_definition_clauses(),
                (self.declaration_clause(),)))))

    def alter_clause_add_graph(self, gqm):
        alias_defs = chain(self.alias_manager.alias_statements(),
                           self.alias_manager.where_statements(self.nsm))
        return self.alter_clause(gqm.declaration_clause(), alias_defs)

    def alter_clause(self, clause, alias_defs=None):
        alias_defs = alias_defs or ()
        return WrapSparqlStatement(AlterQuadStorageStmt(
            self, clause, list(alias_defs)))

    def patterns_iter(self):
        for qmp in self.native_graphmaps:
            for pat in qmp.patterns_iter():
                yield pat
        for qmp in self.imported_graphmaps:
            for pat in qmp.patterns_iter():
                yield pat

    def add_graphmap(self, graphmap):
        self.native_graphmaps.append(graphmap)
        graphmap.set_storage(self)

    def drop(self, session, force=False, storage=None):
        super(QuadStorage, self).drop(session, force, self)

DefaultNSM = NamespaceManager(Graph())
DefaultNSM.bind('virtrdf', VirtRDF)
DefaultQuadStorage = QuadStorage(VirtRDF.DefaultQuadStorage,
                                 add_default=False, nsm=DefaultNSM)
DefaultQuadMap = GraphQuadMapPattern(
    None, DefaultQuadStorage, VirtRDF.DefaultQuadMap)
