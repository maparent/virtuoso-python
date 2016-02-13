import re
from itertools import chain
from collections import OrderedDict

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.inspection import inspect
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.util import AliasedInsp
from sqlalchemy.schema import Column
from sqlalchemy.sql.expression import (
    ClauseElement, Executable, FunctionElement)
from sqlalchemy.sql.visitors import Visitable
from sqlalchemy.types import TypeEngine
from rdflib import Namespace, URIRef, Graph
from rdflib.namespace import NamespaceManager
from future.utils import string_types

from virtuoso.quadextractor import (
    GroundedClassAlias, GatherColumnsVisitor, ConditionSet,
    ClassPatternExtractor)
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
                for vname, vtype in mapping.vars.iteritems())
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
        column = next(iter(self.table.columns.values()))
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
        if c_alias_set.conditions.condition is None:
            return ""
        alias_names = {
            c_alias_set.get_alias_name(a) for a in c_alias_set.aliases}

        def quote(value, force=None):
            if value in alias_names:
                return "^{%s.}^" % value
            return old_quote(value, force)
        compiler.preparer.quote = quote
        condition = compiler.process(
            c_alias_set.conditions.condition, **kwargs)
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
        # else:
        #     stmt = 'create ' + stmt
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
            clause = ",\n   %s" % (object_)
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

    def exists(self, session, nsm):
        assert self.mapping_type
        r = session.execute(
            'sparql ask where { graph virtrdf: { %s a %s }}' % (
                self.name.n3(nsm), self.mapping_type.n3(nsm)))
        return bool(len(r))

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
        return [x for x in (
                    iri.definition_statement()
                    for iri in set(self.patterns_iter()))
                if x is not None]

    @staticmethod
    def resolve_argument(arg, classes):
        if isinstance(arg, (
                InstrumentedAttribute, ClauseElement, GroundedClassAlias)):
            return arg
        if isinstance(classes, (list, tuple)):
            classes = {cls.__name__: cls for cls in classes}
        if isinstance(arg, (int, bool)):
            return arg
        if isinstance(arg, string_types):
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
        elif isinstance(arg, string_types + (int, bool)):
            return arg
        raise TypeError()

    def __repr__(self):
        name = self.name
        if name:
            if isinstance(name, URIRef):
                name = name.n3(self.nsm)
            return "<%s %s>" % (
                self.__class__.__name__, name)
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
        self.alias_set = None
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

    def __repr__(self):
        def _repr(a):
            if isinstance(a, InstrumentedAttribute):
                return ".".join((a.class_.__name__, a.key))
            return repr(a)
        return "<ApplyFunction %s%s>" % (
            self.fndef,
            tuple([_repr(a) for a in self.arguments]))

    def set_namespace_manager(self, nsm):
        super(ApplyFunction, self).set_namespace_manager(nsm)
        self.fndef.set_namespace_manager(nsm)
        for arg in self.arguments:
            if isinstance(arg, Mapping):
                arg.set_namespace_manager(nsm)

    def set_alias_set(self, alias_set):
        super(ApplyFunction, self).set_alias_set(alias_set)
        # self.fndef.set_alias_set(alias_set)
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

    mapping_type = VirtRDF.QuadMapFormat

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
        for k, v in self.vars.iteritems():
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


class QuadMapPattern(Mapping):

    mapping_type = VirtRDF.QuadMap

    def __init__(self, subject=None, predicate=None, obj=None,
                 graph_name=None, name=None, conditions=None, nsm=None):
        super(QuadMapPattern, self).__init__(name, nsm)
        self.graph_name = graph_name
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.condition_set = ConditionSet(conditions)
        self.alias_set = None

    @property
    def mapping_name(self):
        return "quad map"

    def and_condition(self, condition):
        self.condition_set.add_condition(condition)

    def and_conditions(self, conditions):
        self.condition_set.add_conditions(conditions)

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
        condition = condition if not self.condition_set \
            else self.condition_set.as_list()
        return self.__class__(subject, self.predicate, new_obj, graph_name,
                              name, condition, self.nsm)

    def aliased_classes(self, as_alias=True):
        v = GatherColumnsVisitor(self.alias_set.cpe.class_reg)

        def add_term(t):
            if isinstance(t, ApplyFunction):
                for sub in t.arguments:
                    add_term(sub)
            elif isinstance(t, Visitable):
                v.traverse(t)
            elif isinstance(t, InstrumentedAttribute):
                parent = t._parententity
                if (isinstance(parent, AliasedInsp)
                        and isinstance(parent.entity, GroundedClassAlias)
                        and parent.entity.get_name() in self.alias_set.aliases_by_name):
                    v.columns.add(getattr(
                        self.alias_set.aliases_by_name[parent.entity.get_name()], t.key))
                else:
                    v.columns.add(getattr(t._parententity.c, t.key))
            elif isinstance(t, Column):
                v.columns.add(t)
        for term in self.terms():
            add_term(term)
        classes = v.get_classes()
        if as_alias:
            alias_of = {inspect(alias)._target: alias
                        for alias in self.alias_set.aliases}
            # This should not happen.
            assert not {cls for cls in classes
                        if isinstance(cls, type) and cls not in alias_of}
            classes = {alias_of.get(cls, cls) for cls in classes}
        return classes

    def missing_aliases(self):
        term_aliases = self.aliased_classes()
        return set(self.alias_set.aliases) - term_aliases

    def declaration_clause(
            self, share_subject=False, share_predicate=False, initial=True):
        return DeclareQuadMapStmt(
            self,
            self.as_clause(self.subject)
                if not share_subject else None,
            self.as_clause(self.predicate)
                if not (share_predicate and share_subject) else None,
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
            representations.insert(0, self.alias_set.uid)
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


class GraphQuadMapPattern(Mapping):

    mapping_type = VirtRDF.QuadMap

    def __init__(self, graph_iri, storage, name=None, option=None, nsm=None):
        super(GraphQuadMapPattern, self).__init__(name, nsm)
        self.iri = graph_iri
        self.qmps = set()
        self.option = option
        self.storage = storage
        if storage is not None:
            storage.add_graphmap(self)
            self.set_namespace_manager(storage.nsm)

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
        #assert self.alias_manager
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

    def add_patterns(self, base_class, patterns):
        for pattern in patterns:
            self.add_pattern(base_class, pattern)

    def add_pattern(self, base_class, pattern):
        assert self.nsm
        assert isinstance(pattern, QuadMapPattern)
        if pattern not in self.qmps:
            pattern.set_namespace_manager(self.nsm)
            self.qmps.add(pattern)
            #self.alias_manager.add_quadmap(base_class, pattern)

    def set_storage(self, storage):
        self.storage = storage
        #self.alias_manager = storage.alias_manager
        self.set_namespace_manager(storage.nsm)


class PatternGraphQuadMapPattern(GraphQuadMapPattern):
    "Reprensents a graph where the graph name is an IRI. Not functional."
    def __init__(self, graph_iri_pattern, storage, alias_set,
                 name=None, option=None, nsm=None):
        super(PatternGraphQuadMapPattern, self).__init__(
            graph_iri_pattern, storage, name, option, nsm)
        self.alias_set = alias_set


class QuadStorage(Mapping):

    mapping_type = VirtRDF.QuadStorage

    def __init__(self, name, alias_manager, imported_graphmaps=None,
                 add_default=True, nsm=None):
        super(QuadStorage, self).__init__(name, nsm)
        self.alias_manager = alias_manager
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
DefaultQuadStorage = QuadStorage(VirtRDF.DefaultQuadStorage, None,
                                 add_default=False, nsm=DefaultNSM)
DefaultQuadMap = GraphQuadMapPattern(
    None, DefaultQuadStorage, VirtRDF.DefaultQuadMap)
