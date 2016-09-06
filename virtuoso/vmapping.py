from itertools import chain

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import (
    ClauseElement, Executable)
from rdflib import Namespace, URIRef, Graph
from rdflib.namespace import NamespaceManager

from virtuoso.quadextractor import (
    GroundedClassAlias, ClassPatternExtractor)
from .vstore import VirtuosoNamespaceManager
from .mapping import (
    Mapping, ApplyFunction, IriClass, PatternIriClass,
    QuadMapPattern, GraphQuadMapPattern, AbstractFunction)


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

class VirtuosoMapping(Mapping):

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

    def known_prefix_uris(self, session):
        return {uri for (prefix, uri)
                in session.execute('XML_SELECT_ALL_NS_DECLS()')}


class VirtuosoAbstractFunction(AbstractFunction):
    def apply(self, *arguments):
        return VirtuosoApplyFunction(self, self.nsm, *arguments)


class VirtuosoApplyFunction(ApplyFunction, VirtuosoMapping, SparqlMappingStatement):
    @property
    def mapping_name(self):
        raise NotImplemented()

    def _compile(self, compiler, **kwargs):
        nsm = kwargs.get('nsm', self.nsm or self.fndef.nsm)
        args = [compiler.process(self.as_clause(arg), **kwargs)
                for arg in self.arguments]
        return "%s (%s) " % (
            self.name.n3(nsm), ', '.join(args))

    def clone(self):
        return VirtuosoApplyFunction(self.fndef, self.nsm, *self.arguments)

compiles(VirtuosoApplyFunction)(VirtuosoApplyFunction._compile)


class VirtuosoIriClass(VirtuosoAbstractFunction, IriClass, VirtuosoMapping):
    mapping_type = VirtRDF.QuadMapFormat

    @property
    def mapping_name(self):
        return "iri class"


class VirtuosoPatternIriClass(PatternIriClass, VirtuosoIriClass):
    def definition_statement(self):
        return CreateIriClassStmt(self)

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
                yield VirtuosoGraphQuadMapPattern(
                    None, None, name=URIRef(mapname), nsm=self.nsm)
            else:
                yield VirtuosoQuadMapPattern(name=URIRef(mapname), nsm=self.nsm)


class VirtuosoQuadMapPattern(QuadMapPattern, VirtuosoMapping):

    mapping_type = VirtRDF.QuadMap

    @property
    def mapping_name(self):
        return "quad map"

    def import_stmt(self, storage_name):
        assert self.name
        assert self.nsm
        return "create %s using storage %s . " % (
            self.name.n3(self.nsm), storage_name.n3(self.nsm))

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


class VirtuosoGraphQuadMapPattern(GraphQuadMapPattern, VirtuosoMapping):

    mapping_type = VirtRDF.QuadMap

    def __init__(self, graph_iri, storage, name=None, option=None, nsm=None):
        super(VirtuosoGraphQuadMapPattern, self).__init__(graph_iri, name, nsm)
        self.option = option
        self.storage = storage
        if storage is not None:
            storage.add_graphmap(self)
            self.set_namespace_manager(storage.nsm)

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
            yield VirtuosoQuadMapPattern(
                name=URIRef(mapname), graph_name=self.name, nsm=self.nsm)

    def declaration_clause(self):
        assert self.nsm
        #assert self.alias_manager
        qmps = list(self.qmps)
        qmps.sort(key=VirtuosoQuadMapPattern.term_representations)
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

    def set_storage(self, storage):
        self.storage = storage
        #self.alias_manager = storage.alias_manager
        self.set_namespace_manager(storage.nsm)


class PatternGraphQuadMapPattern(VirtuosoGraphQuadMapPattern):
    "Reprensents a graph where the graph name is an IRI. Not functional."
    def __init__(self, graph_iri_pattern, storage, alias_set,
                 name=None, option=None, nsm=None):
        super(PatternGraphQuadMapPattern, self).__init__(
            graph_iri_pattern, storage, name, option, nsm)
        self.alias_set = alias_set


class QuadStorage(VirtuosoMapping):

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
            yield VirtuosoGraphQuadMapPattern(
                None, self, name=URIRef(mapname), nsm=self.nsm)

    def declaration_clause(self):
        graph_statements = [
            graph.declaration_clause() for graph in self.native_graphmaps]
        import_clauses = [
            graph.import_clause(self) for graph in self.imported_graphmaps]
        alias_defs = chain(self.alias_statements(),
                           self.where_statements())
        return CreateQuadStorageStmt(
            self, graph_statements, list(alias_defs), import_clauses)

    def full_declaration_clause(self):
        return WrapSparqlStatement(
            CompoundSparqlStatement(list(chain(
                self.iri_definition_clauses(),
                (self.declaration_clause(),)))))

    def alter_clause_add_graph(self, gqm):
        alias_defs = chain(self.alias_statements(),
                           self.where_statements())
        return self.alter_clause(gqm.declaration_clause(), alias_defs)

    def alias_statements(self):
        return chain(*(self.alias_statements_for(alias_set) for alias_set
                       in self.alias_manager.get_alias_makers()))

    def where_statements(self):
        return [self.where_statement_for(alias_set)
                for alias_set in self.alias_manager.get_alias_makers()]

    def alias_statements_for(self, alias_maker):
        return (DeclareAliasStmt(
                inspect(alias).mapper.local_table,
                alias_maker.get_alias_name(alias))
                for alias in alias_maker.aliases)

    def where_statement_for(self, alias_maker):
        return AliasConditionStmt(self.nsm, alias_maker)

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
DefaultQuadMap = VirtuosoGraphQuadMapPattern(
    None, DefaultQuadStorage, VirtRDF.DefaultQuadMap)
