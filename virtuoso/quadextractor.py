from __future__ import print_function
from abc import ABCMeta, abstractmethod
from itertools import chain, islice
from rdflib.term import Identifier
from sqlalchemy import inspect, Column, and_, Table
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.properties import RelationshipProperty, ColumnProperty
from sqlalchemy.orm.util import AliasedClass, ORMAdapter, AliasedInsp
from sqlalchemy.sql import ClauseVisitor, Alias
from sqlalchemy.sql.util import ClauseAdapter
from sqlalchemy.sql.visitors import Visitable
from sqlalchemy.sql.selectable import Join
from sqlalchemy.util import EMPTY_SET
from future.utils import with_metaclass, string_types
from past.builtins import cmp


def _columnish(term):
    return isinstance(term, Column)\
        or (isinstance(term, InstrumentedAttribute)
            and isinstance(term.impl.parent_token, ColumnProperty))


def _propertish(term):
    return isinstance(term, RelationshipProperty)\
        or (isinstance(term, InstrumentedAttribute)
            and isinstance(term.impl.parent_token, RelationshipProperty))


class GroundedClassAlias(AliasedClass):
    def __init__(self, cls, grounded_path, name=None, **kwargs):
        name = name or "alias_"+str(grounded_path)
        kwargs['name'] = name
        table = inspect(cls).local_table
        kwargs['alias'] = table.alias(name=name)
        super(GroundedClassAlias, self).__init__(cls, **kwargs)
        self.path = grounded_path

    def rebased(self, prefix_path):
        path = prefix_path.clone()
        path.extend(self.path)
        i = inspect(self)
        return GroundedClassAlias(i.class_, path)

    def freeze(self, uid):
        return self.clone(self.get_name() + "_" + str(uid))

    def rename(self, name):
        self._aliased_insp.name = name
        self._aliased_insp.selectable.name = name

    def get_name(self):
        return self._aliased_insp.name

    def get_class(self):
        return self._aliased_insp._target

    def clone(self, newname=None):
        if newname is None:
            newname = self.get_name()
        return GroundedClassAlias(self.get_class(), self.path, newname)

    def __repr__(self):
        return "<%s at %x; %s (%s)>" % (
            self.__class__.__name__, id(self),
            self.get_class().__name__, self.get_name())


class GatherColumnsVisitor(ClauseVisitor):
    def __init__(self, class_reg=None):
        super(GatherColumnsVisitor, self).__init__()
        self.columns = set()
        self.class_reg = class_reg

    def visit_column(self, column):
        self.columns.add(column)

    def get_classes(self):
        return {_get_column_class(col, self.class_reg) for col in self.columns}

    def get_tables(self):
        return {col.table for col in self.columns}


class DeferredPath(object):
    def __init__(self, path_sig):
        self.sig = path_sig

    def __str__(self):
        return self.sig

    def resolve(self, classes_by_name):
        components = self.sig.split('__')
        class_name = components.pop(0)
        first = cls = classes_by_name[class_name]
        props = []
        for reln_name in components:
            reln_ia = getattr(cls, reln_name)
            assert isinstance(reln_ia, InstrumentedAttribute)
            reln_prop = reln_ia.prop
            assert isinstance(reln_ia, RelationshipProperty)
            props.append(reln_prop)
            cls = reln_prop.mapper.class_
        return GroundedPath(first, *props)

    def __len__(self):
        return len(self.sig.split('__'))

    def __hash__(self):
        return hash(self.sig)


def _sig(condition):
    return unicode(condition.compile(
        compile_kwargs={"literal_binds": True}))


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
        assert not isinstance(table, AliasedClass)
        if isinstance(table, Alias):
            table = table.original
        cls = _get_class_from_table(table, class_registry)
    assert cls is not None,\
        "Cannot obtain the class from the column " + repr(col)
    return cls


class DebugClauseVisitor(ClauseVisitor):
    def visit_binary(self, binary):
        print("visit_binary", repr(binary))

    def visit_column(self, column):
        print("visit_column", repr(column))

    def visit_bindparam(self, bind):
        print("visit_bindparam", repr(bind))


def simple_iri_accessor(sqla_cls):
    """A function that extracts the IRIClass from a SQLAlchemy ORM class.
    This is an example, but different extractors will use different
    iri_accessors"""
    try:
        mapper = inspect(sqla_cls)
        info = mapper.local_table.info
        return info.get('rdf_iri', None)
    except NoInspectionAvailable:
        return None


def _get_class_from_table(table, class_registry):
    for cls in class_registry.itervalues():
        if isinstance(cls, type) and inspect(cls).local_table == table:
            # return highest such class.
            for supercls in cls.mro():
                if not getattr(supercls, '__mapper__', None):
                    return cls
                if inspect(supercls).local_table != table:
                    return cls
                cls = supercls


class SuperClassRelationship(RelationshipProperty):
    key = 'super'

    def __init__(self, superclass, subclass):
        super(SuperClassRelationship, self).__init__(superclass)
        self.parent = subclass.__mapper__


def sqla_inheritance(cls):
    mapper = inspect(cls)
    while mapper:
        yield mapper.class_
        mapper = mapper.inherits


def sqla_inheritance_with_conditions(cls):
    mapper = inspect(cls)
    while mapper:
        if mapper.class_ == cls or mapper.inherits is None or \
                mapper.inherit_condition is not None:
            yield mapper.class_
        mapper = mapper.inherits


class ClassPatternExtractor(with_metaclass(ABCMeta, object)):
    "Obtains RDF quad definitions from a SQLAlchemy ORM class."
    def __init__(self, class_reg):
        self.class_reg = class_reg
        self.base_alias_makers = {}
        self.alias_makers_by_sig = {}
        self.uid = 0

    @abstractmethod
    def iri_accessor(self, sqla_cls):
        return simple_iri_accessor(sqla_cls)

    @abstractmethod
    def get_base_conditions(self, alias_maker, cls, for_graph):
        base = inspect(cls)._single_table_criterion
        if base is not None:
            return [base]
        return []

    def get_alias_makers(self):
        return self.alias_makers_by_sig.itervalues()

    def get_subject_pattern(self, sqla_cls, alias_maker=None):
        try:
            iri = self.iri_accessor(sqla_cls)
            mapper = inspect(sqla_cls)
            keys = [getattr(sqla_cls, key.key) for key in mapper.primary_key]
            return iri.apply(*keys)
        except Exception:
            pass

    @abstractmethod
    def make_column_name(self, cls, column, for_graph):
        pass

    @abstractmethod
    def class_pattern_name(self, cls, for_graph):
        pass

    def get_column_class(self, col, use_annotations=True):
        return _get_column_class(col, self.class_reg, use_annotations)

    def column_as_reference(self, column):
        # Replace with reference
        assert len(column.foreign_keys) == 1
        fk = next(iter(column.foreign_keys))
        target = self.get_column_class(fk.column)
        iri = self.iri_accessor(target)
        return iri.apply(column)

    def property_as_reference(self, prop, alias_maker):
        def find_direct_link(table, target_table):
            for c in table.c:
                if target_table in (fk.column.table for fk in c.foreign_keys):
                    return c
        source_table = prop.parent.local_table
        source_alias = alias_maker.base_alias
        proximal_table = source_table
        proximal_alias = source_alias
        target = prop.mapper.class_
        target_table = prop.mapper.local_table
        target_alias = alias_maker.alias_from_relns(prop)
        iri = self.iri_accessor(target)
        adapter = ORMAdapter(source_alias).chain(ORMAdapter(target_alias))
        if prop.secondary is not None:
            # What if there is more than one secondary
            if isinstance(prop.secondary, Join):
                def traverse_join(element, adapter):
                    proximal_table = None
                    proximal_alias = None
                    if isinstance(element, Table):
                        alias = alias_maker.alias_from_table(element)
                        adapter = adapter.chain(ORMAdapter(alias))
                        if find_direct_link(element, target_table) is not None:
                            proximal_table = element
                            proximal_alias = alias
                    elif isinstance(element, Join):
                        (adapter, pt, pa) = traverse_join(
                            element.left, adapter)
                        if pt is not None:
                            proximal_table = pt
                            proximal_alias = pa
                        (adapter, pt, pa) = traverse_join(
                            element.right, adapter)
                        if pt is not None:
                            proximal_table = pt
                            proximal_alias = pa
                        alias_maker.add_condition(
                            adapter.traverse(element.onclause))
                    return (adapter, proximal_table, proximal_alias)
                (adapter, proximal_table, proximal_alias) = traverse_join(
                    prop.secondary, adapter)
            else:
                sec_alias = alias_maker.alias_from_table(prop.secondary)
                adapter = adapter.chain(ORMAdapter(sec_alias))
                proximal_table = prop.secondary
                proximal_alias = sec_alias
        alias_maker.alias_from_relns(prop)
        # Add conditions, adapting with the aliases.
        alias_maker.add_condition(adapter.traverse(prop.primaryjoin))
        if prop.secondaryjoin is not None:
            alias_maker.add_condition(adapter.traverse(prop.secondaryjoin))
        # Look for a foreign key leading to the target from the proximal
        # table (source or secondary).
        # TODO: Testing with tables, not classes, this might be an issue.
        for col in prop._calculated_foreign_keys:
            if (col.table == proximal_table
                    and len(col.foreign_keys) == 1
                    and next(iter(col.foreign_keys)).column.table
                    == target_table):
                return iri.apply(getattr(proximal_alias, col.key))
        assert False

    def qmp_with_defaults(
            self, qmp, subject_pattern, sqla_cls, alias_maker,
            for_graph, column=None):
        name = None
        if column is not None:
            name = self.make_column_name(sqla_cls, column, for_graph)
            if isinstance(column, RelationshipProperty):
                column = self.property_as_reference(column, alias_maker)
            elif column.foreign_keys:
                column = self.column_as_reference(column)
        return qmp.clone_with_defaults(
            subject_pattern, column, for_graph.name, name)

    def delayed_column(self, sqla_cls, column, for_graph):
        return False

    def extract_qmps(self, sqla_cls, subject_pattern, alias_maker, for_graph):
        mapper = inspect(sqla_cls)
        supercls = next(islice(
            sqla_inheritance_with_conditions(sqla_cls), 1, 2), None)
        if supercls:
            supermapper = inspect(supercls)
            super_props = set(chain(
                supermapper.columns, supermapper.relationships))
        else:
            super_props = set()
        for c in chain(mapper.columns, mapper.relationships):
            # Local columns only to avoid duplication
            if c in super_props:
                # But there are exceptions
                if not self.delayed_column(sqla_cls, c, for_graph):
                    continue
                # in this case, make sure superclass is in aliases.
                c = self.add_superclass_path(c, sqla_cls, alias_maker)
            if 'rdf' in getattr(c, 'info', ()):
                from virtuoso.mapping import QuadMapPattern
                qmp = c.info['rdf']
                if isinstance(qmp, QuadMapPattern):
                    qmp = self.qmp_with_defaults(
                        qmp, subject_pattern, sqla_cls, alias_maker,
                        for_graph, c)
                    if qmp is not None and qmp.graph_name == for_graph.name:
                        qmp.resolve(sqla_cls)
                        yield qmp

    def can_treat_class(self, cls):
        return self.get_subject_pattern(cls, None) is not None

    def get_base_alias_maker(self, cls, for_graph):
        if cls not in self.base_alias_makers:
            base_alias_maker = AliasMaker(cls, self)
            base_alias_maker.add_conditions(self.get_base_conditions(
                base_alias_maker, cls, for_graph))
            self.base_alias_makers[cls] = base_alias_maker
        return self.base_alias_makers[cls]

    def add_class(self, cls, for_graph):
        if not self.can_treat_class(cls):
            return
        base_alias_maker = self.get_base_alias_maker(cls, for_graph)
        subject_pattern = self.get_subject_pattern(cls, base_alias_maker)
        for qmp in self.extract_qmps(
                cls, subject_pattern, base_alias_maker, for_graph):
            self.add_pattern(
                cls, qmp, for_graph, base_alias_maker, subject_pattern)

    def add_pattern(self, cls, qmp, in_graph, base_alias_maker=None,
                    subject_pattern=None):
        # print "add_pattern", str(qmp.name)
        if base_alias_maker is None:
            base_alias_maker = self.get_base_alias_maker(cls, in_graph)
        if subject_pattern is None:
            subject_pattern = self.get_subject_pattern(cls, base_alias_maker)
        alias_maker = base_alias_maker.clone()
        self.gather_conditions(qmp, alias_maker, in_graph)
        signature = alias_maker.signature()
        if signature in self.alias_makers_by_sig:
            alias_maker = self.alias_makers_by_sig[signature]
        else:
            alias_maker = alias_maker.freeze(self.uid)
            self.uid += 1
            self.alias_makers_by_sig[signature] = alias_maker
        qmp.set_alias_set(alias_maker)
        #self.process_quadmap(qmp, alias_maker, in_graph)
        in_graph.add_pattern(cls, qmp)

    def gather_conditions(self, quadmap, alias_maker, for_graph):
        from virtuoso.mapping import ApplyFunction
        for term_index in ('subject', 'predicate', 'object', 'graph_name'):
            term = getattr(quadmap, term_index)
            if isinstance(term, ApplyFunction):
                args = [self.resolve_term(arg, alias_maker, for_graph)
                        for arg in term.arguments]
                # Add conditions from target terms
                #term.set_arguments(*args)
            elif _columnish(term) or _propertish(term):
                arg = self.resolve_term(term, alias_maker, for_graph)
                # Add conditions from target terms
                #setattr(quadmap, term_index, arg)
        alias_maker.add_conditions(quadmap.condition_set)

    def add_superclass_path(self, column, cls, alias_maker):
        path = []
        for i, sup in enumerate(sqla_inheritance_with_conditions(cls)):
            # if getattr(inspect(sup), 'local_table', None) is None:
            #     continue
            condition = inspect(cls).inherit_condition
            if condition is not None:
                alias_maker.add_condition(condition)
            if i:
                path.append(SuperClassRelationship(sup, cls))
            cls = sup
            alias_maker.alias_from_relns(*path)
            if _columnish(column):
                local_keys = {c.key for c in inspect(cls).local_table.columns}
                if column.key in local_keys:
                    column = getattr(cls, column.key)
                    return column
            elif _propertish(column):
                if isinstance(column, InstrumentedAttribute):
                    column = column.impl.parent_token
                if column.parent == inspect(cls):
                    return column
            else:
                assert False, "what is this column?"
        else:
            assert False, "The column is found in the "\
                "class and not in superclasses?"

    def include_foreign_conditions(self, dest_class_path):
        return True

    def resolve_term(self, term, alias_maker, for_graph):
        # Options:
        # myclass.column
        # otherclass.column -> Calculate natural join
        # alias(othercls, DeferredPath).column
        # When other class involved, use that classe's base_condition.
        """Columns defined on superclass may come from another table.
        Here we calculate the necessary joins.
        Also class identity conditions for single-table inheritance
        """
        if isinstance(term, string_types + (int, Identifier)):
            return {}, term
        # TODO: Allow DeferredPaths.
        # Create conditions from paths, if needed
        # Create a variant of path link to allow upclass?
        if isinstance(term, GroundedClassAlias):
            return term
        assert _columnish(term), term.__class__.__name__
        column = term
        cls_or_alias = self.get_column_class(column)
        cls = cls_or_alias.path.final_class if isinstance(cls_or_alias, GroundedClassAlias) else cls_or_alias
        # we may be a new clone, and not have that alias.
        # Should it be copied first?
        if (isinstance(term, InstrumentedAttribute)
                and isinstance(term.class_, GroundedClassAlias)
                and cls != term.class_.path.root_cls):
            # This alias was reapplied to the aliasmaker;
            # Rebuild conditions from the path.
            # Alternative design: store conditions in path?
            alias_maker.add_alias(term.class_)
            alias_maker.add_conditions_for_path(term.class_.path)
        local_keys = {c.key for c in inspect(cls).local_table.columns}
        if (getattr(cls, column.key, None) is not None
                and column.key not in local_keys):
            column = self.add_superclass_path(column, cls_or_alias, alias_maker)
        foreign_keys = getattr(column, 'foreign_keys', ())
        dest_class = None
        dest_class_path = None
        for foreign_key in foreign_keys:
            # Do not bother with inheritance here
            if foreign_key.column in inspect(cls).primary_key:
                continue
            dest_class = self.get_column_class(foreign_key.column)
            if dest_class:
                # find a relation
                orm_reln = [r for r in inspect(cls).relationships
                            if column in r.local_columns]
                if len(orm_reln) == 1:
                    dest_class_path = list(orm_reln)
                    break
        if dest_class_path and self.include_foreign_conditions(
                GroundedPath(cls_or_alias, *dest_class_path)):
            relative_am = alias_maker.relative(dest_class_path)
            other_conditions = self.get_base_conditions(
                relative_am, dest_class, for_graph)
            if other_conditions:
                alias = alias_maker.alias_from_relns(*dest_class_path)
                alias_maker.add_alias(alias)
                # Should I add the relation's join condition? I think so.
                reln = dest_class_path[0]
                for join_name in ('primaryjoin', 'secondaryjoin'):
                    join_cond = getattr(reln, join_name)
                    if join_cond is not None:
                        alias_maker.add_condition(join_cond)
                # OR add to relative? need to rebase?
                alias_maker.add_conditions(other_conditions)
        # alias_maker.add_term(term)
        return term


class GatherAliasVisitor(ClauseVisitor):
    def __init__(self, class_reg=None):
        super(GatherAliasVisitor, self).__init__()
        self.aliases = set()
        self.class_reg = class_reg

    def visit_column(self, column):
        cls = _get_column_class(column, self.class_reg)
        if cls and isinstance(cls, AliasedClass):
            self.aliases.add(cls)

    def visit_alias(self, alias):
        self.aliases.add(alias)

    def get_aliases(self):
        return self.aliases


class AliasSetORMAdapter(ClauseAdapter):

    def __init__(self, alias_set):
        super(AliasSetORMAdapter, self).__init__(None)
        self.alias_set = alias_set

    magic_flag = True

    def _corresponding_column(self, col, require_embedded,
                              _seen=EMPTY_SET):
        if not isinstance(col, Column):
            print("_corresponding_column not a Column: ", col)
            return
        alias = self.alias_set.alias_from_table(col.table, col.key)
        if alias:
            return getattr(alias, col.key)
        # else:
        #     import pdb
        #     pdb.set_trace()
        #     # try again...
        #     self.alias_set.alias_from_table(col.table, col.key)


class ConditionSet(object):
    def __init__(self, initial_conditions=None):
        self._conditions = {}
        self._sig = None
        if initial_conditions is not None:
            self.add_conditions(initial_conditions)

    @classmethod
    def condition_signature(cls, condition):
        # TODO: remove alias numbers.
        return _sig(condition)

    def clone(self):
        # Shallow clone
        return ConditionSet(self._conditions.values())

    def update(self, other):
        self._conditions.update(other._conditions)
        self._sig = None

    def add_conditions(self, conditions):
        # This may actually receive a single condition.
        if isinstance(conditions, (list, tuple)):
            for c in conditions:
                self.add_condition(c)
            return
        elif isinstance(conditions, ConditionSet):
            # this should not happen...
            for c in conditions._conditions.itervalues():
                self.add_condition(c)
        else:
            self.add_condition(conditions)

    def add_condition(self, condition):
        sig = self.condition_signature(condition)
        if sig not in self._conditions:
            self._conditions[sig] = condition
            self._sig = None

    def __str__(self):
        if self._sig is None:
            sigs = self._conditions.keys()
            sigs.sort()
            self._sig = ' AND '.join(sigs)
        return self._sig

    @property
    def condition(self):
        if self._conditions:
            return and_(*self._conditions.values())

    def as_list(self):
        return self._conditions.values()

    def __hash__(self):
        return hash(str(self))

    def __nonzero__(self):
        return bool(self._conditions)

    def __cmp__(self, other):
        if self.__class__ != other.__class__:
            return cmp(self.__class__, other.__class__)
        return cmp(str(self), str(other))

    def __equals__(self, other):
        if self.__class__ != other.__class__:
            return cmp(self.__class__, other.__class__)
        return str(self) == str(other)


class GroundedConditionSet(ConditionSet):
    def __init__(self, cls, initial_conditions=None):
        self.root_cls = cls
        super(GroundedConditionSet, self).__init__(initial_conditions)


class GroundedPath(object):
    def __init__(self, *relations):
        assert len(relations)
        self.root_cls = None
        if isinstance(relations[0], type):
            relations = list(relations)
            self.root_cls = relations.pop(0)
            self.path = [r.prop if isinstance(r, InstrumentedAttribute) else r
                         for r in relations]
        elif isinstance(relations[0], self.__class__):
            self.path = relations[0].path
            self.root_cls = relations[0].root_cls
        else:
            self.path = [r.prop if isinstance(r, InstrumentedAttribute) else r
                         for r in relations]
        assert all((isinstance(r, RelationshipProperty) for r in self.path))
        if not self.root_cls:
            self.root_cls = relations[0].parent.class_
        if len(self.path):
            self.validate()
            self.final_class = self.path[-1].mapper.class_
        else:
            self.final_class = self.root_cls

    def validate(self):
        origin = self.root_cls
        for r in self.path:
            assert issubclass(origin, r.parent.class_)
            origin = r.mapper.class_

    def append(self, relation):
        if isinstance(relation, InstrumentedAttribute):
            relation = relation.prop
        assert isinstance(relation, RelationshipProperty)
        assert issubclass(self.final_class, relation.parent.class_)
        self.path.append(relation)
        self.final_class = relation.mapper.class_
        return self

    def extend(self, path_or_relations):
        if (len(path_or_relations) == 1
                and isinstance(path_or_relations[0], type)
                and issubclass(path_or_relations[0], self.__class__)):
            path = path_or_relations[0]
            assert issubclass(self.final_class, path.root_cls)
            self.path.extend(path.path)
            self.final_class = path.final_class
        else:
            for reln in path_or_relations:
                self.append(reln)
        return self

    def clone(self):
        return self.__class__(self.root_cls, *self.path)

    def path_signature(self):
        return self.signature_of()

    def signature_of(self, path=None):
        if path:
            mypath = self.path[:]
            mypath.extend(path)
            path = mypath
        else:
            path = self.path
        return "__".join(chain(
            (self.root_cls.__name__,), (r.key for r in path)))

    def __str__(self):
        return self.path_signature()

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.path_signature())

    def __hash__(self):
        return hash(str(self))

    def __len__(self):
        return len(self.path)

    def len_no_super(self):
        return len([p for p in self.path
                    if not isinstance(p, SuperClassRelationship)])

    def __cmp__(self, other):
        if self.__class__ != other.__class__:
            return 1
        return cmp(str(self), str(other))


class AliasMaker(GroundedPath):
    """This represents a coherent set of conditions with aliases.
    It owns a ConditionSet.
    """
    def __init__(self, cls, cpe, parent=None, path_from_root=None,
                 conditions=None):
        super(AliasMaker, self).__init__(cls)
        if path_from_root:
            self.extend(path_from_root)
        self.cpe = cpe
        self.parent = parent
        cls = self.final_class
        self.base_alias = GroundedClassAlias(
            cls, GroundedPath(cls), name=self.name_for_alias(self.path))
        self.aliases_by_path = {}
        self.aliases_by_name = {}
        self.add_alias(self.base_alias, path=GroundedPath(cls))
        if conditions:
            self.conditions = conditions.clone()
        else:
            self.conditions = GroundedConditionSet(cls)

    def base_alias(self):
        return self.base_alias

    def name_for_alias(self, path):
        return "alias_"+self.signature_of(path).lower()

    def get_column_class(self, col, use_annotations=True):
        return self.cpe.get_column_class(col, use_annotations)

    def add_alias(self, alias, name=None, path=None):
        name = name or alias.get_name()
        path = path or alias.path
        if path not in self.aliases_by_path:
            self.aliases_by_path[path] = alias
            self.aliases_by_name[name] = alias

        assert len(self.aliases_by_name) >= len(self.aliases_by_path)

    def add_conditions_for_path(self, grounded_path):
        self.add_conditions_for_relationships(*grounded_path.path)

    def add_conditions_for_relationships(self, *relns):
        for reln in relns:
            self.add_conditions_for_relationship(reln)

    def add_conditions_for_relationship(self, reln):
        assert isinstance(reln, RelationshipProperty)
        if isinstance(reln, SuperClassRelationship):
            condition = reln.parent.inherit_condition
            if condition:
                # assume alias added independently
                self.add_condition(condition, False)
        else:
            if reln.primaryjoin is not None:
                self.add_condition(reln.primaryjoin, False)
            if reln.secondaryjoin is not None:
                self.add_condition(reln.secondaryjoin, False)

    def adapter(self):
        adapter = None
        for alias in self.aliases_by_path.itervalues():
            adapter = ORMAdapter(alias).chain(adapter)
        return adapter

    def aliased_term(self, term=None):
        assert term is not None
        #term = term if term is not None else self.term
        if isinstance(term, Visitable):
            return self.adapter().traverse(term)
        elif _propertish(term):
            # Could I just use alias_from_relns?
            return self.get_reln_alias(term)
        elif _columnish(term):
            return self.get_column_alias(term)
        else:
            assert False, term

    def get_column_alias(self, column):
        if isinstance(column, InstrumentedAttribute):
            parent = column._parententity
            if (isinstance(parent, AliasedInsp)
                    and isinstance(parent.entity, GroundedClassAlias)):
                assert parent.entity.get_name() in self.aliases_by_name, \
                    "column %s not in known aliases" % column
                return getattr(self.aliases_by_name[parent.entity.get_name()],
                               column.key)
            column = getattr(column._parententity.c, column.key)
        for alias in self.aliases_by_path.itervalues():
            # TODO: What if there's many?
            if inspect(alias).mapper.local_table == column.table:
                return getattr(alias, column.key)
        assert False, "column %s not in known aliases" % column

    def get_reln_alias(self, relationship):
        if isinstance(relationship, InstrumentedAttribute):
            parent = relationship._parententity
            if (isinstance(parent, AliasedInsp)
                    and isinstance(parent.entity, GroundedClassAlias)
                    and parent.entity.get_name() in self.aliases_by_name):
                return getattr(self.aliases_by_name[parent.entity.get_name()],
                               relationship.key)
            r = getattr(relationship._parententity.c, relationship.key, None)
            if not r:
                r = getattr(relationship._parententity.relationships, relationship.key, None)
            assert r
            relationship = r
        for alias in self.aliases_by_path.itervalues():
            # TODO: What if there's many?
            if inspect(alias).mapper.local_table == relationship.table:
                return alias
        assert False, "relationship %s not in known aliases" % relationship

    @staticmethod
    def get_alias_name(alias):
        return inspect(alias).selectable.name

    def alias_from_relns(self, *relns):
        return self.alias_from_path(GroundedPath(self.final_class, *relns))

    def alias_from_path(self, path):
        if path not in self.aliases_by_path:
            fullpath = GroundedPath(self.root_cls, *self.path)
            fullpath.extend(path.path)
            self.add_alias(GroundedClassAlias(
                path.final_class, fullpath, name="alias_"+str(fullpath).lower()))
        return self.aliases_by_path[path]

    def alias_from_join(self, query):
        raise NotImplementedError()

    def alias_from_class(self, cls, col_key=None, add_conditions=False):
        if cls == self.root_cls:
            return self.base_alias
        aliases = [a for a in self.aliases_by_path.itervalues()
                   if issubclass(a.get_class(), cls)]
        if len(aliases):
            aliases.sort(key=lambda a: (a.path.len_no_super(), len(a.path)))
            exact = [a for a in aliases if a.get_class() == cls]
            if exact:
                return exact[0]
            # take shortest
            alias = aliases[0]
            path = alias.path.clone()
            last_class = alias.path.final_class
            inherit_conditions = []
            for i, sup in enumerate(sqla_inheritance_with_conditions(last_class)):
                if i:
                    path.append(SuperClassRelationship(sup, last_class))
                c = inspect(sup).inherit_condition
                if c is not None:
                    inherit_conditions.append(c)
                if sup == cls:
                    break
                last_class = sup
            alias = self.alias_from_path(path)
            if add_conditions:
                self.add_alias(alias)
                self.add_conditions(inherit_conditions)
            return alias
        # otherwise guess natural join
        known_aliases = self.aliases_by_path.values()
        known_aliases.sort(key=lambda a: a.path.len_no_super())
        last_len = -1
        found_paths = []
        for a in known_aliases:
            this_len = a.path.len_no_super()
            if found_paths and this_len > last_len:
                break
            last_len = this_len
            rels = self.guess_joins(a.path.final_class, cls, col_key)
            for rel in rels:
                found_paths.append(a.path.clone().append(rel))
        if not found_paths:
            print("ERROR: No path for ", cls, col_key)
            return
        #print found_paths
        if len(found_paths) > 1:
            found_paths.sort(key=lambda path: len(path))
            print("WARNING: Too many paths for ", cls, col_key, found_paths)
        alias = self.alias_from_path(found_paths[0])
        if add_conditions:
            # This will happen later and loop...
            self.add_alias(alias)
            reln = found_paths[0].path[-1]
            if reln.primaryjoin is not None:
                self.add_condition(reln.primaryjoin)
            if reln.secondaryjoin is not None:
                self.add_condition(reln.secondaryjoin)
        return alias

    def guess_joins(self, source_cls, dest_cls, col_key=None):
        relns = [r for r in inspect(source_cls).relationships
                 if issubclass(dest_cls, r.mapper.class_)]
        if col_key is not None:
            relns2 = [
                r for r in relns if col_key in {
                    c.key for c in r.local_columns}]
            if relns2:
                return relns2
        return relns

    def alias_from_foreign_key_column(self, column):
        # guess natural join... can I turn it into a path?
        raise NotImplementedError()

    def relative(self, relns):
        # Do we want the parent to be absolute or relative?
        parent = self.parent or self
        path = self.path[:]
        path.extend(relns)
        return AliasMaker(self.root_cls, self.cpe, parent, path)

    def freeze(self, uid):
        return AliasSet(self, uid)

    def add_condition_aliases(self, condition):
        g = GatherColumnsVisitor(self.cpe.class_reg)
        g.traverse(self.conditions.condition)
        for col in g.columns:
            alias = self.alias_from_table(col.table, col.key, True)
            if alias:
                self.add_alias(alias)

    def alias_from_table(self, table, col_key=None, add_conditions=False):
        if isinstance(table, GroundedClassAlias):
            return table
        cls = None
        if isinstance(table, Alias):
            name = getattr(table, 'name', None)
            if name in self.aliases_by_name:
                return self.aliases_by_name[name]
            table = table.element
            assert isinstance(table, Table)
            cls = _get_class_from_table(table, self.cpe.class_reg)
        elif isinstance(table, AliasedClass):
            name = table._aliased_insp.name
            if name in self.aliases_by_name:
                return self.aliases_by_name[name]
            cls = table._aliased_insp._target
        elif isinstance(table, Table):
            cls = _get_class_from_table(table, self.cpe.class_reg)
        else:
            assert cls.__mapper__ is not None
            cls = table
        assert cls == self.cpe.class_reg[cls.__name__]
        return self.alias_from_class(cls, col_key, add_conditions)

    def add_condition(self, condition, with_aliases=True):
        self.conditions.add_condition(condition)
        if with_aliases:
            self.add_condition_aliases(condition)

    def add_conditions(self, conditions, with_aliases=True):
        self.conditions.add_conditions(conditions)
        if with_aliases:
            if isinstance(conditions, ConditionSet):
                conditions = conditions.as_list()
            for c in conditions:
                self.add_condition_aliases(c)

    def clone(self):
        clone = AliasMaker(
            self.root_cls, self.cpe, self.parent, self.path[:],
            self.conditions)
        clone.aliases_by_path.update({
            k: v for (k, v) in self.aliases_by_path.iteritems()
            if v != self.base_alias
        })
        clone.aliases_by_name.update({
            k: v for (k, v) in self.aliases_by_name.iteritems()
            if v != self.base_alias
        })
        conditions = {p: clone.aliased_term(c)
                      for p, c in self.conditions._conditions.iteritems()}
        clone.conditions._conditions.update(conditions)
        return clone

    def __eq__(self, other):
        return self is other

    def signature(self):
        return self.path_signature() + str(self.conditions)

    @property
    def aliases(self):
        return self.aliases_by_path.itervalues()

    def __len__(self):
        return 1   # overshadow grounded path truth value


class AliasSet(AliasMaker):
    """Immutable AliasMaker"""
    def __init__(self, orig, uid):
        self.uid = uid
        super(AliasSet, self).__init__(
            orig.root_cls, orig.cpe, orig.parent, orig.path)
        # ALSO: Replace aliases in conditions with frozen versions
        self.aliases_by_name[orig.base_alias.get_name()] = self.base_alias
        for path, alias in orig.aliases_by_path.iteritems():
            if not len(path):
                continue  # We already have base_alias
            # use old names
            self.add_alias(alias.freeze(uid), alias.get_name())
        conditions = {p: self.aliased_term(c)
                      for p, c in orig.conditions._conditions.iteritems()}
        self.conditions._conditions.update(conditions)

    def adapter(self):
        return AliasSetORMAdapter(self)

    def name_for_alias(self, path):
        return super(AliasSet, self).name_for_alias(path) + "_" + str(self.uid)

    def add_condition(self, condition):
        raise RuntimeError("AliasSet's condition should be immutable")

    def add_conditions(self, condition):
        raise RuntimeError("AliasSet's condition should be immutable")


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
