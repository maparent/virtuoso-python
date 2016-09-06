import re
from collections import OrderedDict

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.inspection import inspect
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.util import AliasedInsp
from sqlalchemy.schema import Column
from sqlalchemy.sql.expression import (
    ClauseElement, FunctionElement, TextClause)
from sqlalchemy.sql.visitors import Visitable
from sqlalchemy.types import TypeEngine
from rdflib import URIRef
from future.utils import string_types

from virtuoso.quadextractor import (
    GroundedClassAlias, GatherColumnsVisitor, ConditionSet)


class RdfLiteralStmt(ClauseElement):
    def __init__(self, literal, nsm=None):
        self.nsm = nsm
        self.literal = literal

    def __bool__(self):
        # Avoid many tedious "is not None"
        return True

    def _compile(self, compiler, **kwargs):
        nsm = kwargs.get('nsm', self.nsm)
        return self.literal.n3(nsm)

compiles(RdfLiteralStmt)(RdfLiteralStmt._compile)


class Mapping(object):
    def __init__(self, name, nsm=None):
        self.name = name
        self.nsm = nsm

    def patterns_iter(self):
        return ()

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
                    raise ValueError("Please provide class: " + cls)
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
            # return TextClause(arg.n3(self.nsm))
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


class ApplyFunction(Mapping, FunctionElement):
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


class AbstractFunction(Mapping):
    def __init__(self, name, nsm=None, *arguments):
        super(AbstractFunction, self).__init__(name, nsm)
        self.arguments = tuple(arguments)

    def patterns_iter(self):
        yield self

    def apply(self, *arguments):
        return ApplyFunction(self, self.nsm, *arguments)

    def definition_statement(self):
        return None


class IriClass(AbstractFunction):
    pass


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

    def __init__(self, subject=None, predicate=None, obj=None,
                 graph_name=None, name=None, conditions=None, nsm=None):
        super(QuadMapPattern, self).__init__(name, nsm)
        self.graph_name = graph_name
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.condition_set = ConditionSet(conditions)
        self.alias_set = None

    def and_condition(self, condition):
        self.condition_set.add_condition(condition)

    def and_conditions(self, conditions):
        self.condition_set.add_conditions(conditions)

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

    def __init__(self, graph_iri, name=None, nsm=None):
        super(GraphQuadMapPattern, self).__init__(name, nsm)
        self.iri = graph_iri
        self.qmps = set()

    def known_submaps(self):
        return self.qmps

    def patterns_iter(self):
        for qmp in self.qmps:
            for pat in qmp.patterns_iter():
                yield pat

    def add_patterns(self, base_class, patterns):
        for pattern in patterns:
            self.add_pattern(base_class, pattern)

    def add_pattern(self, base_class, pattern):
        assert self.nsm
        assert isinstance(pattern, QuadMapPattern)
        if pattern not in self.qmps:
            pattern.set_namespace_manager(self.nsm)
            self.qmps.add(pattern)

