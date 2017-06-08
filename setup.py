from setuptools import setup, find_packages
import sys
import os

version = '0.12.6'
try:
    from mercurial import ui, hg, error
    repo = hg.repository(ui.ui(), ".")
    ver = repo[version]
except ImportError:
    pass
except error.RepoLookupError:
    tip = repo["tip"]
    version = version + ".%s.%s" % (tip.rev(), tip.hex()[:12])
except error.RepoError:
    pass


def readme():
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, "README.txt")
    return open(filename).read()

setup(name='virtuoso',
      version=version,
      description="OpenLink Virtuoso Support for SQLAlchemy and RDFLib",
      long_description=readme(),
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='',
      author='Marc-Antoine Parent and Open Knowledge Foundation',
      author_email='maparent@acm.org, okfn-help@okfn.org',
      url='http://packages.python.org/virtuoso',
      license='BSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      tests_require=["nose"],
      requires=[
          'SQLAlchemy',
          'pyodbc',
          'rdflib',
          'uricore'
      ],
      dependency_links=[
          'https://pypi.python.org/packages/source/S/SQLAlchemy/SQLAlchemy-0.9.8.tar.gz#egg=SQLAlchemy',
          'http://github.com/maparent/pyodbc/tarball/v3-virtuoso#egg=pyodbc',
          'https://pypi.python.org/packages/source/u/uricore/uricore-0.1.2.tar.gz#egg=uricore'
      ],
      entry_points="""
          [sqlalchemy.dialects]
          virtuoso = virtuoso:alchemy.VirtuosoDialect

          [rdf.plugins.store]
          Virtuoso = virtuoso:vstore.Virtuoso
      """,
      )
