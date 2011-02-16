from setuptools import setup, find_packages
import sys, os

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
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='',
      author='Open Knowledge Foundation',
      author_email='okfn-help@okfn.org',
      url='http://packages.python.org/virtuoso',
      license='BSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
          "pyodbc==virtuoso-2.1.9-beta14",
      ],
      entry_points="""
          [sqlalchemy.dialects]
          virtuoso = virtuoso:alchemy.VirtuosoDialect

          [rdf.plugins.store]
          Virtuoso = virtuoso:vstore.Virtuoso
      """,
      )
