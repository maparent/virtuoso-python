from setuptools import setup, find_packages
import sys
import os

from pip.download import PipSession
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=PipSession())
requires = [str(ir.req) for ir in install_reqs]

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
      install_requires=requires,
      entry_points="""
          [sqlalchemy.dialects]
          virtuoso = virtuoso:alchemy.VirtuosoDialect

          [rdf.plugins.store]
          Virtuoso = virtuoso:vstore.Virtuoso
      """,
      )
