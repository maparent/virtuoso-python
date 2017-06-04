from future import standard_library
standard_library.install_aliases()
from ConfigParser import ConfigParser
from os.path import dirname, join, expanduser
from os import putenv
import decimal


putenv('ODBCINI', expanduser('~/.odbc.ini'))

config = ConfigParser()
config.read(join(dirname(dirname(dirname(__file__))), 'setup.cfg'))

sqla_connection = config.get('tests', 'sqla-connection')
rdflib_connection = config.get('tests', 'rdflib-connection')
