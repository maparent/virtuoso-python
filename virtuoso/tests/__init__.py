from ConfigParser import ConfigParser
from os.path import dirname, join

config = ConfigParser()
config.read(join(dirname(dirname(dirname(__file__))), 'setup.cfg'))

sqla_connection = config.get('tests', 'sqla-connection')
rdflib_connection = config.get('tests', 'rdflib-connection')
