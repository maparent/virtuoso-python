To reproduce the deadlock...

0. make sure you have python development headers installed 
    as well as the "pip" and "virtualenv" packages, and have iodbc
    set up with a reasonable odbc.ini that defines a database
    called VOS

1. make a testing environment and activate it

    virtualenv deadlock; . ./deadlock/bin/activate

2. install the patched pyodbc:

    pip install http://eris.okfn.org/ww/2010/10/pyodbc-spasql/pyodbc-virtuoso-2.1.9-beta14.tar.gz

3. install the test runner:

    pip install nose

4. install rdflib:

    pip install rdflib

5. clone and install the virtuoso store implementation for rdflib

    hg clone https://bitbucket.org/ww/virtuoso
    cd virtuoso
    python setup.py install

6. run the rdflib tests:

    nosetests -v -s virtuoso/tests/test_rdflib3.py

7. watch as the last test deadlocks. this test does something very simple,
    it just reads an rdf/xml fixture, puts it in the store and then reads it out
    again, in each case one triple at a time.
