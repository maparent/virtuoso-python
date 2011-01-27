drop procedure NQ_DUMP;
create procedure NQ_DUMP(in dir varchar := 'dumps', in file_length_limit integer := 100000000)
{
    declare file_name varchar;
    declare buf, tmpbuf any;
    declare buf_len, max_buf_len, file_len, file_idx integer;
    set isolation = 'uncommitted';
    max_buf_len := 1000000;
    file_len := 0;
    file_idx := 1;
    file_name := sprintf ('%s/dump_%06d.nq', dir, file_idx);
    buf := string_output();

    for (select * from (sparql define input:storage "" select ?s ?p ?o ?g where { graph ?g { ?s ?p ?o } }) as sub option(loop)) do {
        -- subject
        tmpbuf := string_output();
        http_nt_object("s", tmpbuf);
	tmpbuf := string_output_string(tmpbuf);
	if (starts_with(tmpbuf, 'b')) {
	    http('_:', buf);
        }
        http(tmpbuf, buf);
	http(' ', buf);

	-- predicate (cannot be blank node)
        http_nt_object("p", buf);
        http(' ', buf);

	-- object
        tmpbuf := string_output();
        http_nt_object("o", tmpbuf);
	tmpbuf := string_output_string(tmpbuf);
	if (starts_with(tmpbuf, 'b')) {
	    http('_:', buf);
        }
        http(tmpbuf, buf);
	http(' ', buf);

	-- graph (cannot be blank node)
        http_nt_object("g", buf);
	http(' .\n', buf);

        buf_len := length (buf);
        if (buf_len > max_buf_len) {
            file_len := file_len + buf_len;
            if (file_len > file_length_limit) {
                file_len := 0;
                file_idx := file_idx + 1;
                file_name := sprintf('%s/dump_%06d.nq', dir, file_idx);
            }
            string_to_file(file_name, buf, -1);
            buf := string_output ();
        }
    }
    if (length(buf) > 0) {
        string_to_file(file_name, buf, -1);
    }
};

DROP PROCEDURE NQ_LOAD;
CREATE PROCEDURE NQ_LOAD (in dir varchar := 'dumps/')
{
    declare arr any;

    arr := sys_dirlist (dir, 1);
    log_enable (2, 1);
    foreach (varchar f in arr) do {
        if (f like '*.nq') {
            declare continue handler for sqlstate '*' {
                log_message (sprintf ('Error in %s', f));
            };
	    -- flags
	    -- 1 - Single quoted and double quoted strings may with newlines.
	    -- 2 - Allows bnode predicates (but SPARQL processor may ignore them!).
	    -- 4 - Allows variables, but triples with variables are ignored.
	    -- 8 - Allows literal subjects, but triples with them are ignored.
	    -- 16 - Allows '/', '#', '%' and '+' in local part of QName ("Qname with path")
	    -- 32 - Allows invalid symbols between '<' and '>', i.e. in relative IRIs.
	    -- 64 - Relax TURTLE syntax to include popular violations.
	    -- 128 - Try to recover from lexical errors as much as it is possible.
	    -- 256 - Allows TriG syntax, thus loading data in more than one graph.
	    -- 512 - Allows loading N-quad dataset files with and optional context value to 
	    --           indicate provenance as detailed
	    --
	    -- we use 1 | 2 | 4 | 8 | 128 | 512 = 655

            DB.DBA.TTLP_MT (file_open (dir || '/' || f), '', 'http://example.org/', 655, 1);
        }
    }
    exec ('checkpoint');
};