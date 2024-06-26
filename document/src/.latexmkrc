# Build PGF images from Python scripts, and add any dependencies
# and generated images to latexmk's database.
add_cus_dep('py', 'pypgf', 0, 'pypgf');
sub pypgf {
    # Run the script and ask pgfutils to give us a list
    # of all dependencies and extra generated files.
    my @tracked = `PGFUTILS_TRACK_FILES=1 python3 $_[0].py`;

    # Process the tracked files.
    foreach (@tracked){
        my ($mode, $fn) = /(.):(.+)/;

        # Files opened for reading: dependency.
        if($mode eq "r"){
            rdb_ensure_file($rule, $fn);
        }

        # Opened for writing: generated in addition to the .pypgf file.
        elsif($mode eq "w"){
            rdb_add_generated($fn);
        }
    }
}
@default_files = ("main.tex");
$bibtex_use = 2;
$pdf_mode = 1;
$out_dir = 'build';
$cleanup_includes_cusdep_generated = 1;
