#!/usr/bin/env bash
cd code
for year in {2013..2017}; do
    python NIPS_crawl.py $year ../data/nips_2013-2017/$year
    mkdir -p ../data/nips_2013-2017/$year/parsed_pdfs/
    for pdf_filename in $( ls ../data/nips_2013-2017/$year/pdfs/*.pdf ); do
      java -Xmx6g -jar lib/science-parse-cli-assembly*.jar $pdf_filename > ../data/nips_2013-2017/$year/parsed_pdfs/$(basename ${pdf_filename}).json
    done
done