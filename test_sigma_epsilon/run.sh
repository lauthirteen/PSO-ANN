#!/usr/bin/env bash
gmx make_ndx -f md.tpr < index
echo -e 0 1 | gmx rdf -f md.xtc -s md.tpr -n index.ndx -b 1500 -o OW_spc_OW_spc.xvg
