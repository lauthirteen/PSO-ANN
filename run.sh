#!/usr/bin/env bash
rm ./#* *.xvg *.edr *.xtc *.trr *.tpr
gmx grompp -f minim.mdp -c init.gro -p top.top -o em.tpr -maxwarn 5
gmx mdrun -v -deffnm em -nt 6
gmx grompp -f md.mdp -c em.gro -p top.top -o md.tpr -maxwarn 5
gmx mdrun -v -deffnm md -nt 6
gmx energy -f md.edr -o den.xvg -b 100 <<< Density >& den.log
gmx make_ndx -f md.tpr < index
echo -e 0 ' \n 1 ' | gmx rdf -f md.xtc -s md.tpr -n index.ndx -b 100 -o OW_spc_OW_spc.xvg
echo done > tag_finished
