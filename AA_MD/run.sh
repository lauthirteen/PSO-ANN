gmx grompp -f minim.mdp -c init.gro -p top.top -o em.tpr -maxwarn 5
gmx mdrun -v -deffnm em -nt 6
gmx grompp -f md.mdp -c em.gro -p top.top -o md.tpr -maxwarn 5
gmx mdrun -v -deffnm md -nt 6
rm ./#*

