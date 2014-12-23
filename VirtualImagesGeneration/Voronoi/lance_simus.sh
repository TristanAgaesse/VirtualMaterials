#!/bin/bash
for i in $(ls -d */ )
 do 
	echo $i
	cd /home/greentoto/Documents/MATLAB/PNM_Matlab/Simulations/Demokritos/$i
	python createFibreuxIsotrope.py
	cd /home/greentoto/Documents/MATLAB/PNM_Matlab/Simulations/Demokritos/
 done


