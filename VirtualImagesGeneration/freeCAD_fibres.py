FREECADPATH="C:/Program Files (x86)/FreeCAD0.13/bin/" # adapt this path to your system
import sys
if not("FreeCAD" in dir()): 
	sys.path.append(FREECADPATH)

import FreeCAD
from FreeCAD import Base, Part
import math, random

for i in range(1):
	origin=FreeCAD.Vector(random.random(),random.random()  ,random.random())
	end=FreeCAD.Vector(random.random(),random.random()  ,random.random())
	radius=0.01
	heigth=(end-origin).Length
	#creation des objets
	cylinder=Part.makeCylinder(radius,heigth,origin,end-origin)
#	sphere1=Part.makeSphere(radius,origin)
#	sphere2=Part.makeSphere(radius,end)
#	cylinder=cylinder.fuse(sphere1)
#	cylinder=cylinder.fuse(sphere2)
	triangles = cylinder.tessellate(1)
	faces = []
	for tri in triangles[1]:
    		face = []
    		for j in range(3):
        		vindex = tri[j]
       		face.append(triangles[0][vindex])
   		faces.append(face)	

	print(faces)

	#Part.show(cylinder)
	#ecriture des fichiers stl
	filename="/home/greentoto/Documents/MATLAB/creation_structures_3D/Scripts/"+"avirer"+str(i)+".stl"
	print(filename)
	#print filename
	#cylinder.exportStl(filename)
	#scipy.io.savemat

