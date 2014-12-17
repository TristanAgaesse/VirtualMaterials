import os
import FreeCAD, FreeCADGui, Part ,Mesh 
from FreeCAD import Base
import math, random

def readMatrix(filename,elementType):
  f = open ( filename , 'r')
  if elementType is 'float':
    l = [ map(float,line.split(' ')) for line in f ]
  else :
    l = [ map(int,line.split(' ')) for line in f ]
  return l
  f.close()
              
currentDir="H:/Documents/MATLAB/creation_structures_3D/Scripts/Demokritos_Voronoi_anisotrope/Isotrope/1000pores"
            
fibres=readMatrix(currentDir+"/fibres.txt","int")
radius=readMatrix(currentDir+"/radius.txt","float")
points=readMatrix(currentDir+"/points.txt","float")
vertices=readMatrix(currentDir+"/vertices.txt","int")

nFibre=len(fibres)
print nFibre
nVertice=len(vertices)
print nVertice

for iVertice in range(nVertice):
  thisRadius=400e-5
  iPoint=vertices[iVertice][1]-1
  originPoint=points[iPoint]
  origin=FreeCAD.Vector(100*originPoint[0],100*originPoint[1],100*originPoint[2])
  sphere=Part.makeSphere(thisRadius,origin)
  filename=currentDir+"/fichiers_stl/"+"sphere"+str(iVertice)+".stl"
  sphere.exportStl(filename)
#  try:
#    sphere.exportStl(filename)
#  except:
#    print iVertice


for iFibre in range(nFibre):
  iPoint1=fibres[iFibre][1]
  iPoint2=fibres[iFibre][2]
  point1=points[iPoint1]
  point2=points[iPoint2]
  thisRadius=400*radius[iPoint1][0]
  origin=FreeCAD.Vector(100*point1[0],100*point1[1],100*point1[2])
  end=FreeCAD.Vector(100*point2[0],100*point2[1],100*point2[2])
  heigth=(end-origin).Length
  #creation des objets
  cylinder=Part.makeCylinder(thisRadius,heigth,origin,end-origin)
  filename=currentDir+"/fichiers_stl/"+"fibre"+str(iFibre)+".stl"
  #Part.show(cylinder)
  cylinder.exportStl(filename)
