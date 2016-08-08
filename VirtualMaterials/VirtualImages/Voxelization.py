# -*- coding: utf-8 -*-

import numpy as np
import vtk
import math
import VirtualMaterials as vmat
#from vtk.util import numpy_support
#from numba import jit

#--------------------------------------------------------------------
#      Voxelization
#--------------------------------------------------------------------

    
#--------------------------------------------------------------------
#@jit    
def Voxelize(vtkPolyDataObject,gridX,gridY,gridZ,raydirection='xz'):
    #Voxelize the object on a window adapted to its bounds. The windows will 
    #be chosed to be a subsample or an extention of the whole image voxel 
    #gridX,Y,Z. This function uses VTK VoxelModel to voxelize the surface, 
    #then the FillInside function to fill the inside.

    #Prepare a subwindows zooming on the object
    subWindowBound=vtkPolyDataObject.GetBounds()    
    nVoxSubImage,boundSubgrid,gridRelativePosition = GetSubWindowInformation(
                                                        subWindowBound,gridX,gridY,gridZ)
    
    
    subImage=__VoxelizeRayTracing__(vtkPolyDataObject,nVoxSubImage,boundSubgrid,raydirection)
    

    #Get back to the original window
    nVoxImage = (len(gridX)-1,len(gridY)-1,len(gridZ)-1)    
    wholeImage = InsertSubimageInImage(subImage,nVoxImage,gridRelativePosition)
    
    return wholeImage.astype(np.bool)


#-----------------------------------------------------------------------------
#@jit
def __VoxelizeRayTracing__(vtkPolyDataObject,nVoxSubImage,boundSubgrid,raydirection='z'):

    #Count the number of voxels in each direction:

#    points = numpy_support.vtk_to_numpy(vtkPolyDataObject.GetPoints().GetData())
#    polys = numpy_support.vtk_to_numpy(vtkPolyDataObject.GetPolys().GetData())
#    
#    nTriangle=len(polys)/4
#    meshXYZ=np.zeros((nTriangle,3,3),dtype=np.float)
#    for iTri in range(nTriangle):
#        for iVert in range(3):
#                assert polys[4*iTri]==3
#                meshXYZ[iTri,:,iVert]=points[polys[4*iTri+iVert+1]]
    
    gridCOx=np.linspace(boundSubgrid[0],boundSubgrid[1],nVoxSubImage[0])
    gridCOy=np.linspace(boundSubgrid[2],boundSubgrid[3],nVoxSubImage[1])
    gridCOz=np.linspace(boundSubgrid[4],boundSubgrid[5],nVoxSubImage[2])

    voxcountX,voxcountY,voxcountZ = nVoxSubImage[0],nVoxSubImage[1],nVoxSubImage[2]

    
    # Prepare logical array to hold the voxelised data:
    gridOUTPUT = np.zeros((voxcountX,voxcountY,voxcountZ,len(raydirection)),dtype=np.bool)
    countdirections = 0;
    
    if raydirection.find('x')>-1:
      countdirections = countdirections + 1
      #rotatedMesh = meshXYZ[:,[1,2,0],:]
      rotationCenter=(0,0,0)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateY(vtkPolyDataObject,rotationCenter,-90)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateZ(rotatedMesh,rotationCenter,-90)
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( 
          __VoxelizeRayTracingZDirectionVTK__(rotatedMesh,gridCOy,gridCOz,gridCOx),
                                  axes=[2,0,1] )
    
    #TODO : debug y direction
    if raydirection.find('y')>-1:    
      countdirections = countdirections + 1
      #rotatedMesh = meshXYZ[:,[2,0,1],:]
      rotationCenter=(0,0,0)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateZ(vtkPolyDataObject,rotationCenter,-90)
      rotatedMesh=vmat.VirtualImages.BasicShapes.MeshRotateX(rotatedMesh,rotationCenter,-90)
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( 
          __VoxelizeRayTracingZDirectionVTK__(rotatedMesh,gridCOz,gridCOx,gridCOy),
                                  axes=[1,2,0] )
    
    if raydirection.find('z')>-1:
      countdirections = countdirections + 1
      gridOUTPUT[:,:,:,countdirections-1] = __VoxelizeRayTracingZDirectionVTK__(
                                  vtkPolyDataObject,gridCOx,gridCOy,gridCOz)
    
    # Combine the results of each ray-tracing direction:
    if len(raydirection)>1:
        gridOUTPUT = (np.sum(gridOUTPUT,axis=3)== len(raydirection)) # >=len(raydirection)/2.0)
    else:
        gridOUTPUT = gridOUTPUT[:,:,:,0]
        
    return gridOUTPUT
    


#-----------------------------------------------------------------------------
#@jit
def __VoxelizeRayTracingZDirectionVTK__(polydata,gridCOx,gridCOy,gridCOz):
    # Remark : most of the time is spent in this function. This is because of 
    # the python loop on VTK_Rays. Maybe a loop through a VTK collection of rays 
    # using an iterator object would be faster, because python could recognize
    # the loop should be spent in C++. 
    #See https://blog.kitware.com/pythonic-callbacks-and-iteration-in-vtk/
    
    #Identify the min and max x,y coordinates (cm) of the mesh:
#    meshXmin = meshXYZ[:,0,:].min()
#    meshXmax = meshXYZ[:,0,:].max()
#    meshYmin = meshXYZ[:,1,:].min()
#    meshYmax = meshXYZ[:,1,:].max()
#    meshZmin = meshXYZ[:,2,:].min()
#    meshZmax = meshXYZ[:,2,:].max()
    
    meshXmin,meshXmax,meshYmin,meshYmax,meshZmin,meshZmax = polydata.GetBounds()
    assert (meshXmin<meshXmax and meshYmin<meshYmax and meshZmin<meshZmax)
    
    #Identify the min and max x,y coordinates (pixels) of the mesh:
    meshXminp = np.nonzero(abs(gridCOx-meshXmin)==min(abs(gridCOx-meshXmin)))[0][0]
    meshXmaxp = np.nonzero(abs(gridCOx-meshXmax)==min(abs(gridCOx-meshXmax)))[0][0]
    meshYminp = np.nonzero(abs(gridCOy-meshYmin)==min(abs(gridCOy-meshYmin)))[0][0]
    meshYmaxp = np.nonzero(abs(gridCOy-meshYmax)==min(abs(gridCOy-meshYmax)))[0][0]
        
    #meshXYZmin = np.min(meshXYZ,axis=2)
    #meshXYZmax = np.max(meshXYZ,axis=2)    
        
    sampleDimensions=(len(gridCOx),len(gridCOy),len(gridCOz))
    image=np.zeros(sampleDimensions,dtype=np.bool)
    #Identify the min and max x,y,z coordinates of each facet:
    
    
    #Construct octree of the mesh for fast access to its geometry
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(polydata)
    obbTree.BuildLocator()
    
    #correctionLIST = []  
    
    Yrange=range(meshYminp,meshYmaxp+1)
    Xrange=range(meshXminp,meshXmaxp+1)
    
    correctionLIST = np.zeros( (len(Xrange),len(Yrange)) )    
    
    for loopY in Yrange:
        for loopX in Xrange:
            
            epsilon=(meshZmax-meshZmin)/10
            pSource=[gridCOx[loopX],gridCOy[loopY],meshZmax+epsilon]
            pTarget=[gridCOx[loopX],gridCOy[loopY],meshZmin-epsilon]
            pointsIntersection=__VTKRayCasting__(obbTree,pSource,pTarget)
            
            
            if len(pointsIntersection)>0: 

              #gridCOzCROSS=np.asarray([pointsIntersection[i][2] for i in range(len(pointsIntersection))])
              gridCOzCROSS=pointsIntersection[:,2]
              
              #Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
              gridCOzCROSS = gridCOzCROSS[ np.logical_and(
                  np.greater_equal(gridCOzCROSS,(meshZmin-1e-12)*np.ones(gridCOzCROSS.shape)),
                  np.less_equal(gridCOzCROSS,(meshZmax+1e-12)*np.ones(gridCOzCROSS.shape))) ]
          
              #Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
              gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
              gridCOzCROSS = np.unique(gridCOzCROSS)
    
              if gridCOzCROSS.size%2 == 0: 
                  for loopASSIGN in range(gridCOzCROSS.size/2):
                      voxelsINSIDE = np.logical_and(
                                    np.greater(gridCOz,gridCOzCROSS[2*loopASSIGN]), 
                                    np.less(gridCOz,gridCOzCROSS[2*loopASSIGN+1]))
                                    
                      image[loopX,loopY,voxelsINSIDE] = 1
              elif len(gridCOzCROSS)>0:
                  #correctionLIST.append([loopX,loopY])
                  correctionLIST[loopX-meshXminp,loopY-meshYminp]=1
    
    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
    #For rays where the voxelisation did not give a clear result, the ray is
    #computed by interpolating from the surrounding rays.    

    nzX,nzY=np.nonzero(correctionLIST)
    nzX,nzY = nzX+meshXminp,nzY+meshYminp
    #correctionLIST=[[nzX[i],nzY[i]] for i in range(nzX.size)]
    
    correctionLIST=np.transpose(np.vstack((nzX,nzY)))
    image=__InterpolateRemainingVoxels__(correctionLIST,sampleDimensions,image)
    
    return image





#-----------------------------------------------------------------------------
#@jit
def __VTKRayCasting__(polydataObbTree,pSource,pTarget):
#https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/


#    pSource = [loopX, loopY, +10]
#    pTarget = [loopX, loopY, -10]
    
    pointsVTKintersection = vtk.vtkPoints()
    code = polydataObbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, None)

    #check code
#    if code==0:
#        1
#    elif code==-1:
#        1
#    elif code==1:
#        1
   #"Take the passed line segment and intersect it with the data set. This method 
   #assumes that the data set is a vtkPolyData that describes a closed surface, 
   #and the intersection points that are returned in ‘points’ alternate between 
   #entrance points and exit points. The return value of the function is 0 if no 
   #intersections were found, –1 if point ‘a0’ lies inside the closed surface, or 
   #+1 if point ‘a0’ lies outside the closed surface. Either ‘points’ or ‘cellIds’ 
   #can be set to NULL if you don’t want to receive that information."

    pointsVTKIntersectionData = pointsVTKintersection.GetData()
    nPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
#   pointsIntersection = []
#    for idx in range(nPointsVTKIntersection):
#        _tup = pointsVTKIntersectionData.GetTuple3(idx)
#        pointsIntersection.append(_tup)

#    pointsIntersection=np.asarray([pointsVTKIntersectionData.GetTuple3(idx) 
#                                for idx in range(nPointsVTKIntersection)])
    
    pointsIntersection=np.zeros((nPointsVTKIntersection,3))
    for i in range(nPointsVTKIntersection):
        pointsIntersection[i,:]=pointsVTKIntersectionData.GetTuple3(i)
     
    return pointsIntersection


#-----------------------------------------------------------------------------
#@jit
def __InterpolateRemainingVoxels__(correctionLIST,sampleDimensions,image):
    
#    nzX,nzY = correctionLIST
#    correctionLIST=[[nzX[i],nzY[i]] for i in range(nzX.size)]    
    
    countCORRECTIONLIST = correctionLIST.shape[0]
    
    if countCORRECTIONLIST>0:
        
        #If necessary, add a one-pixel border around the x and y edges of the
        #array.  This prevents an error if the code tries to interpolate a ray at
        #the edge of the x,y grid.
#        cond0 = min([correctionLIST[i][0] for i in range(len(correctionLIST))])==0
#        cond1 = max([correctionLIST[i][0] for i in range(len(correctionLIST))])==sampleDimensions[0]-1
#        cond2 = min([correctionLIST[i][1] for i in range(len(correctionLIST))])==0
#        cond3 = max([correctionLIST[i][1] for i in range(len(correctionLIST))])==sampleDimensions[1]-1
    
    
        cond0 = np.min(correctionLIST[:,0] )==0
        cond1 = np.max(correctionLIST[:,0])==sampleDimensions[0]-1
        cond2 = np.min(correctionLIST[:,1])==0
        cond3 = np.max(correctionLIST[:,1])==sampleDimensions[1]-1
    
    
        if cond0 or cond1 or cond2 or cond3:
            image = np.hstack( (np.zeros((sampleDimensions[0],1,sampleDimensions[2])),
                                image,np.zeros((sampleDimensions[0],1,sampleDimensions[2]))))
            image = np.vstack( (np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2])),
                                image,np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2]))))
#            correctionLIST = [ [correctionLIST[i][0]+1,correctionLIST[i][1]+1] 
#                                            for i in range(len(correctionLIST)) ]
            correctionLIST = correctionLIST+1
                                            
        for loopC in range(countCORRECTIONLIST):
#            voxelsforcorrection = np.squeeze( np.sum( [ 
#                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]-1,:],
#                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1],:],
#                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]+1,:],
#                image[correctionLIST[loopC][0],correctionLIST[loopC][1]-1,:],
#                image[correctionLIST[loopC][0],correctionLIST[loopC][1]+1,:],
#                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]-1,:],
#                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1],:],
#                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]+1,:],
#                ], axis=0 ) )
#            voxelsforcorrection = (voxelsforcorrection>=4)
#            image[correctionLIST[loopC][0],correctionLIST[loopC][1],voxelsforcorrection] = 1
            
            voxelsforcorrection = np.squeeze( np.sum( [ 
                image[correctionLIST[loopC,0]-1,correctionLIST[loopC,1]-1,:],
                image[correctionLIST[loopC,0]-1,correctionLIST[loopC,1],:],
                image[correctionLIST[loopC,0]-1,correctionLIST[loopC,1]+1,:],
                image[correctionLIST[loopC,0],correctionLIST[loopC,1]-1,:],
                image[correctionLIST[loopC,0],correctionLIST[loopC,1]+1,:],
                image[correctionLIST[loopC,0]+1,correctionLIST[loopC,1]-1,:],
                image[correctionLIST[loopC,0]+1,correctionLIST[loopC,1],:],
                image[correctionLIST[loopC,0]+1,correctionLIST[loopC,1]+1,:],
                ], axis=0 ) )
            voxelsforcorrection = (voxelsforcorrection>=4)
            image[correctionLIST[loopC][0],correctionLIST[loopC][1],voxelsforcorrection] = 1
    #Remove the one-pixel border surrounding the array, if this was added
    #previously.
    if image.shape[0]>sampleDimensions[0] or image.shape[1]>sampleDimensions[1]:
        image = image[1:-1,1:-1,:]
    return image





#-------------------------------------------------------------------- 
#@jit   
def GetSubWindowInformation(subWindowBounds,gridX,gridY,gridZ): 

    nVoxGridX = len(gridX)-1  
    nVoxGridY = len(gridY)-1
    nVoxGridZ = len(gridZ)-1    
    bounds=subWindowBounds    
    
    
    Xmin = min(gridX)
    deltaX = (max(gridX)-Xmin)/float(nVoxGridX)    
    subNxMin = int(math.floor((bounds[0]-Xmin)/deltaX)-1)   
    subNxMax = int(math.ceil((bounds[1]-Xmin)/deltaX)+1)
    subgridXmin = Xmin+deltaX*subNxMin
    subgridXmax = Xmin+deltaX*subNxMax
    nVoxSubgridX = subNxMax - subNxMin
    
    Ymin = min(gridY)
    deltaY = (max(gridY)-Ymin)/float(nVoxGridY)
    subNyMin = int(math.floor((bounds[2]-Ymin)/deltaY)-1)
    subNyMax = int(math.ceil((bounds[3]-Ymin)/deltaY)+1)
    subgridYmin = Ymin+deltaY*subNyMin
    subgridYmax = Ymin+deltaY*subNyMax
    nVoxSubgridY = subNyMax - subNyMin
    
    Zmin = min(gridZ)
    deltaZ = (max(gridZ)-Zmin)/float(nVoxGridZ)
    subNzMin = int(math.floor((bounds[4]-Zmin)/deltaZ)-1)
    subNzMax = int(math.ceil((bounds[5]-Zmin)/deltaZ)+1)
    subgridZmin = Zmin+deltaZ*subNzMin
    subgridZmax = Zmin+deltaZ*subNzMax
    nVoxSubgridZ = subNzMax - subNzMin

    nVoxSubImage = (nVoxSubgridX,nVoxSubgridY,nVoxSubgridZ)
    
    boundSubgrid = (subgridXmin,subgridXmax,
                    subgridYmin,subgridYmax,
                    subgridZmin,subgridZmax,)
                    
    gridRelativePosition=(subNxMin,subNxMax,
                          subNyMin,subNyMax,
                          subNzMin,subNzMax,)
    
    return nVoxSubImage,boundSubgrid,gridRelativePosition

#-------------------------------------------------------------------- 
#@jit   
def InsertSubimageInImage(subImage,nVoxImage,gridRelativePosition): 
    
    subNxMin,subNxMax,subNyMin,subNyMax,subNzMin,subNzMax = gridRelativePosition   
    
    wXmin = max(0,subNxMin)
    wXmax = min(nVoxImage[0],subNxMax)
    subwXmin = max(0,-subNxMin)
    subwXmax = min(nVoxImage[0]-subNxMin,subNxMax-subNxMin)
    #print subwXmax, subwXmin , wXmax,wXmin
    assert subwXmax>=subwXmin &  (wXmax>=wXmin)    
    
    wYmin = max(0,subNyMin)
    wYmax = min(nVoxImage[1],subNyMax)
    subwYmin = max(0,-subNyMin)
    subwYmax = min(nVoxImage[1]-subNyMin,subNyMax-subNyMin)
    assert subwYmax>=subwYmin &  (wYmax>=wYmin)    
    
    wZmin = max(0,subNzMin)
    wZmax = min(nVoxImage[2],subNzMax)
    subwZmin = max(0,-subNzMin)
    subwZmax = min(nVoxImage[2]-subNzMin,subNzMax-subNzMin)
    assert subwZmax>=subwZmin &  (wZmax>=wZmin)
    
    objectImage = np.zeros(nVoxImage,dtype=np.uint8)
    objectImage[wXmin:wXmax,wYmin:wYmax,wZmin:wZmax] = subImage[subwXmin:subwXmax,
                                            subwYmin:subwYmax,subwZmin:subwZmax]

    return objectImage
    
