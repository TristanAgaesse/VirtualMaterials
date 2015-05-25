# -*- coding: utf-8 -*-


import numpy as np
import vtk
from vtk.util import numpy_support
import math
from scipy import ndimage
import random
from skimage import morphology
import SimpleITK as sitk
from scipy.spatial import Voronoi
import time

from VirtualMaterials.Utilities  import tifffile as tff
from VirtualMaterials.Simulation  import FullMorphology


#--------------------------------------------------------------------    
def VisualizeMesh(polydata):

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
     
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
     
    renderer.AddActor(actor)
     
    renderWindow.Render()
    renderWindowInteractor.Start()



#--------------------------------------------------------------------
def VisualizeVolumeRendering(image):
    from mayavi import mlab
    mlab.pipeline.volume(mlab.pipeline.scalar_field(image))

#    from tvtk.api import tvtk
#    data = image
#    i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
#    i.point_data.scalars = data.ravel()
#    i.point_data.scalars.name = 'scalars'
#    i.dimensions = data.shape


#--------------------------------------------------------------------
def VisualizeCutPlanes(image):
    from mayavi import mlab
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(image),
                                plane_orientation='x_axes',
                                slice_index=10,
                            )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(image),
                                plane_orientation='y_axes',
                                slice_index=10,
                            )
    mlab.outline()


#--------------------------------------------------------------------
def VisualizeIsoSurface(image):
    
    from mayavi import mlab
    mlab.contour3d(image)





#--------------------------------------------------------------------
def VisualizeImageVTK(image):
                                         
    data_matrix=image.astype(np.uint8)
    
    shape=data_matrix.shape
    shape=[int(shape[i]) for i in range(len(shape))]
    w, d, h = shape
    
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(data_matrix, data_matrix.nbytes)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
    dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
    dataImporter.Update() 
    
    
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    colorFunc = vtk.vtkColorTransferFunction()
    for i in range(256):
        alphaChannelFunc.AddPoint(i, 0.2)
        colorFunc.AddRGBPoint(i,i/255.0,i/255.0,i/255.0)
    # for our test sample, we set the black opacity to 0 (transparent) so as
    #to see the sample  
    alphaChannelFunc.AddPoint(0, 0.0)
    colorFunc.AddRGBPoint(0,0,0,0)
    
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    #volumeProperty.ShadeOn()
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    
    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # We can finally create our volume. We also have to specify the data for
    # it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    # function to reduce the spacing between each image
    volumeMapper.SetMaximumImageSampleDistance(0.01)
    
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    
    # The class vtkVolume is used to pair the preaviusly declared volume as 
    #well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    
    # With almost everything else ready, its time to initialize the renderer and window,
    # as well as creating a method for exiting the application
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
    
    # We add the volume to the renderer ...
    renderer.AddVolume(volume)
    # ... set background color to white ...
    renderer.SetBackground(1, 1, 1)
    # ... and set window size.
    renderWin.SetSize(550, 550)
    renderWin.SetMultiSamples(4)
    # A simple function to be called when the user decides to quit the application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)
    
    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
    
    #to auit, press q
    renderInteractor.Initialize()
    # Because nothing will be rendered without any input, we order the first
    # render manually before control is handed over to the main-loop.
    renderWin.Render()
    renderInteractor.Start()


