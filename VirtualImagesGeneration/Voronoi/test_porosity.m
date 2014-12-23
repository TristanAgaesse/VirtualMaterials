function [network,porosity]=test_porosity(nPore,anisotropy)
    
    disp(nPore)

    geometry=MacroscopicGeometry;
    geometry.LoadGeometry('1block3D');
    geometry.Blocks{1, 1}.Filling.PoreNumber=nPore;

    band=struct;
    band.Direction='z';
    band.Intensity=anisotropy;
    band.Location=[-100,500];
    geometry.AddAnisotropyBand(band)
    
    
    
    builder=NetworkBuilder(geometry);
    network=builder.BuildNetwork;
    
    disp('calcul void volume')
    tic
    voidVolumes=network.ComputeAllPoreVolume;
    toc
    
    diameters=network.EdgeDataList.EdgeDatas.FiberDiameter;
    
    network.EdgeDataList.EdgeDatas.FiberDiameter=0*network.EdgeDataList.EdgeDatas.FiberDiameter;
    disp('calcul total volume')
    tic;
    totalVolumes=network.ComputeAllPoreVolume;
    toc
    
    porosity=sum(voidVolumes)/sum(totalVolumes);
    
    network.EdgeDataList.EdgeDatas.FiberDiameter=diameters;
end