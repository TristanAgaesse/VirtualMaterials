function network=make_network_demokritos(nPore,anisotropy)
    
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
    

    
end