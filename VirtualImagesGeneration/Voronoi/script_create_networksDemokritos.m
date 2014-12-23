for i=1:5:20
    
    for jRealisation=0:9
        nPore=ceil(300*(2.7-1.6*(i-1)/15));
        anisotropy=1+(i-1)*0.1;

        geometry=MacroscopicGeometry;
        geometry.LoadGeometry('1block3D');
        geometry.Blocks{1, 1}.Filling.PoreNumber=nPore;

        band=struct;
        band.Direction='z';
        band.Intensity=anisotropy;
        band.Location=[-100,500];
        geometry.AddAnisotropyBand(band)

        geometry.RandomSeed=jRealisation;

        builder=NetworkBuilder(geometry);
        network=builder.BuildNetwork;



        network.ExportToFreecad(sprintf('%dPores-%fAnisotropy-%dRandomSeed',nPore,anisotropy,jRealisation));
    
    end
end