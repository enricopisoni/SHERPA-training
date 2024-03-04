'''
Created on 5-gen-2017

@author: roncolato
'''
import numpy as np
import netCDF4 as cdf
import time
#import matplotlib.pyplot as plt
#import sherpa.validation.DrawShape as ds

def saveToNetcdf(alpha,omega,flatWeight,x,y,nameDirOut,aqiFil,domain,radFull,radFlat,flagFlat,
                                        rf, radStep1, Ide, flagRegioMatFile, nametest, explain_step_1, explain_step_2,
                                        Order_Pollutant, alpha_physical_intepretation, omega_physical_intepretation, nPrec):
        
    alphaF = np.zeros((alpha.shape[2],alpha.shape[0],alpha.shape[1]));
    omegaF = np.zeros((omega.shape[2],omega.shape[0],omega.shape[1]));
    # omegaAccuracyF = np.zeros((omegaAccuracy.shape[2], omegaAccuracy.shape[0], omegaAccuracy.shape[1]));
    # omegaNotFiltF = np.zeros((omegaNotFilt.shape[2], omegaNotFilt.shape[0], omegaNotFilt.shape[1]));

    flatWeightF = np.zeros((flatWeight.shape[2],flatWeight.shape[0],flatWeight.shape[1]));
    for k in range(0, nPrec):
        alphaF[k,:,:] = np.flipud(alpha[:,:,k]);
        omegaF[k,:,:] = np.flipud(omega[:,:,k]);
        # omegaAccuracyF[k, :, :] = np.flipud(omegaAccuracy[:, :, k]);
        # omegaNotFiltF[k, :, :] = np.flipud(omegaNotFilt[:, :, k]);
        flatWeightF[k,:,:] = np.flipud(flatWeight[:,:,k]);

    #20231107 EP, this is required because the WRF domain is regular in km but irregular in lat-lon
    # while the CAMS one is regular in lat-lon but irregular in km
    if domain=='wrfchem_china_27kmres_2023' :
        latitude = np.flipud(y);
        longitude = np.flipud(x);
    else :
        latitude = np.flipud(y);
        longitude = x;

    ncfile = nameDirOut+'SR_'+aqiFil+'.nc';
    ncid = cdf.Dataset(ncfile, 'w', format='NETCDF3_CLASSIC');
    latDimId = ncid.createDimension("latitude", latitude.shape[0]);
    lonDimId = ncid.createDimension("longitude", longitude.shape[1]);
    pollDimId = ncid.createDimension("pollutant", 5);
    
    varid_lat = ncid.createVariable("lat","f8",('latitude','longitude'));
    varid_lon = ncid.createVariable("lon","f8",('latitude','longitude'));
    varid_alpha = ncid.createVariable("alpha","f8",('pollutant','latitude','longitude'));
    varid_omega = ncid.createVariable("omega","f8",('pollutant','latitude','longitude'));
    # varid_omegaAccuracy = ncid.createVariable("omegaAccuracy", "f8", ('pollutant', 'latitude', 'longitude'));
    # varid_omegaNotFilt = ncid.createVariable("omegaNotFilt", "f8", ('pollutant', 'latitude', 'longitude'));

    if flagFlat:                             
        varid_flatWeight = ncid.createVariable("flatWeight","f8",('pollutant','latitude','longitude'));

    now = time.strftime("%c")
    ncid.date_of_production = now
    ncid.flag_weight_used = str(flagFlat)

    ncid.step1_receptor_window = str(rf)
    ncid.step1_area_of_influence = str(radStep1)
    ncid.step1_algorithm_for_training = explain_step_1
    ncid.step1_omega_physical_intepretation = omega_physical_intepretation;

    ncid.step2_used_training_scenario = np.array_str(Ide)
    ncid.step2_area_of_influence = str(radFull)
    ncid.step2_algorithm_for_training = explain_step_2
    ncid.step2_alpha_physical_intepretation = alpha_physical_intepretation;

    ncid.flag_used_to_mask_sea_cells = flagRegioMatFile
    ncid.name_of_the_test = nametest

    ncid.Order_Pollutant = Order_Pollutant;

    #ncid.Glossary = 'm=met categories; p=precursor; in=inside radius; out= outside radius';
    #if flagFlat:
    #    ncid.Varying_weight_inside_radius = 'F_mp^in=1./((1+d).^omega_mp)';

    if flagFlat:
        ncid.flatWeight = 'is used as a weigth for all emission domain but the one in the radius of influence';
    if flagFlat:
        ncid.Radius_of_influence = radFlat;
    else:
        ncid.Radius_of_influence = radFull;
        
    #ncid.Emission_and_concentrations = 'in delta values';
    #ncid.Type_of_source_receptor = 'linear, varying per cell';

    varid_lat[:] = latitude;
    varid_lon[:] = longitude;
    varid_alpha[:] = alphaF;
    varid_omega[:] = omegaF;
    # varid_omegaAccuracy[:] = omegaAccuracyF;
    # varid_omegaNotFilt[:] = omegaNotFiltF;

    if flagFlat:           
        varid_flatWeight[:] = flatWeightF;

    ncid.close();
                 
#    alpha_pol = ('alpha_nox','alpha_voc','alpha_nh3','alpha_ppm','alpha_so2');
#    omega_pol = ('omega_nox','omega_voc','omega_nh3','omega_ppm','omega_so2');
    
#    for poll in range(0, 5):
#        h = plt.figure(1);
#        ax = h.gca();
#        plt.contourf(x, y, alpha[:,:,poll], 15, vmin=0, vmax=0.8);
#        
#        shapeFile = 'input/'+domain+'/Cntry02/cntry02.shp';
#        ax.axis('scaled');
#        ds.drawShape(ax,shapeFile);
#                
#        ax.set_xlim([np.min(x),np.max(x)]);
#        ax.set_ylim([np.min(y),np.max(y)]);
#        plt.colorbar();
#        name = nameDirOut+alpha_pol[poll];
#        print(name);
#        h.savefig(name+'_py.png', format="png");
#        plt.close(h);
#    
#    for poll in range(0, 5):
#        h = plt.figure(1);
#        ax = h.gca();
#        plt.contourf(x, y, omega[:,:,poll], 15, vmin=1, vmax=3);
#        
#        shapeFile = 'input/'+domain+'/Cntry02/cntry02.shp';
#        ax.axis('scaled');
#        ds.drawShape(ax,shapeFile);
#                
#        ax.set_xlim([np.min(x),np.max(x)]);
#        ax.set_ylim([np.min(y),np.max(y)]);
#        plt.colorbar();
#        name = nameDirOut+omega_pol[poll];
#        print(name);
#        h.savefig(name+'_py.png', format="png");
#        plt.close(h);

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE TO NECTDF BEGIN
for k=1:5
    alphaF(:,:,k)=fliplr(alpha(:,:,k)'); 
    omegaF(:,:,k)=fliplr(omega(:,:,k)');
    flatWeightF(:,:,k)=fliplr(flatWeight(:,:,k)');
end

latitude=fliplr(y');
longitude=fliplr(x');

% mode = netcdf.getConstant('NETCDF4');
mode = netcdf.getConstant('classic_model');
ncid = netcdf.create(strcat(nameDirOut,'SR_',aqiFil,'.nc'),mode);

latDimId = netcdf.defDim(ncid,'latitude',448);
lonDimId = netcdf.defDim(ncid,'longitude',384);
pollDimId = netcdf.defDim(ncid,'pollutant',5);

varid_lat = netcdf.defVar(ncid,'lat','double',[lonDimId latDimId]);
varid_lon = netcdf.defVar(ncid,'lon','double',[lonDimId latDimId]);
varid_alpha = netcdf.defVar(ncid,'alpha','double',[lonDimId latDimId pollDimId]);
varid_omega = netcdf.defVar(ncid,'omega','double',[lonDimId latDimId pollDimId]);
varid_flatWeight = netcdf.defVar(ncid,'flatWeight','double',[lonDimId latDimId pollDimId]);

varid = netcdf.getConstant('GLOBAL');

netcdf.putAtt(ncid,varid,'Order_Pollutant','NOx, NMVOC, NH3, PPM, SOx');
netcdf.putAtt(ncid,varid,'Glossary','m=met categories; p=precursor; in=inside radius; out= outside radius');
netcdf.putAtt(ncid,varid,'Varying weight inside radius','F_mp^in=1./((1+d).^omega_mp)');
netcdf.putAtt(ncid,varid,'Alpha','specifies the precursor relative importance')
netcdf.putAtt(ncid,varid,'Omega','optimized per meteorological categories and per precursor')
netcdf.putAtt(ncid,varid,'flatWeight','is used as a weigth for all emission domain but the one in the radius of influence')
netcdf.putAtt(ncid,varid,'Radius of influence',rad(1))
netcdf.putAtt(ncid,varid,'Emission and concentrations','in delta values')
netcdf.putAtt(ncid,varid,'Type of source-receptor','linear, varying per cell')

netcdf.endDef(ncid);

netcdf.putVar(ncid,varid_lat,latitude);
netcdf.putVar(ncid,varid_lon,longitude);
netcdf.putVar(ncid,varid_alpha,alphaF);
netcdf.putVar(ncid,varid_omega,omegaF);
netcdf.putVar(ncid,varid_flatWeight,flatWeightF);

netcdf.close(ncid)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CREATE MAPS OF ALPHA AND OMEGA
geoshow(y,x,alpha(:,:,1),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([0:0.01:0.8]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'alpha_nox'));

geoshow(y,x,alpha(:,:,2),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([0:0.01:0.8]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'alpha_voc'));

geoshow(y,x,alpha(:,:,3),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([0:0.01:0.8]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'alpha_nh3'));

geoshow(y,x,alpha(:,:,4),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([0:0.01:0.8]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'alpha_ppm'));

geoshow(y,x,alpha(:,:,5),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([0:0.01:0.8]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'alpha_so2'));

geoshow(y,x,omega(:,:,1),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([1:.1:3]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'omega_nox'));

geoshow(y,x,omega(:,:,2),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([1:.1:3]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'omega_voc'));

geoshow(y,x,omega(:,:,3),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([1:.1:3]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'omega_nh3'));

geoshow(y,x,omega(:,:,4),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([1:.1:3]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'omega_ppm'));

geoshow(y,x,omega(:,:,5),'DisplayType','texturemap');
geoshow(strcat('../input/',domain,'/Cntry02/cntry02.shp'),'FaceColor','none');        %map to be plotted
contourcmap(([1:.1:3]),'jet','colorbar','off');
tmp=gca; tmp.XLim=[min(min(x)) max(max(x))]; tmp.YLim=[min(min(y)) max(max(y))]; colorbar
print('-dpng',strcat(nameDirOut,'omega_so2'));
'''