clear all
close all

% DataMat{iStorm}:
    % col1: HURDAT: latitude
    % col2: HURDAT: longitude
    % col3: HURDAT-derived: distance traveled (km) (i-1->i)
    % col4: HURDAT: date/time (matlab format)
    % col5: HURDAT-derived: azimuth (heading; clockwise from North) (i-1->i)
    % col6: HURDAT-derived: speed (km/hr)(i-1->i)
    % col7: HURDAT: central pressure (HURDAT unit)
    % col8: HURDAT: windspeed (HURDAT units)
    % col9: HURDAT-derived: landfall flag (=1 for landfall value; =0 otherwise)       
    % col10: HURDAT-derived: overland flag (=1 when point is over land; =0 when over water)
    % col11: EIBtracks-interpolated: maximum wind speed (km/hr)
    % col12: EIBtracks-interpolated: minimum central pressure (hPa)
    % col13: EIBtracks-interpolated: radius of maximum wind speed (km)
    % col14: EIBtracks-interpolated: eye diameter (nm)
    % col15: EIBtracks-interpolated: pressure of the outer closed isobar (hPa)
    % col16: EIBtracks-interpolated: radius of the outer closed isobar (nm) 

HData = importdata('Latest_imputeddataset.mat');
stormData = HData.DataMatComb;
dateData = HData.DateVecComb;
bestStormData = {};
bestStormDates = {};

for i = 1:length(stormData)
    if length(stormData{1, i}) >= 95;
        bestStormData{end + 1} = stormData{1, i};
        bestStormDates{end + 1} = dateData{1, i};
    end
end


for i = 1:length(bestStormData)
    writematrix(bestStormData{i}, ...
        append('/Users/jasonluo/Documents/Hurricane_proj/matToPyData/bestStormData_', string(i), '.csv'));
end


for i = 1:length(bestStormDates)
    writematrix(bestStormDates{i}, ...
        append('/Users/jasonluo/Documents/Hurricane_proj/matToPyDates/bestStormDates_', string(i), '.csv'))
end






