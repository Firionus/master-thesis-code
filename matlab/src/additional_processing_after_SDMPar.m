function [SRIR_data] = additional_processing_after_SDMPar(SRIR_data)

% from `Analyze_SRIR`

% crop
SRIR_data.Raw_RIR = SRIR_data.Raw_RIR(SRIR_data.DSonset:end,:);
SRIR_data.DOA = SRIR_data.DOA(SRIR_data.DSonset:end, :);
SRIR_data.DS_idx = SRIR_data.DS_idx - SRIR_data.DSonset;
SRIR_data.DSonset = 1;

% smooth DOA
SRIR_data = Smooth_DOA(SRIR_data);

% from `PreProcess_Synthesize_SDM_Binaural`

% DOA NaN removal

[DOA_rad(:,1), DOA_rad(:,2), DOA_rad(:,3)] = cart2sph( ...
    SRIR_data.DOA(:,1), SRIR_data.DOA(:,2), SRIR_data.DOA(:,3));
az = DOA_rad(:,1);
el = DOA_rad(:,2);

% ---- hack ----
% Sometimes you get NaNs from the DOA analysis
% Replace NaN directions with uniformly distributed random angle
az(isnan(az)) = pi-rand(size(az(isnan(az))))*2*pi;
el(isnan(el)) = pi/2-rand(size(el(isnan(el))))*pi;

az(az>pi) = az(az>pi) - 2*pi;
az(az<-pi) = az(az<-pi) + 2*pi;

el(el>pi/2) = el(el>pi/2) - pi;
el(el<-pi/2) = el(el<-pi/2) + pi;

[DOA_x, DOA_y, DOA_z] = sph2cart(az, el, 1);
SRIR_data.DOA = [DOA_x, DOA_y, DOA_z];

end