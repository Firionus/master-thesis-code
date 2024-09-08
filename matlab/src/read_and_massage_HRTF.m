% Copyright (c) Facebook, Inc. and its affiliates.
% modified from BinauralSDM/Src/PreProcess_Synthesize_SDM_Binaural.m

function [BRIR_data, HRIR, HRIR_data] = read_and_massage_HRTF(BRIR_data)

HRIR_data = Read_HRTF(BRIR_data);

switch upper(BRIR_data.HRTF_Type)
    case 'FRL_HRTF'
        % Restrict the HRTF directions to az=-180:180 degrees, el=-90:90 degrees
        HRIR_data.directions(HRIR_data.directions(:,1)>pi,1) = ...
            HRIR_data.directions(HRIR_data.directions(:,1)>pi,1) - 2*pi;

        [BRIR_data.HRTF_cartDir(:,1), BRIR_data.HRTF_cartDir(:,2), ...
            BRIR_data.HRTF_cartDir(:,3)] = sph2cart(HRIR_data.directions(:,1), ...
            HRIR_data.directions(:,2), HRIR_data.directions(:,3));

        BRIR_data.HRTF_cartDirNSTree = createns(BRIR_data.HRTF_cartDir);
    
        HRIR = cat(3, HRIR_data.left_HRTF, HRIR_data.right_HRTF);
        HRIR = permute(HRIR, [2, 3, 1]);

    case 'SOFA' 
        TempSourcePosition(:,1) = deg2rad(HRIR_data.SourcePosition(:,1));
        TempSourcePosition(:,2) = deg2rad(HRIR_data.SourcePosition(:,2));

        [BRIR_data.HRTF_cartDir(:,1), BRIR_data.HRTF_cartDir(:,2), ...
            BRIR_data.HRTF_cartDir(:,3)] = sph2cart(TempSourcePosition(:,1), ...
            TempSourcePosition(:,2), HRIR_data.SourcePosition(:,3));    

        HRIR = permute(HRIR_data.Data.IR, [3, 2, 1]);
    
    otherwise
        error('Invalid HRIR format "%s".', BRIR_data.HRTF_Type);
end

if BRIR_data.FFTshiftHRIRs
    HRIR = fftshift(HRIR, 1);
end

end
