function [BRIR_merged, SRIR_data] = render_brirs(SRIR_data, BRIR_data, HRIR)
    % modify mixing time to yield round number of samples (neede for SplitBRIR line 44)
    BRIR_data.MixingTime = round(BRIR_data.MixingTime*BRIR_data.fs)/BRIR_data.fs;

    % from PreProcess_Synthesize_SDM_Binaural
    % Extend SRIR data to BRIR length
    BRIR_length = ceil(BRIR_data.Length * SRIR_data.fs);
    if BRIR_length > length(SRIR_data.P_RIR)
        SRIR_data.DOA = [SRIR_data.DOA; repmat([1, 0, 0], BRIR_length - length(SRIR_data.DOA), 1)];
        SRIR_data.P_RIR(end:BRIR_length) = 0;
    end

    % from Demo_BinauralSDM_QuantizedDOA_andRTModAP
    % -----------------------------------------------------------------------
    % 3. Quantize DOA information, if required

    if BRIR_data.QuantizeDOAFlag
        [SRIR_data, ~] = QuantizeDOA(SRIR_data, ...
            BRIR_data.DOADirections, SRIR_data.DOAOnsetLength);
    end

    % -----------------------------------------------------------------------
    % 4. Compute parameters for RTMod Compensation

    % Synthesize one direction to extract the reverb compensation - solving the
    % SDM synthesis spectral whitening
    BRIR_Pre = Synthesize_SDM_Binaural(SRIR_data, BRIR_data, HRIR, [0, 0], true);

    % Using the pressure RIR as a reference for the reverberation compensation
    BRIR_data.ReferenceBRIR = [SRIR_data.P_RIR, SRIR_data.P_RIR];

    % Get the desired T30 from the Pressure RIR and the actual T30 from one
    % rendered BRIR
    [BRIR_data.DesiredT30, BRIR_data.OriginalT30, BRIR_data.RTFreqVector] = ...
        GetReverbTime(SRIR_data, BRIR_Pre, BRIR_data.BandsPerOctave, BRIR_data.EqTxx);
    clear BRIR_Pre;

    % -----------------------------------------------------------------------
    % 5. Render BRIRs with RTMod compensation for the specified directions

    nDirs = size(BRIR_data.Directions, 1);

    % Render early reflections
    parfor iDir = 1 : nDirs
        BRIR_early_temp = Synthesize_SDM_Binaural( ...
            SRIR_data, BRIR_data, HRIR, BRIR_data.Directions(iDir, :), false);
        BRIR_early(:, :, iDir) = Modify_Reverb_Slope(BRIR_data, BRIR_early_temp);
    end
    clear iDir;

    % Render late reverb
    BRIR_late = Synthesize_SDM_Binaural(SRIR_data, BRIR_data, HRIR, [0, 0], true);
    BRIR_late = Modify_Reverb_Slope(BRIR_data, BRIR_late);

    % Remove leading zeros
    % [BRIR_early, BRIR_late] = Remove_BRIR_Delay(BRIR_early, BRIR_late, -20);
    % NOTE: commented out since it is not required and we'll pad in Python again anyway

    % -----------------------------------------------------------------------
    % 6. Apply AP processing for the late reverb

    % AllPass filtering for the late reverb (increasing diffuseness and
    % smoothing out the EDC)
    BRIR_data.allpass_delays = [37, 113, 215];  % in samples
    BRIR_data.allpass_RT     = [0.1, 0.1, 0.1]; % in seconds

    BRIR_late = Apply_Allpass(BRIR_late, BRIR_data);

    
    % -----------------------------------------------------------------------
    % 7. Split the BRIRs into components by time windowing

    [BRIR_DSER, BRIR_LR, BRIR_DS, BRIR_ER] = Split_BRIR(...
    BRIR_early, BRIR_late, BRIR_data.MixingTime, BRIR_data.fs, 256);

    % end from Demo_BinauralSDM_QuantizedDOA_andRTModAP

    % merge to full BRIR
    % shape of BRIR_late: time, 2
    % shape of BRIR_early: mixing_time, 2, directions (?)
    BRIR_merged = zeros(size(BRIR_LR, 1), 2, size(BRIR_DSER, 3));
    for i = 1:size(BRIR_DSER,3)
        BRIR_merged(:,:,i) = BRIR_LR; % copy for each direction
    end
    BRIR_merged(1:size(BRIR_DSER,1), :, :) = BRIR_merged(1:size(BRIR_DSER,1), :, :) + BRIR_DSER(:,:,:);
end