close all; clear all; clc

%%
raw_audio_folder_path = 'data/raw_audio_data';
acoustic_data_folder_path = 'data/acoustic_data'; % Save generated wavelet spectrums. 

audio_paths = dir(fullfile(raw_audio_folder_path, '*.wav'));
audio_clip_length = 128; % In sample points. Corresponding to ~1ms audio and 30 image frames. 
audio_sampling_stride = 64;
fs = 96e3; % Sampling rate; 
wavelet = 'bump';
fig_resolution = 512;
OMIT_DURATION = [0.069, 0.058, 0.061, 0.065, 0.061, 0.066, ...
                 0.065, 0.070, 0.064, 0.065, 0.062, ... 
                 0.063, 0.066, 0.063, 0.058, 0.063, 0.062, ...
                 0.068, 0.059, 0.058, 0.065, 0.063, 0.062, ...
                 0.064, 0.059, 0.059, 0.060, 0.068, 0.068]; % In s.


%%
poolobj = parpool(8);
parfor i = 1:length(audio_paths)
    acoustic_data_subfolder_path = sprintf('%s/%04d', acoustic_data_folder_path, i-1);
%     if exist(acoustic_data_subfolder_path, 'dir') ~= 0
%         rmdir(acoustic_data_subfolder_path);
%     end
%     mkdir(acoustic_data_subfolder_path);

    audio_path = sprintf('%s/%s', audio_paths(i).folder, audio_paths(i).name);
    [y, ~] = audioread(audio_path);
    y_eff = y(round(OMIT_DURATION(i)*fs):end);

    partitionAndCWTtoFolder(y_eff, audio_clip_length, audio_sampling_stride, ...
                            fs, wavelet, fig_resolution, acoustic_data_subfolder_path);

end

delete(poolobj);


%% Help functions.
function partitionAndCWTtoFolder(data, window_length, stride, fs, wavelet, ...
                                 fig_resolution, result_folder_path)
    %{
    Partition the input data with specified window length, stride, and
    generate corresponding individual wavelet power spectrums using
    wavelet, sampling frequency and figure resolution. 
    
    Parameters:
    ----------
        data: Float vector. The target raw signal. 
    %}
    
    signal_total_length = length(data);
    sample_num = fix((signal_total_length-window_length)/stride) + 1;

    for i = 1:sample_num
        start_ind = (i-1)*stride+1;
        clip_temp = data(start_ind:start_ind+window_length-1);
        CWT(clip_temp, wavelet, fs, fig_resolution, ...
            sprintf('%s/%06d.png', result_folder_path, i-1));

        clear clip_temp;
    end
    
end


function CWT(data, wavelet, fs, fig_resolution, file_name)
    %{
    Create a single continuous wavelet transform power spectrum. 

    Parameters:
    ----------
        data: Float vector. The transformation target. 
        wavelet: String token. The type of the selected mother wavelet function. f
        fs: Float. The samping rate of the target signal. 
        fig_resolution: Int. The eventual resolution of the saved spectrum
                        image. 
        file_name: String. The saving path of the output spectrum image. 
    %}

    [cfs, frq] = cwt(data, wavelet, fs);
    t = 0:1/fs:(size(data,1)-1)/fs;

    figure('visible','off');
    surface(t, frq, abs(cfs));
    shading flat;
    set(gca, 'visible', 'off');
    set(colorbar, 'visible', 'off');
    exportgraphics(gca, file_name, 'Resolution', fig_resolution);

    clear cfs frq;

end

