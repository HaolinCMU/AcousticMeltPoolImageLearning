close all; clear all; clc

%%
raw_audio_folder_path = 'data/raw_audio_data';
acoustic_data_folder_path = 'data/acoustic_data'; % Save generated wavelet spectrums. 

audio_paths = dir(fullfile(raw_audio_folder_path, '*.wav'));
audio_clip_length = 256; % In sample points. Length of 128 corresponds to ~(>)1ms audio and 30 image frames. 
audio_sampling_stride = 64;
fs = 96e3; % Sampling rate; 
wavelet = 'amor'; % Default: 'amor' or 'bump'. 
OMIT_DURATION = [0.0720, 0.0619, 0.0638, 0.0682, 0.0658, 0.0696, ...
                 0.0686, 0.0731, 0.0680, 0.0686, 0.0658, ...
                 0.0672, 0.0704, 0.0673, 0.0622, 0.0673, 0.0657, ...
                 0.0717, 0.0628, 0.0622, 0.0696, 0.0669, 0.0660, ...
                 0.0680, 0.0627, 0.0631, 0.0645, 0.0726, 0.0720]; % In s.

fig_resolution = 256; % Default: 512.
is_filter_bank = 0; % Default: 0. 
is_log_scale = 1; % Default: 0. 
is_multiview = 0; % Default: 0. 


%%
poolobj = parpool(8);
parfor i = 1:length(audio_paths)
    acoustic_data_subfolder_path = sprintf('%s/%04d', acoustic_data_folder_path, i-1);

    if exist(acoustic_data_subfolder_path, 'dir') == 0
        mkdir(acoustic_data_subfolder_path);
    end

    audio_path = sprintf('%s/%s', audio_paths(i).folder, audio_paths(i).name);
    [y, ~] = audioread(audio_path);
    y_eff = y(round(OMIT_DURATION(i)*fs):end);

    partitionAndCWTtoFolder(y_eff, audio_clip_length, audio_sampling_stride, ...
                            fs, wavelet, fig_resolution, acoustic_data_subfolder_path, ...
                            is_filter_bank, is_log_scale, is_multiview);

end

delete(poolobj);


%% Help functions.
function partitionAndCWTtoFolder(data, window_length, stride, fs, wavelet, ...
                                 fig_resolution, result_folder_path, ...
                                 is_filter_bank, is_log_scale, is_multiview)
    %{
    Partition the input data with specified window length, stride, and
    generate corresponding individual wavelet power spectrums using
    wavelet, sampling frequency and figure resolution. 
    
    Parameters:
    ----------
        data: Float vector. The target raw signal. 
        window_length: Int. The length of the clip in number of DPs.
        stride: Int. The length of stride (window_length-overlap) between
                consecutive clips while segmenting the signal. In DPs. 
        fs: Float. The sampling rate of the data. 
        wavelet: String token. The type of the selected mother wavelet function.
        fig_resolution: Int. The eventual resolution of the saved spectrum
                        image.
        result_folder_path: String. The path of the folder for saving the
                            clip spectrums. 
        is_log_scale: Int/Bool. 0: not using log scale. 1: using log scale.
        is_multiview: Int/Bool. 0: plot CWT spectrum only. 1: plot both CWT
                        sperctrum and temporal waveform. 
    %}
    
    signal_total_length = length(data);
    sample_num = fix((signal_total_length-window_length)/stride) + 1;

    for i = 1:sample_num
        start_ind = (i-1)*stride+1;
        clip_temp = data(start_ind:start_ind+window_length-1);
        CWT(clip_temp, wavelet, fs, fig_resolution, ...
            sprintf('%s/%06d.png', result_folder_path, i-1), ...
            is_filter_bank, is_log_scale, is_multiview);

        clear clip_temp;
    end
    
end


function CWT(data, wavelet, fs, fig_resolution, file_name, ...
             is_filter_bank, is_log_scale, is_multiview)
    %{
    Create a single continuous wavelet transform power spectrum. 
    Parameters:
    ----------
        data: Float vector. The transformation target. 
        wavelet: String token. The type of the selected mother wavelet function. 
        fs: Float. The samping rate of the target signal. 
        fig_resolution: Int. The eventual resolution of the saved spectrum
                        image. 
        file_name: String. The saving path of the output spectrum image. 
        is_log_scale: Int/Bool. 0: not using log scale. 1: using log scale.
        is_multiview: Int/Bool. 0: plot CWT spectrum only. 1: plot both CWT
                        sperctrum and temporal waveform. 
    %}
    

    data_len = size(data,1);

    if is_filter_bank 
        fb = cwtfilterbank(SignalLength=data_len, wavelet=wavelet, ...
                           SamplingFrequency=fs);
        [cfs, frq] = cwt(data, FilterBank=fb);
    else
        [cfs, frq] = cwt(data, wavelet, fs);
    end

    t = 0:1/fs:(data_len-1)/fs;
    
    if ~is_multiview % CWT scalogram only. No axes, labels and colorbars. 
        figure('visible','off');
        surface(t, frq, abs(cfs));
        axis tight;
        shading flat;
        if is_log_scale
            set(gca,"yscale","log"); % Apply log scale on y-axis. 
        end
        set(gca, 'visible', 'off');
        set(colorbar, 'visible', 'off');
        exportgraphics(gca, file_name, 'Resolution', fig_resolution);

    else % CWT spectrum + waveform plots. With axes, labels and colorbars. 
        f = figure;

        f1 = subplot(2,1,1);
        plot(t, data);
        axis tight;
        ylim(f1, [-0.25 0.1]);
        title("Signal and Scalogram");
        xlabel("Time (s)");
        ylabel("Amplitude");

        f2 = subplot(2,1,2);
        surface(t, frq, abs(cfs));
        axis tight;
        shading flat;
        xlabel("Time (s)");
        ylabel("Frequency (Hz)");
        if is_log_scale
            set(f2,"yscale","log"); % Apply log scale on y-axis. 
        end

        exportgraphics(f, file_name, 'Resolution', fig_resolution);
    end

    clear cfs frq;

end


