close all; clear all; clc

%%
raw_audio_folder_path = 'F:/data/processed/acoustic/clips';
spectrum_data_folder_path = 'F:/data/processed/acoustic/wavelet_spectrums_short'; % Save generated wavelet spectrums. 
clips_folder_list = dir(fullfile(raw_audio_folder_path));

fs = 100e3; % Sampling rate; 

wavelet = 'amor'; % Default: 'amor' or 'bump'. 
fig_resolution = 600; % dpi. Default: 300.
colorbar_lim = [0 0.025]; % Default: [0 0.03]. Must keep consistent throughout the wavelet spectrogram graph generation. 

is_filter_bank = 0; % Default: 0. 
is_log_scale = 1; % Default: 1. 
is_multiview = 0; % Default: 0. 
is_colorbar = 0; % Default: 0. 
is_colorbar_fixed = 1; % Default: 1. 


%%
poolobj = parpool(8); % Max: 8 workers.
parfor i = 1:length(clips_folder_list)
    if i <= 2 % Skip the first two fake "folders". 
        continue
    end
    
    clips_subfolder_path = sprintf('%s/%s', raw_audio_folder_path, clips_folder_list(i).name);
    spectrum_subfolder_path = sprintf('%s/%s', spectrum_data_folder_path, clips_folder_list(i).name);

    if exist(spectrum_subfolder_path, 'dir') == 0
        mkdir(spectrum_subfolder_path);
    end

    clips_file_paths = dir(fullfile(clips_subfolder_path, "*.mat"));
    clips_mat = load(sprintf('%s/%s', clips_subfolder_path, clips_file_paths(1).name));

    for j = 1:size(clips_mat.clips_mat, 1)
        clip_temp = clips_mat.clips_mat(j,:);
        clip_temp = reshape(clip_temp, size(clip_temp, 2), []); % Reshape it as a row vector. 
        file_name = sprintf('%s/%06d.png', spectrum_subfolder_path, j-1);

        CWT(clip_temp, wavelet, fs, fig_resolution, file_name, ...
            is_filter_bank, is_log_scale, is_multiview, ...
            is_colorbar, is_colorbar_fixed, colorbar_lim);
    end

    clips_mat = []; % Release memory. 
end

delete(poolobj);


%% Help functions.
function partitionAndCWTtoFolder(data, window_length, stride, fs, wavelet, ...
                                 fig_resolution, result_folder_path, ...
                                 is_filter_bank, is_log_scale, is_multiview, ...
                                 is_colorbar, is_colorbar_fixed, colorbar_lim)
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
            is_filter_bank, is_log_scale, is_multiview, ...
            is_colorbar, is_colorbar_fixed, colorbar_lim);

        clear clip_temp;
    end
    
end


function CWT(data, wavelet, fs, fig_resolution, file_name, ...
             is_filter_bank, is_log_scale, is_multiview, ...
             is_colorbar, is_colorbar_fixed, colorbar_lim)
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
    
    if ~is_multiview % CWT scalogram only. Default: no axes, labels and colorbars. 
        f = figure('visible','off');
        f1 = subplot(1,1,1);

        surface(t, frq, abs(cfs));
        axis tight;
        shading flat;

        if is_log_scale
            set(f1, "yscale", "log"); % Apply log scale on y-axis. 
        end
        set(f1, 'visible', 'off');

        if ~is_colorbar
            set(colorbar, 'visible', 'off');
            if is_colorbar_fixed
                caxis(colorbar_lim);
            end
        else
            set(colorbar);
            if is_colorbar_fixed
                caxis(colorbar_lim);
            end
        end

        ExportGraph(f, file_name, fig_resolution);

        delete(f)
        delete(f1)
        clear ExportGraph;
        clf;

    else % CWT spectrum + waveform plots. With axes, labels and colorbars. 
        f = figure;

        f1 = subplot(2,1,1);
        plot(t, data);
        axis tight;
        ylim(f1, [-0.3 0.3]);
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
            set(f2, "yscale", "log"); % Apply log scale on y-axis. 
        end

        if ~is_colorbar
            set(colorbar, 'visible', 'off');
            if is_colorbar_fixed
                caxis(colorbar_lim);
            end
        else
            set(colorbar);
            if is_colorbar_fixed
                caxis(colorbar_lim);
            end
        end
        
        ExportGraph(f, file_name, fig_resolution);

        delete(f)
        delete(f1)
        delete(f2)
        clear ExportGraph;
        clf;
    end

    clear t cfs frq;

end


function ExportGraph(f, fig_name, fig_dpi)
    %{
    Export graph. 
    
    Parameters:
    ----------
    f: Handle/object of figure subplot. 
    fig_name: Path to save the graph. 
    fig_dpi: Dpi of the saved graph. 
    %}

    exportgraphics(f, fig_name, 'Resolution', fig_dpi);
end
