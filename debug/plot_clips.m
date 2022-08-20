close all; clear all; clc

%%
dir = "F:/data/processed/acoustic/clips/Layer0197_P250_V1200_C001H001S0001/Layer0197_P250_V1200_C001H001S0001.mat";
clip_id = 1;
sr = 100000; % Hz. 

clips_mat = load(dir);
clip = clips_mat.clips_mat(clip_id,:);
t = 0:1/sr:(size(clip,2)-1)/sr;

plot(t, clip);