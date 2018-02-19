import numpy as np
from scipy.stats import skew, kurtosis


class FeatureProcessing:
    with_mean = True
    with_std = True
    with_skew = False
    with_kurtosis = False

    def __init__(self, with_mean=True, with_std=True, with_skew=False, with_kurtosis=False):
        self.with_kurtosis = with_kurtosis
        self.with_mean = with_mean
        self.with_skew = with_skew
        self.with_std = with_std

    def mean(self, data):
        return np.mean(data, axis=0)

    def std(self, data):
        return np.std(data, axis=0)

    def skew(self, data):
        return skew(data, axis=0)

    def kurtosis(self, data):
        return kurtosis(data, axis=0)

    def process_feature(self, track):
        mfcc_feature = track.mfcc_features
        spectral_feature = track.spectral_features
        timing_feature = track.timing_features
        result_timing_feature = []
        result_mfcc_feature = []
        result_spectral_feature = []
        if self.with_std:
            std_mfcc = self.std(mfcc_feature)
            std_spectral = self.std(spectral_feature)
            std_timing = self.std(timing_feature)
            result_mfcc_feature = np.concatenate((result_mfcc_feature, std_mfcc))
            result_spectral_feature = np.concatenate((result_spectral_feature, std_spectral))
            result_timing_feature = np.concatenate((result_timing_feature, std_timing))
        if self.with_mean:
            mean_mfcc = self.mean(mfcc_feature)
            mean_spectral = self.mean(spectral_feature)
            mean_timing = self.mean(timing_feature)
            result_mfcc_feature = np.concatenate((result_mfcc_feature, mean_mfcc))
            result_spectral_feature = np.concatenate((result_spectral_feature, mean_spectral))
            result_timing_feature = np.concatenate((result_timing_feature, mean_timing))
        if self.with_kurtosis:
            kurtosis_mfcc = self.kurtosis(mfcc_feature)
            kurtosis_spectral = self.kurtosis(spectral_feature)
            kurtosis_timing = self.kurtosis(timing_feature)
            result_mfcc_feature = np.concatenate((result_mfcc_feature, kurtosis_mfcc))
            result_spectral_feature = np.concatenate((result_spectral_feature, kurtosis_spectral))
            result_timing_feature = np.concatenate((result_timing_feature, kurtosis_timing))
        if self.with_skew:
            skew_mfcc = self.skew(mfcc_feature)
            skew_spectral = self.skew(spectral_feature)
            skew_timing = self.skew(timing_feature)
            result_mfcc_feature = np.concatenate((result_mfcc_feature, skew_mfcc))
            result_spectral_feature = np.concatenate((result_spectral_feature, skew_spectral))
            result_timing_feature = np.concatenate((result_timing_feature, skew_timing))
        track.mfcc_features = np.nan_to_num(result_mfcc_feature)
        track.spectral_features = np.nan_to_num(result_spectral_feature)
        track.timing_features = np.nan_to_num(result_timing_feature)
        track.percussion_features = np.nan_to_num(track.percussion_features)
        return track
