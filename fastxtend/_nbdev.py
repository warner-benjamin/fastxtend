# AUTOGENERATED BY NBDEV! DO NOT EDIT!

__all__ = ["index", "modules", "custom_doc_links", "git_url"]

index = {"is_listish": "0_basics.ipynb",
         "listify_store_attr": "0_basics.ipynb",
         "audio_extensions": "audio.01_core.ipynb",
         "show_audio_signal": "audio.01_core.ipynb",
         "TensorAudio": "audio.01_core.ipynb",
         "show_spectrogram": "audio.01_core.ipynb",
         "get_usable_kwargs": "audio.01_core.ipynb",
         "TensorSpec": "audio.01_core.ipynb",
         "TensorMelSpec": "audio.01_core.ipynb",
         "TfmdDL.to": "audio.02_data.ipynb",
         "Spectrogram": "audio.02_data.ipynb",
         "MelSpectrogram": "audio.02_data.ipynb",
         "AudioBlock": "audio.02_data.ipynb",
         "SpecBlock": "audio.02_data.ipynb",
         "MelSpecBlock": "audio.02_data.ipynb",
         "Flip": "audio.03_augment.ipynb",
         "Roll": "audio.03_augment.ipynb",
         "AudioPadMode": "audio.03_augment.ipynb",
         "TensorAudio.crop_pad": "audio.03_augment.ipynb",
         "RandomCropPad": "audio.03_augment.ipynb",
         "VolumeMode": "audio.03_augment.ipynb",
         "Volume": "audio.03_augment.ipynb",
         "PeakNorm": "audio.03_augment.ipynb",
         "VolumeOrPeakNorm": "audio.03_augment.ipynb",
         "NoiseColor": "audio.03_augment.ipynb",
         "Noise": "audio.03_augment.ipynb",
         "VolumeBatch": "audio.03_augment.ipynb",
         "TensorAudio.pitch_shift": "audio.03_augment.ipynb",
         "PitchShift": "audio.03_augment.ipynb",
         "PitchShiftTA": "audio.03_augment.ipynb",
         "TensorAudio.time_stretch": "audio.03_augment.ipynb",
         "TimeStretch": "audio.03_augment.ipynb",
         "PitchShiftOrTimeStretch": "audio.03_augment.ipynb",
         "TimeMasking": "audio.03_augment.ipynb",
         "FrequencyMasking": "audio.03_augment.ipynb",
         "AmplitudeToDBMode": "audio.03_augment.ipynb",
         "AmplitudeToDB": "audio.03_augment.ipynb",
         "AudioNormalize": "audio.03_augment.ipynb",
         "StackSpecCallback": "audio.04_learner.ipynb",
         "audio_learner": "audio.04_learner.ipynb",
         "AudioMixHandler": "audio.05_mixup.ipynb",
         "AudioMixUp": "audio.05_mixup.ipynb",
         "AudioCutMix": "audio.05_mixup.ipynb",
         "AudioCutMixUp": "audio.05_mixup.ipynb",
         "AudioCutMixUpAugment": "audio.05_mixup.ipynb",
         "ChannelsLastTfm": "callback.channelslast.ipynb",
         "ChannelsLastCallback": "callback.channelslast.ipynb",
         "Learner.to_channelslast": "callback.channelslast.ipynb",
         "Learner.to_contiguous": "callback.channelslast.ipynb",
         "CutMix": "callback.cutmixup.ipynb",
         "CutMixUp": "callback.cutmixup.ipynb",
         "CutMixUpAugment": "callback.cutmixup.ipynb",
         "EMACallback": "callback.ema.ipynb",
         "LRFinder": "callback.lr_finder.ipynb",
         "Learner.lr_find": "callback.lr_finder.ipynb",
         "MESALoss": "callback.mesa.ipynb",
         "MESACallback": "callback.mesa.ipynb",
         "Callback.__call__": "callback.simpleprofiler.ipynb",
         "Learner.all_batches": "callback.simpleprofiler.ipynb",
         "Learner.show_training_loop": "callback.simpleprofiler.ipynb",
         "SimpleProfilerPostCallback": "callback.simpleprofiler.ipynb",
         "SimpleProfilerCallback": "callback.simpleprofiler.ipynb",
         "Learner.profile": "callback.simpleprofiler.ipynb",
         "TerminateOnTrainNaN": "callback.tracker.ipynb",
         "SaveModelAtEnd": "callback.tracker.ipynb",
         "KFoldColSplitter": "data.transforms.ipynb",
         "ParentSplitter": "data.transforms.ipynb",
         "GreatGrandparentSplitter": "data.transforms.ipynb",
         "ClassBalanced": "losses.ipynb",
         "ClassBalancedCrossEntropyLoss": "losses.ipynb",
         "ClassBalancedBCEWithLogitsLoss": "losses.ipynb",
         "LogMetric": "metrics.ipynb",
         "MetricType": "metrics.ipynb",
         "ActivationType": "metrics.ipynb",
         "MetricX": "metrics.ipynb",
         "AvgMetricX": "metrics.ipynb",
         "AccumMetricX": "metrics.ipynb",
         "AvgSmoothMetricX": "metrics.ipynb",
         "AvgLossX": "metrics.ipynb",
         "AvgSmoothLossX": "metrics.ipynb",
         "ValueMetricX": "metrics.ipynb",
         "Recorder.__init__": "metrics.ipynb",
         "Recorder.before_fit": "metrics.ipynb",
         "Recorder.after_batch": "metrics.ipynb",
         "Recorder.before_epoch": "metrics.ipynb",
         "Recorder.before_train": "metrics.ipynb",
         "Recorder.before_validate": "metrics.ipynb",
         "Recorder.after_train": "metrics.ipynb",
         "Recorder.after_validate": "metrics.ipynb",
         "Recorder.after_epoch": "metrics.ipynb",
         "Recorder.train_mets": "metrics.ipynb",
         "Recorder.valid_mets": "metrics.ipynb",
         "func_to_metric": "metrics.ipynb",
         "skm_to_fastxtend": "metrics.ipynb",
         "accuracy": "metrics.ipynb",
         "Accuracy": "metrics.ipynb",
         "error_rate": "metrics.ipynb",
         "ErrorRate": "metrics.ipynb",
         "top_k_accuracy": "metrics.ipynb",
         "TopKAccuracy": "metrics.ipynb",
         "APScoreBinary": "metrics.ipynb",
         "BalancedAccuracy": "metrics.ipynb",
         "BrierScore": "metrics.ipynb",
         "CohenKappa": "metrics.ipynb",
         "F1Score": "metrics.ipynb",
         "FBeta": "metrics.ipynb",
         "HammingLoss": "metrics.ipynb",
         "Jaccard": "metrics.ipynb",
         "Precision": "metrics.ipynb",
         "Recall": "metrics.ipynb",
         "RocAuc": "metrics.ipynb",
         "RocAucBinary": "metrics.ipynb",
         "MatthewsCorrCoef": "metrics.ipynb",
         "accuracy_multi": "metrics.ipynb",
         "AccuracyMulti": "metrics.ipynb",
         "APScoreMulti": "metrics.ipynb",
         "BrierScoreMulti": "metrics.ipynb",
         "F1ScoreMulti": "metrics.ipynb",
         "FBetaMulti": "metrics.ipynb",
         "HammingLossMulti": "metrics.ipynb",
         "JaccardMulti": "metrics.ipynb",
         "MatthewsCorrCoefMulti": "metrics.ipynb",
         "PrecisionMulti": "metrics.ipynb",
         "RecallMulti": "metrics.ipynb",
         "RocAucMulti": "metrics.ipynb",
         "mse": "metrics.ipynb",
         "MSE": "metrics.ipynb",
         "rmse": "metrics.ipynb",
         "RMSE": "metrics.ipynb",
         "mae": "metrics.ipynb",
         "MAE": "metrics.ipynb",
         "msle": "metrics.ipynb",
         "MSLE": "metrics.ipynb",
         "exp_rmspe": "metrics.ipynb",
         "ExpRMSE": "metrics.ipynb",
         "ExplainedVariance": "metrics.ipynb",
         "R2Score": "metrics.ipynb",
         "PearsonCorrCoef": "metrics.ipynb",
         "SpearmanCorrCoef": "metrics.ipynb",
         "foreground_acc": "metrics.ipynb",
         "ForegroundAcc": "metrics.ipynb",
         "Dice": "metrics.ipynb",
         "DiceMulti": "metrics.ipynb",
         "JaccardCoeff": "metrics.ipynb",
         "CorpusBLEUMetric": "metrics.ipynb",
         "Perplexity": "metrics.ipynb",
         "perplexity": "metrics.ipynb",
         "LossMetric": "metrics.ipynb",
         "LossMetrics": "metrics.ipynb",
         "init_loss": "multiloss.ipynb",
         "MultiLoss": "multiloss.ipynb",
         "MultiTargetLoss": "multiloss.ipynb",
         "MultiAvgLoss": "multiloss.ipynb",
         "MultiAvgSmoothLoss": "multiloss.ipynb",
         "MultiLossCallback": "multiloss.ipynb",
         "Learner.fit_flat_varied": "schedulers.fit_flat_varied.ipynb",
         "TEST_AUDIO": "test_utils.ipynb",
         "BatchRandTransform": "transform.ipynb",
         "free_gpu_memory": "utils.ipynb",
         "less_random": "utils.ipynb",
         "TensorImage|TensorMask.resize": "vision.augment.itemtensor.ipynb",
         "TensorImage|TensorMask.crop_pad": "vision.augment.itemtensor.ipynb",
         "RandomCrop.encodes": "vision.augment.itemtensor.ipynb",
         "CropPad.encodes": "vision.augment.itemtensor.ipynb",
         "Resize.encodes": "vision.augment.itemtensor.ipynb",
         "RandomResizedCrop.encodes": "vision.augment.itemtensor.ipynb",
         "RatioResize.encodes": "vision.augment.itemtensor.ipynb",
         "PreBatchAsItem": "vision.data.ipynb",
         "PostBatchAsItem": "vision.data.ipynb",
         "ImageCPUBlock": "vision.data.ipynb",
         "MaskCPUBlock": "vision.data.ipynb",
         "ECA": "vision.models.attention_modules.ipynb",
         "ShuffleAttention": "vision.models.attention_modules.ipynb",
         "ZPool": "vision.models.attention_modules.ipynb",
         "AttentionGate": "vision.models.attention_modules.ipynb",
         "TripletAttention": "vision.models.attention_modules.ipynb",
         "BlurPool": "vision.models.pooling.ipynb",
         "MaxBlurPool": "vision.models.pooling.ipynb",
         "ResBlock": "vision.models.xresnet.ipynb",
         "ResNeXtBlock": "vision.models.xresnet.ipynb",
         "SEBlock": "vision.models.xresnet.ipynb",
         "SEResNeXtBlock": "vision.models.xresnet.ipynb",
         "ECABlock": "vision.models.xresnet.ipynb",
         "ECAResNeXtBlock": "vision.models.xresnet.ipynb",
         "SABlock": "vision.models.xresnet.ipynb",
         "SAResNeXtBlock": "vision.models.xresnet.ipynb",
         "TABlock": "vision.models.xresnet.ipynb",
         "TAResNeXtBlock": "vision.models.xresnet.ipynb",
         "XResNet": "vision.models.xresnet.ipynb",
         "xresnet18": "vision.models.xresnet.ipynb",
         "xresnet34": "vision.models.xresnet.ipynb",
         "xresnet50": "vision.models.xresnet.ipynb",
         "xresnet101": "vision.models.xresnet.ipynb",
         "xresnext18": "vision.models.xresnet.ipynb",
         "xresnext34": "vision.models.xresnet.ipynb",
         "xresnext50": "vision.models.xresnet.ipynb",
         "xresnext101": "vision.models.xresnet.ipynb",
         "xse_resnet18": "vision.models.xresnet.ipynb",
         "xse_resnet34": "vision.models.xresnet.ipynb",
         "xse_resnet50": "vision.models.xresnet.ipynb",
         "xse_resnet101": "vision.models.xresnet.ipynb",
         "xse_resnext18": "vision.models.xresnet.ipynb",
         "xse_resnext34": "vision.models.xresnet.ipynb",
         "xse_resnext50": "vision.models.xresnet.ipynb",
         "xse_resnext101": "vision.models.xresnet.ipynb",
         "xeca_resnet18": "vision.models.xresnet.ipynb",
         "xeca_resnet34": "vision.models.xresnet.ipynb",
         "xeca_resnet50": "vision.models.xresnet.ipynb",
         "xeca_resnet101": "vision.models.xresnet.ipynb",
         "xeca_resnext18": "vision.models.xresnet.ipynb",
         "xeca_resnext34": "vision.models.xresnet.ipynb",
         "xeca_resnext50": "vision.models.xresnet.ipynb",
         "xeca_resnext101": "vision.models.xresnet.ipynb",
         "xsa_resnet18": "vision.models.xresnet.ipynb",
         "xsa_resnet34": "vision.models.xresnet.ipynb",
         "xsa_resnet50": "vision.models.xresnet.ipynb",
         "xsa_resnet101": "vision.models.xresnet.ipynb",
         "xsa_resnext18": "vision.models.xresnet.ipynb",
         "xsa_resnext34": "vision.models.xresnet.ipynb",
         "xsa_resnext50": "vision.models.xresnet.ipynb",
         "xsa_resnext101": "vision.models.xresnet.ipynb",
         "xta_resnet18": "vision.models.xresnet.ipynb",
         "xta_resnet34": "vision.models.xresnet.ipynb",
         "xta_resnet50": "vision.models.xresnet.ipynb",
         "xta_resnet101": "vision.models.xresnet.ipynb",
         "xta_resnext18": "vision.models.xresnet.ipynb",
         "xta_resnext34": "vision.models.xresnet.ipynb",
         "xta_resnext50": "vision.models.xresnet.ipynb",
         "xta_resnext101": "vision.models.xresnet.ipynb"}

modules = ["basics.py",
           "audio/core.py",
           "audio/data.py",
           "audio/augment.py",
           "audio/learner.py",
           "audio/mixup.py",
           "callback/channelslast.py",
           "callback/cutmixup.py",
           "callback/ema.py",
           "callback/lr_finder.py",
           "callback/mesa.py",
           "callback/simpleprofiler.py",
           "callback/tracker.py",
           "data/transforms.py",
           "losses.py",
           "metrics.py",
           "multiloss.py",
           "schedulers/fit_flat_varied.py",
           "test_utils.py",
           "transform.py",
           "utils/__init__.py",
           "vision/augment/itemtensor.py",
           "vision/data.py",
           "vision/models/attention_modules.py",
           "vision/models/pooling.py",
           "vision/models/xresnet.py"]

doc_url = "https://warner-benjamin.github.io/fastxtend/"

git_url = "https://github.com/warner-benjamin/fastxtend/tree/main/"

def custom_doc_links(name): return None
