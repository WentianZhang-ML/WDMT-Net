#encoding:utf-8

# training settings
class BaselineConfig(object):
    dataset = {
        'IDRiD': '/apdcephfs/share_1290796/waltszhang/Anomaly_detection/labels/pixel_level/IDRiD/',
        'ADAM': '/apdcephfs/share_1290796/waltszhang/Anomaly_detection/labels/pixel_level/ADAM/',  
    }

    savedir = '/apdcephfs/share_1290796/waltszhang/MICCAI/models_logs/'

    batch_size_base = 32  
    lr_base = 1e-3 
    n_epochs_base = 1000 
    eval_epoch_base = 1
    weight_decay_base = 5e-5
