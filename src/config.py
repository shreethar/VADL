def build_config():
    cfg = type('', (), {})()
    cfg.feat_dim = 512
    cfg.dropout_gat = 0.6
    cfg.out_dim = 32
    cfg.alpha = 0.1
    cfg.train_bs = 1
    cfg.val_bs = 1
    cfg.workers = 4
    cfg.device = "cuda:0"
    cfg.load_ckpt = False
    cfg.WANDB = False
    cfg.eval_on_cpu = False
    cfg.head_num = 4
    cfg.dataset = 'ucfcrime-detection'
    cfg.has_feature_input = False
    cfg.gamma = 0.6
    cfg.bias = 0.2
    cfg.norm = True
    cfg.temporal = True
    # training settings
    cfg.temp = 0.09
    cfg.lamda = 1
    cfg.seed = 11 #9
    # test settings
    cfg.test_bs = 1
    # prompt
    cfg.preprompt = False
    cfg.backbone = 'ViT-B/16'
    cfg.mask_rate = 0.55
    cfg.std_init = 0.01
    cfg.head_num = 4 #4
    cfg.cls_hidden = 128

    cfg.entity_name = "your-name-goes-here"
    cfg.project_name = "video-anomaly-detection-localization"

    cfg.epochs = 50
    cfg.max_seqlen = 256
    cfg.lr_detect = 1e-3
    cfg.lr_localize = 1e-3
    cfg.accum_steps = 16
    cfg.patience = 10
    cfg.save_path = 'src/results/model_files/best_model.pth'
    cfg.onnx_model_save_path = 'src/results/model_files/VADL_model_ONNX.onnx'

    cfg.val_cfm_path = 'src/results/images/Validation Set Confusion Matrix.png'
    cfg.test_cfm_path = 'src/results/images/Test Set Confusion Matrix.png'

    cfg.chunk_size = 256
    cfg.frame_queue_max = 2
    cfg.result_queue_max = 2
    cfg.heatmap_threshold = -0.29
    cfg.anomaly_prob_threshold = 0.5
    cfg.merge_distance_threshold = 60

    


    return cfg