import ml_collections


def base()                                : 
    config                                = ml_collections.ConfigDict()
    config.train                          = ml_collections.ConfigDict()
    config.train.run_name                 = "eit_anchor"
    config.train.logdir                   = "/logs/wzx_data/emo_grpo"
    config.train.seed                     = 42
    config.train.mixed_precision          = "fp16"
    config.train.max_grad_norm            = 1.0
    config.train.print_freq               = 50  # 50 steps
    config.train.save_freq                = 50  # 50 epoch
    config.train.batch_size               = 64  # 64
    config.train.epochs                   = 200
    config.train.find_unused_parameters   = True
    # ------------
    #     model
    config.model                          = ml_collections.ConfigDict()
    config.model.sd_feature_dim           = 4096  #
    # ------------
    config.eit_ckpt                       = ""
    config.data_cache_path                = "?"
    config.prompt_cache_path              = "?"
    config.config_dir                     = "configs"
    config.lr                             = 1e-4
    config.scale_factor                   = 1.5
    config.anchor_weight                  = 0.2
    config.enable_density                 = False
    config.num_workers                    = 4
    config.test_size                      = 0.01
    config.allow_tf32                     = True  # A100 显卡：启用 TF32 加速训练

    return config


def sdxl_eit_scale1_5()                   : 
    config                                = base()
    config.train.run_name                 = "sdxl_eit_scale1_5"
    config.train.logdir                   = "/logs/wzx_data/sdxl_eit_scale1_5"
    config.enable_density                 = False
    config.anchor_weight                  = 0.0
    config.scale_factor                   = 1.5
    config.model.sd_feature_dim           = 2048
    config.train.batch_size               = 64  # x8 gus
    config.data_cache_path                = "data/data-sdxl-fp16.safetensors"
    config.prompt_cache_path              = "data/data-sdxl-fp16.prompt.txt"
    return config


def sdxl_eit_scale1_5_anchor()            : 
    config                                = base()
    config.train.run_name                 = "sdxl_eit_scale1_5_anchor"
    config.train.logdir                   = "/logs/wzx_data/sdxl_eit_scale1_5_anchor"
    config.enable_density                 = False
    config.anchor_weight                  = 0.2
    config.scale_factor                   = 1.5
    config.model.sd_feature_dim           = 2048
    config.train.batch_size               = 64  # x8 gus
    config.data_cache_path                = "data/data-sdxl-fp16.safetensors"
    config.prompt_cache_path              = "data/data-sdxl-fp16.prompt.txt"
    return config


def sd3tiny_scale1_1_anchor()            : 
    # python scripts/02_train_eit.py --config=configs/eit.py:sd3tiny_scale1_1_anchor
    config                                = base()
    config.train.run_name                 = "sd3tiny_scale1_1_anchor"
    config.train.logdir                   = "/logs/wzx_data/sd3tiny_scale1_1_anchor"
    config.enable_density                 = False
    config.anchor_weight                  = 0.2
    config.scale_factor                   = 1.
    config.model.sd_feature_dim           = 32
    config.train.batch_size               = 64  # x8 gus
    config.data_cache_path                = "data/data-sd3-fp16.safetensors"
    config.prompt_cache_path              = "data/data-sd3-fp16.txt"

    config.train.epochs                   =1
    return config

def sd35M_scale1_1_anchor()            : 
    # torchrun --nproc_per_node=8  scripts/02_train_eit.py --config=configs/eit.py:sd35M_scale1_1_anchor
    config                                = base()
    config.train.run_name                 = "sd35M_scale1_1_anchor"
    config.train.logdir                   = "/logs/wzx_data/sd35M_scale1_1_anchor"
    config.enable_density                 = False
    config.anchor_weight                  = 0.2
    config.scale_factor                   = 1.
    config.model.sd_feature_dim           = 4096
    config.train.batch_size               = 64  # x8 gus
    config.data_cache_path                = "data/data-sd3-fp16.safetensors"
    config.prompt_cache_path              = "data/data-sd3-fp16.txt"

    config.train.epochs                   =200
    return config


def get_config(name)                      : 
    return globals()[name]()
