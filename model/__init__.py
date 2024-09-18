import os
from functools import partial

import torch

from .vmamba import VSSM
from .lamamba import LaMambaDiff
try:
    from .heat import HeatM
except:
    HeatM = None

# try:
#     from .vim import build_vim
# except Exception as e:
#     build_vim = lambda *args, **kwargs: None


# still on developing...
def build_vssm_model(config,args, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["LaMamba-Diff"]:
        model = LaMambaDiff(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=args.num_classes, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            unconditional=args.unconditional,
            sd_unet_decoder_design=config.MODEL.VSSM.SD_UNET_DECODER_DESIGN,
            bottleneck_depth=config.MODEL.VSSM.BOTTLENECK_DEPTH,
            skip_concat=config.MODEL.SKIP_CONCAT,
            # ===================
            in_resolution=args.image_size//8,
            num_heads=args.num_heads,
            window_size=config.MODEL.VSSM.WINDOW_SIZE,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            fused_window_process=False,
            # ===================
            continuous_scan=config.MODEL.VSSM.CONTINUOUS_SCAN,
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        return model

    return None


# used for analyze
def build_vssm_models_(cfg="vssm_tiny", ckpt=True, only_backbone=False, with_norm=True,
    CFGS = dict(
        vssm_tiny=dict(
            model=dict(
                depths=[2, 2, 9, 2], 
                dims=96, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.1, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth"),
        ),
        vssm_small=dict(
            model=dict(
                depths=[2, 2, 27, 2], 
                dims=96, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.3, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth"),
        ),
        vssm_base=dict(
            model=dict(
                depths=[2, 2, 27, 2], 
                dims=128, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.6, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ),  
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmbase/ckpt_epoch_260.pth"),
        ),
    ),
    ckpt_key="model",
    **kwargs):
    if cfg not in CFGS:
        return None
    
    model_params = CFGS[cfg]["model"]
    model_ckpt = CFGS[cfg]["ckpt"]

    model = VSSM(**model_params)
    if only_backbone:
        if with_norm:
            def forward(self: VSSM, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = self.classifier.norm(x)
                x = x.permute(0, 3, 1, 2).contiguous()
                return x
            model.forward = partial(forward, model)
            del model.classifier.norm
            del model.classifier.head
            del model.classifier.avgpool
        else:
            def forward(self: VSSM, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = x.permute(0, 3, 1, 2).contiguous()
                return x
            model.forward = partial(forward, model)
            del model.classifier

    if ckpt:
        ckpt = model_ckpt
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = model.load_state_dict(_ckpt[ckpt_key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    return model





def build_model(config, is_pretrain=False):
    model = None
    
    if model is None:
        model = build_vssm_model(config, is_pretrain)
    if model is None:
        model = build_heat_model(config, is_pretrain)
    if model is None:
        model = build_mmpretrain_models(config.MODEL.TYPE, ckpt=config.MODEL.MMCKPT)
    if model is None:
        model = build_vim(config, is_pretrain)
    return model




