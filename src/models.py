from monai.networks.nets import UNETR, UNet

from hippo_data import IMAGE_SIZE


def build_model(cfg):
    arch = cfg.get("arch", "unet")
    if arch == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=tuple(cfg["channels"]),
            strides=(2,) * (len(cfg["channels"]) - 1),
            num_res_units=cfg["res_units"],
        )
    if arch == "unetr":
        return UNETR(
            in_channels=1,
            out_channels=3,
            img_size=IMAGE_SIZE,
            feature_size=cfg.get("feature_size", 16),
            hidden_size=cfg.get("hidden_size", 768),
            mlp_dim=cfg.get("mlp_dim", 3072),
            num_heads=cfg.get("num_heads", 12),
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    raise ValueError(f"Nieznana architektura: {arch}")