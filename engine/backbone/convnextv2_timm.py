import os
import torch.nn as nn

from ..core import register


@register()
class TimmConvNeXtV2(nn.Module):
    def __init__(
        self,
        name='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
        pretrained=True,
        return_idx=[1, 2, 3],
        drop_path_rate=0.1,
        freeze_at=-1,
        freeze_norm=False,
        local_weight_path='',
        cache_dir='',
    ):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                'TimmConvNeXtV2 requires timm. Please install it with: pip install -U timm safetensors huggingface_hub'
            ) from e

        self.return_idx = tuple(return_idx)

        create_kwargs = dict(
            features_only=True,
            out_indices=self.return_idx,
            drop_path_rate=drop_path_rate,
        )

        if cache_dir:
            create_kwargs['cache_dir'] = cache_dir

        # 方式 1：从本地 safetensors / pth 文件加载
        if local_weight_path:
            if not os.path.exists(local_weight_path):
                raise FileNotFoundError(f'local_weight_path does not exist: {local_weight_path}')

            self.model = timm.create_model(
                name,
                pretrained=True,
                pretrained_cfg_overlay={'file': local_weight_path},
                **create_kwargs,
            )

        # 方式 2：timm 自动下载并加载预训练权重
        else:
            self.model = timm.create_model(
                name,
                pretrained=pretrained,
                **create_kwargs,
            )

        self._out_channels = self.model.feature_info.channels()
        self._out_strides = self.model.feature_info.reduction()

        if freeze_norm:
            self._freeze_norm()

        if freeze_at >= 0:
            self._freeze_stages(freeze_at)

    def _freeze_norm(self):
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _freeze_stages(self, freeze_at):
        # timm 的 FeatureListNet 已经包装过 backbone，最稳妥的是按参数名前缀冻结前几层
        # 对 ConvNeXt V2 来说，一般包含 stem / stages.0 / stages.1 / stages.2 / stages.3
        freeze_prefixes = []

        if freeze_at >= 0:
            freeze_prefixes.append('model.stem')
            freeze_prefixes.append('stem')

        for i in range(freeze_at):
            freeze_prefixes.append(f'model.stages.{i}')
            freeze_prefixes.append(f'stages.{i}')

        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefixes):
                param.requires_grad = False


    def forward(self, x):
        return [feat.contiguous() for feat in self.model(x)]
