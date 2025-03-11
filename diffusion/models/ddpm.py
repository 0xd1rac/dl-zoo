import torch 
import torch.nn as nn
from torch import Tensor

class LinearNoiseScheduler:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float):
        """
        Linear noise scheduler that computes beta values linearly from beta_start to beta_end.
        Pre-computes alpha cumulative products and their square roots.
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        # Compute square roots of cumulative products
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Adds noise to the original images based on the timestep t.
        
        :param original: Original images tensor of shape [B, C, H, W]
        :param noise: Noise tensor of the same shape as original.
        :param t: Tensor of time steps of shape [B] (each value between 0 and num_timesteps-1).
        :return: Noisy images tensor.
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        # Select the precomputed factors for each sample based on t.
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        # Unsqueeze to allow broadcasting to the image shape.
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        # Compute the original image from noisy input xt
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0: return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod[t-1]) / (1. - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0


class TimeEmbedding(nn.Module):
    """
    Module to convert 1D time step tensor into a sinusoidal time embedding.
    
    The embedding is computed using the formula from the DDPM paper.
    """
    def __init__(self, temb_dim: int) -> None:
        """
        Initialize the TimeEmbedding module.
        
        :param temb_dim: Dimension of the time embedding (must be divisible by 2)
        """
        super(TimeEmbedding, self).__init__()
        if temb_dim % 2 != 0:
            raise ValueError("Time embedding dimension must be divisible by 2")
        self.temb_dim = temb_dim

    def forward(self, time_steps: Tensor) -> Tensor:
        """
        Compute sinusoidal embeddings for the given time steps.
        
        :param time_steps: 1D tensor of shape [batch_size] with time step indices.
        :return: Tensor of shape [batch_size, temb_dim] with the time embeddings.
        """
        half_dim = self.temb_dim // 2
        # Compute scaling factor: factor = 10000^(2i/temb_dim)
        factor = 10000 ** (torch.arange(0, half_dim, dtype=torch.float32, device=time_steps.device) / half_dim)
        # Expand time_steps to shape [B, half_dim] and apply scaling.
        t_emb = time_steps[:, None].repeat(1, half_dim) / factor
        # Concatenate sine and cosine components.
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return t_emb


class DownBlock(nn.Module):
    """
    Downsampling block with integrated time conditioning and attention.
    
    The block performs:
      1. A ResNet block with time embedding.
      2. An attention block to capture long-range dependencies.
      3. An optional downsampling via strided convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int,
                 down_sample: bool = True, num_heads: int = 4, num_layers: int = 1) -> None:
        """
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param t_emb_dim: Dimension of the time embedding.
        :param down_sample: Whether to perform downsampling at the end of the block.
        :param num_heads: Number of heads for multi-head attention.
        :param num_layers: Number of ResNet/attention layers within this block.
        """
        super(DownBlock, self).__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample

        # First convolution block(s) for the ResNet part.
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels,
                          out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for i in range(num_layers)
        ])

        # Layers to transform the time embedding to match the number of channels.
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        # Second convolution block(s) for the ResNet part.
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for _ in range(num_layers)
        ])

        # GroupNorm layers applied before the attention blocks.
        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        # Multi-head attention layers.
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # 1x1 convolution layers to adjust the residual connection if needed.
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        # Downsampling convolution (if enabled).
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          kernel_size=4, stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass of the DownBlock.
        
        :param x: Input tensor of shape [B, in_channels, H, W].
        :param t_emb: Time embedding tensor of shape [B, t_emb_dim].
        :return: Processed tensor.
        """
        out = x
        for i in range(self.num_layers):
            # --- ResNet Block ---
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            # Add time conditioning (broadcasted spatially).
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # --- Attention Block ---
            B, C, H, W = out.shape
            # Reshape to [B, H*W, C] for attention.
            attn_input = out.reshape(B, C, H * W)
            attn_input = self.attention_norms[i](attn_input)
            attn_input = attn_input.transpose(1, 2)
            attn_output, _ = self.attentions[i](attn_input, attn_input, attn_input)
            # Reshape back to [B, C, H, W].
            attn_output = attn_output.transpose(1, 2).reshape(B, C, H, W)
            out = out + attn_output

        # --- Downsampling ---
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    """
    Middle block with attention and time conditioning.
    
    The block consists of:
      1. An initial ResNet block with time embedding.
      2. For each layer: an attention block followed by another ResNet block with time embedding.
    """
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int,
                 num_heads: int = 4, num_layers: int = 1) -> None:
        """
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param t_emb_dim: Dimension of the time embedding.
        :param num_heads: Number of heads for multi-head attention.
        :param num_layers: Number of attention blocks (and subsequent ResNet blocks).
        """
        super(MidBlock, self).__init__()
        self.num_layers = num_layers

        # Create (num_layers + 1) ResNet blocks.
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels,
                          out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for i in range(num_layers + 1)
        ])

        # Time embedding layers for each ResNet block.
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])

        # Second convolution blocks for each ResNet block.
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for _ in range(num_layers + 1)
        ])

        # Attention normalization layers.
        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        # Multi-head attention layers.
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # 1x1 convolution layers for residual connections.
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass of the MidBlock.
        
        :param x: Input tensor of shape [B, in_channels, H, W].
        :param t_emb: Time embedding tensor of shape [B, t_emb_dim].
        :return: Processed tensor.
        """
        out = x
        # --- First ResNet Block ---
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # --- Attention and subsequent ResNet Blocks ---
        for i in range(self.num_layers):
            # Attention Block.
            B, C, H, W = out.shape
            attn_input = out.reshape(B, C, H * W)
            attn_input = self.attention_norms[i](attn_input)
            attn_input = attn_input.transpose(1, 2)
            attn_output, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_output = attn_output.transpose(1, 2).reshape(B, C, H, W)
            out = out + attn_output

            # ResNet Block.
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    """
    Upsampling block with integrated attention and time conditioning.
    
    The block performs:
      1. Upsampling via ConvTranspose2d (if enabled).
      2. Concatenation with the corresponding DownBlock output (skip connection).
      3. A ResNet block with time embedding.
      4. An attention block.
    """
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int,
                 up_sample: bool = True, num_heads: int = 4, num_layers: int = 1) -> None:
        """
        :param in_channels: Number of input channels (after concatenation of skip connection).
        :param out_channels: Number of output channels.
        :param t_emb_dim: Dimension of the time embedding.
        :param up_sample: Whether to perform upsampling.
        :param num_heads: Number of heads for multi-head attention.
        :param num_layers: Number of ResNet/attention layers within this block.
        """
        super(UpBlock, self).__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample

        # ResNet block: first convolution layers.
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels,
                          out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for i in range(num_layers)
        ])

        # Time embedding layers.
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])

        # ResNet block: second convolution layers.
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1)
            )
            for _ in range(num_layers)
        ])

        # Attention normalization layers.
        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])

        # Multi-head attention layers.
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # 1x1 convolution layers for residual connections.
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        # Upsampling layer via transposed convolution (if enabled).
        # Assumes the input channels from the skip connection are half of in_channels.
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 kernel_size=4, stride=2, padding=1) if self.up_sample else nn.Identity()

    def forward(self, x: Tensor, out_down: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass of the UpBlock.
        
        :param x: Input tensor from the previous layer.
        :param out_down: Skip connection tensor from the corresponding DownBlock.
        :param t_emb: Time embedding tensor of shape [B, t_emb_dim].
        :return: Processed tensor.
        """
        # Upsample the input.
        x = self.up_sample_conv(x)
        # Concatenate skip connection along the channel dimension.
        x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # ResNet Block.
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Attention Block.
            B, C, H, W = out.shape
            attn_input = out.reshape(B, C, H * W)
            attn_input = self.attention_norms[i](attn_input)
            attn_input = attn_input.transpose(1, 2)
            attn_output, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_output = attn_output.transpose(1, 2).reshape(B, C, H, W)
            out = out + attn_output

        return out


class Unet(nn.Module):
    """
    U-Net model comprising DownBlocks, MidBlocks, and UpBlocks.
    
    This architecture is designed for diffusion models where time conditioning
    is integrated throughout the network.
    """
    def __init__(self, model_config: dict) -> None:
        """
        Initialize the U-Net model.
        
        :param model_config: Dictionary with configuration parameters:
            - im_channels: Number of input image channels.
            - down_channels: List of channel sizes for DownBlocks.
            - mid_channels: List of channel sizes for MidBlocks.
            - time_emb_dim: Dimension of the time embedding.
            - down_sample: List of booleans indicating whether to downsample at each DownBlock.
            - num_down_layers: Number of ResNet/attention layers in DownBlocks.
            - num_mid_layers: Number of attention layers in MidBlocks.
            - num_up_layers: Number of ResNet/attention layers in UpBlocks.
        """
        super(Unet, self).__init__()
        im_channels: int = model_config['im_channels']
        self.down_channels: list = model_config['down_channels']
        self.mid_channels: list = model_config['mid_channels']
        self.t_emb_dim: int = model_config['time_emb_dim']
        self.down_sample: list = model_config['down_sample']
        self.num_down_layers: int = model_config['num_down_layers']
        self.num_mid_layers: int = model_config['num_mid_layers']
        self.num_up_layers: int = model_config['num_up_layers']
        
        # Validate channel configuration.
        assert self.mid_channels[0] == self.down_channels[-1], "Mismatch between mid_channels and down_channels."
        assert self.mid_channels[-1] == self.down_channels[-2], "Mismatch between mid_channels and down_channels."
        assert len(self.down_sample) == len(self.down_channels) - 1, "Mismatch in down_sample list length."
        
        # Time projection module: further processes the sinusoidal time embedding.
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        # Reverse down_sample list for upsampling.
        self.up_sample: list = list(reversed(self.down_sample))
        # Initial convolution to project image channels to first down channel.
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        # Build DownBlocks.
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                          down_sample=self.down_sample[i], num_layers=self.num_down_layers)
            )
        
        # Build MidBlocks.
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                         num_layers=self.num_mid_layers)
            )
        
        # Build UpBlocks.
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                        self.t_emb_dim, up_sample=self.down_sample[i], num_layers=self.num_up_layers)
            )
        
        # Output normalization and final convolution.
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass of the U-Net model.
        
        :param x: Input image tensor of shape [B, im_channels, H, W].
        :param t: Tensor containing time step indices.
        :return: Output image tensor of shape [B, im_channels, H, W].
        """
        # Initial convolutional projection.
        out: Tensor = self.conv_in(x)
        
        # Create time embeddings using the TimeEmbedding module.
        time_emb_module = TimeEmbedding(self.t_emb_dim).to(x.device)
        t_emb: Tensor = time_emb_module(t.to(torch.float32))
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        # Pass through DownBlocks.
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)
        
        # Process through MidBlocks.
        for mid in self.mids:
            out = mid(out, t_emb)
        
        # Process through UpBlocks, using stored skip connections.
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
        
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out



