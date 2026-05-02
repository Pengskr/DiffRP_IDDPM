import torch as th
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import torchdiffeq
from .nn import mean_flat

from .gaussian_diffusion import GaussianDiffusion

class ConditionalFlowMatch(GaussianDiffusion):
    """
    基于 Conditional Flow Matching (CFM) 的路径生成器。
    继承自 GaussianDiffusion，实现对原扩散框架的无缝替换。
    """
    def __init__(self, use_timesteps, sigma_min=1e-4, **kwargs):
        # 调用父类初始化，保留 betas, loss_type, biased_initialization 等参数配置
        super().__init__(**kwargs)
        
        # 使用 torchcfm 提供标准的最优传输 (OT) 概率路径和速度场
        self.FM = ConditionalFlowMatcher(sigma=sigma_min)

    def training_losses(self, model, M_o, M_r, x_start, t, model_kwargs=None, noise=None):
        """
        计算 CFM 的速度场匹配损失 (Vector Field Matching Loss)。
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        x1 = x_start  # 目标数据分布 (最优路径)
        if noise is None:
            noise = th.randn_like(x1)
            
        # 偏置初始化：x_0 ~ N(M_o * bias, I)
        # x0 = noise + M_o * self.biased_initialization

        # 空间自适应偏置先验 (Spatially Adaptive Biased Prior) 或 非对称信息注入 (Asymmetric Informed Prior):当 M_o 为 -1 时，给予 负 的偏置；当 M_o 为 1 时，给予 正 的偏置
        # bias_map = th.where(M_o == -1, -10*self.biased_initialization, self.biased_initialization)
        # x0 = th.randn(*shape, device=device) + bias_map

        # 空间掩码流 (Spatially Masked Flow) 或 局部冻结向量场 (Locally Frozen Vector Field)
        # 障碍物区域 (M_o == -1) 强制设为 -1 (绝对不流动)
        # 自由区域 (M_o == 1) 保持高斯噪声 (可保留微弱的正向引导 biased_initialization)
        x0 = th.where(M_o == -1, 
                      th.tensor(-1.0, dtype=x1.dtype, device=x1.device), 
                      noise + self.biased_initialization
                )

        # 使用 torchcfm 获取连续时间 t_cfm (0~1), 插值状态 x_t, 和目标速度场 ut
        t_cfm, x_t, ut = self.FM.sample_location_and_conditional_flow(x0, x1, t=t.float() / self.num_timesteps)     # 将 train_util.py 传入的离散 t (0~999) 映射为 CFM 需要的连续时间 (0.0 ~ 1.0)
        # 强制将障碍物区域的目标流速设为 0，教导网络“这里不需要流动”
        ut = th.where(M_o == -1, th.zeros_like(ut), ut)
        
        # 神经网络预测当前状态下的速度场 vt
        t_model = t_cfm * self.num_timesteps        # 将其映射回模型期望的尺度 (0~1000)
        vt = self._run_model(model, x_t, self._scale_timesteps(t_model), M_r, model_kwargs)
        

        # CFM空间加权损失 (Spatially Weighted Loss)
        # 定义惩罚权重 (你可以根据实验效果调整 weight_obs 的大小，推荐 10.0 ~ 50.0)
        weight_obs = 50.0   # 对障碍物区域的误差施加极高惩罚
        weight_free = 1.0   # 自由区域保持正常权重
        # 构建与特征图同尺寸的权重掩码
        loss_weight_map = th.where(M_o == -1, weight_obs, weight_free)
        terms = {}
        #  加权均方误差 (Weighted MSE)
        terms["Weighted MSE"] = mean_flat(loss_weight_map * (vt - ut) ** 2)

        # 保持与 train_util.py 兼容的字典键
        terms["loss"] = terms["Weighted MSE"]

        return terms

    @th.no_grad()
    def sample_loop(
        self, 
        model, 
        M_o, 
        M_r, 
        shape, 
        clip_denoised=True,
        device=None, 
        n_integration_steps=10, 
        model_kwargs=None, 
    ):
        """
        使用 ODE 求解器 (如 Euler 法) 进行反向采样。
        """
        if device is None:
            device = next(model.parameters()).device
            
        M_o = M_o.to(device)
        if M_r is not None:
            M_r = M_r.to(device)
            
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = dict(model_kwargs) # 浅拷贝，防止修改外部字典

        # 推理阶段的硬约束先验
        noise = th.randn(*shape, device=device)
        x0 = th.where(
            M_o == -1, 
            th.tensor(-1.0, dtype=noise.dtype, device=device), 
            noise + self.biased_initialization
        )

        # 封装供 torchdiffeq 调用的 ODE 函数
        def ode_func(t_float, x):
            t_batch = t_float.expand(shape[0]).to(device) * self.num_timesteps
            
            # 模型输出当前位置的预测速度场 \dot{x} = v(t, x)
            vt = self._run_model(model, x, self._scale_timesteps(t_batch), M_r, model_kwargs)
            
            # 局部冻结预测速度场 (物理阻断)
            # 无论网络预测了什么，我们在积分前强制把障碍物区域的速度归零。这样在整个 Euler 积分过程中，墙内的像素值会死死钉在 x0 的初值 (-1.0) 上！
            vt = th.where(M_o == -1, th.zeros_like(vt), vt)
            
            return vt

        # 积分时间区间：从 0 到 1
        t_span = th.linspace(0, 1, n_integration_steps + 1).to(device)

        # 使用常微分方程求解器沿速度场积分
        traj = torchdiffeq.odeint(
            ode_func,
            x0,
            t_span,
            atol=1e-4,
            rtol=1e-4,
            method="euler",  # 欧拉法速度快，适合 Flow Matching 推理
        )
        
        final_sample = traj[-1]
        
        # 终极边界消除 (Post-processing)
        # 由于 CNN 的卷积核感受野会跨越墙壁内外，导致靠近墙根的自由区域可能会有一点点数值溢出。
        # 再加上一道物理掩码，彻底斩断任何穿模的可能性。
        final_sample = th.where(
            M_o == -1, 
            th.tensor(-1.0, dtype=final_sample.dtype, device=device), 
            final_sample
        )
        
        return final_sample
    
    def _run_model(self, model, x, t, M_r, model_kwargs):
        """
        默认的模型调用方式：特征注入 (MFF+MCA范式)
        """
        return model(x, t, M_r=M_r, **model_kwargs)


class ConditionalFlowMatch_without_MFF_MCA(ConditionalFlowMatch):
    """
    不使用 MFF/MCA 模块，而是将 M_r 拼接到输入通道中 (Concat 范式)
    """
    def _run_model(self, model, x, t, M_r, model_kwargs):
        if M_r is None:
            raise ValueError("M_r is required for concatenated conditioning.")
        
        # 将 x 和 M_r 在通道维度拼接 [B, 4, H, W]
        model_input = th.cat([x, M_r.type(x.dtype)], dim=1)
        
        return model(model_input, t, **model_kwargs)