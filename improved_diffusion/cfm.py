import torch as th
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import torchdiffeq
from .nn import mean_flat

from .gaussian_diffusion import GaussianDiffusion
from .losses import loss_path_similarity

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
        x0 = noise + M_o * self.biased_initialization

        # 使用 torchcfm 获取连续时间 t_cfm (0~1), 插值状态 x_t, 和目标速度场 ut
        t_cfm, x_t, ut = self.FM.sample_location_and_conditional_flow(x0, x1, t=t.float() / self.num_timesteps)     # 将 train_util.py 传入的离散 t (0~999) 映射为 CFM 需要的连续时间 (0.0 ~ 1.0)
        
        t_model = t_cfm * self.num_timesteps        # 将其映射回模型期望的尺度 (0~1000)
        # 神经网络预测当前状态下的速度场 vt
        vt = self._run_model(model, x_t, self._scale_timesteps(t_model), M_r, model_kwargs)
        
        terms = {}
        # CFM 基础损失：预测速度场与目标速度场的均方误差 (等效于 MSE)
        mse_loss = mean_flat((vt - ut) ** 2)
        
        # 计算路径相似度损失 (Path Similarity Loss)
        if self.weight_path_similarity > 0.0:
            # 扩展 t_cfm 维度以支持广播计算
            t_expanded = t_cfm.view(-1, *([1] * (len(x_t.shape) - 1)))
            
            # 在 OT-CFM 中，利用 x_t 和预测的 v_t 估算终点 x_1 (即 pred_xstart)
            # 公式: x_1 = x_t + (1 - t) * v_t
            pred_xstart = x_t + (1.0 - t_expanded) * vt
            
            terms['Path_simi_loss'] = loss_path_similarity(self.weight_path_similarity, x_start, pred_xstart)
            terms["mse"] = mse_loss + terms['Path_simi_loss']
        else:
            terms["mse"] = mse_loss

        # 保持与 train_util.py 兼容的字典键
        terms["loss"] = terms["mse"]

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
        取代了传统的逐步去噪 p_sample_loop。
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
        # model_kwargs['M_r'] = M_r

        # 初始化偏置噪声状态 (ODE 的初值 x(0))
        x0 = th.randn(*shape, device=device) + M_o * self.biased_initialization

        # 封装供 torchdiffeq 调用的 ODE 函数
        def ode_func(t_float, x):
            # t_float 是 torchdiffeq 传入的标量时间 (0~1)
            # 扩展为与 batch_size 匹配的张量，并映射回 num_timesteps 尺度
            t_batch = t_float.expand(shape[0]).to(device) * self.num_timesteps
            
            # 模型输出当前位置的速度场 \dot{x} = v(t, x)
            return self._run_model(model, x, self._scale_timesteps(t_batch), M_r, model_kwargs)

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
        
        # 返回积分到 t=1 时的最终状态
        return traj[-1]
    
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