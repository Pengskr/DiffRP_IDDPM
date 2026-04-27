import numpy as np
import torch as th
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import torchdiffeq
from .nn import mean_flat

from .gaussian_diffusion import GaussianDiffusion, GaussianDiffusion_without_MFF_MCA
from .losses import loss_path_similarity


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        # print(f'use_timesteps in GaussianDiffusion is {use_timesteps}')
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class SpacedDiffusion_without_MFF_MCA(SpacedDiffusion, GaussianDiffusion_without_MFF_MCA):
    """
    同时具备步长跳过功能和通道拼接逻辑的扩散类。
    利用多重继承，确保 _run_model 使用的是 GaussianDiffusion_without_MFF_MCA 的版本。
    """
    pass


class ConditionalFlowMatch(GaussianDiffusion):
    """
    基于 Conditional Flow Matching (CFM) 的路径生成器。
    继承自 GaussianDiffusion，实现对原扩散框架的无缝替换。
    """
    def __init__(self, sigma_min=1e-4, **kwargs):
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
            
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = dict(model_kwargs) # 浅拷贝，防止修改外部字典
        model_kwargs['M_r'] = M_r

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