from functools import wraps
from types import MethodType

import torch


def make_tgate_forward(
    sigma_gate=-1,
    sigma_gate_attn1=-1,
    only_cross_attention=True,
    use_cpu_cache=False,
):
    attn_cache = {}

    def tgate_forward(self, x, context=None, transformer_options={}):
        nonlocal attn_cache
        extra_options = {}
        tgate_enable = transformer_options.get("tgate_enable", False)
        tgate_clear = transformer_options.get("tgate_clear", False)
        cond_or_uncond = transformer_options.get("cond_or_uncond", [1, 0])
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if "sigmas" in extra_options else 999999999.9
        transformer_block = (block[0], block[1], block_index) if block is not None else None

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if tgate_enable and not only_cross_attention and sigma < sigma_gate_attn1 and "attn1" in attn_cache:
            # use cache
            n, chunk_count = attn_cache["attn1"]
            if sigma < sigma_gate:
                conds = n.chunk(chunk_count)
                n = sum(conds) / chunk_count
            if use_cpu_cache:
                n = n.to(x.device)
        else:
            n = self.norm1(x)
            context_attn1 = context if self.disable_self_attn else None
            value_attn1 = None

            if "attn1_patch" in transformer_patches:
                patch = transformer_patches["attn1_patch"]
                if context_attn1 is None:
                    context_attn1 = n
                value_attn1 = context_attn1
                for p in patch:
                    n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

            attn1_replace_patch = transformer_patches_replace.get("attn1", {})
            block_attn1 = transformer_block
            if block_attn1 not in attn1_replace_patch:
                block_attn1 = block

            if block_attn1 in attn1_replace_patch:
                if context_attn1 is None:
                    context_attn1 = n
                    value_attn1 = n
                n = self.attn1.to_q(n)
                context_attn1 = self.attn1.to_k(context_attn1)
                value_attn1 = self.attn1.to_v(value_attn1)
                n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
                n = self.attn1.to_out(n)
            else:
                n = self.attn1(n, context=context_attn1, value=value_attn1)

            if "attn1_output_patch" in transformer_patches:
                patch = transformer_patches["attn1_output_patch"]
                for p in patch:
                    n = p(n, extra_options)
            if tgate_enable and not only_cross_attention:
                attn_cache["attn1"] = (
                    n.cpu() if use_cpu_cache else n,
                    len(cond_or_uncond),
                )  # TODO: reduce, cache times

        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if tgate_enable and sigma < sigma_gate and "attn2" in attn_cache:
            n, chunk_count = attn_cache["attn2"]
            if only_cross_attention or sigma < sigma_gate_attn1:
                conds = n.chunk(chunk_count)
                n = sum(conds) / chunk_count
            if use_cpu_cache:
                n = n.to(x.device)
        else:
            if self.attn2 is not None:
                n = self.norm2(x)
                context_attn2 = n if self.switch_temporal_ca_to_sa else context
                value_attn2 = None
                if "attn2_patch" in transformer_patches:
                    patch = transformer_patches["attn2_patch"]
                    value_attn2 = context_attn2
                    for p in patch:
                        n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

                attn2_replace_patch = transformer_patches_replace.get("attn2", {})
                block_attn2 = transformer_block
                if block_attn2 not in attn2_replace_patch:
                    block_attn2 = block

                if block_attn2 in attn2_replace_patch:
                    if value_attn2 is None:
                        value_attn2 = context_attn2
                    n = self.attn2.to_q(n)
                    context_attn2 = self.attn2.to_k(context_attn2)
                    value_attn2 = self.attn2.to_v(value_attn2)
                    n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                    n = self.attn2.to_out(n)
                else:
                    n = self.attn2(n, context=context_attn2, value=value_attn2)

            if "attn2_output_patch" in transformer_patches:
                patch = transformer_patches["attn2_output_patch"]
                for p in patch:
                    n = p(n, extra_options)

            if tgate_enable:
                attn_cache["attn2"] = (n.cpu() if use_cpu_cache else n, len(cond_or_uncond))
        if tgate_clear:
            attn_cache.clear()

        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x

    return tgate_forward


original_sampling_function_ref = None


def sampling_function_wrapper(fn):
    @wraps(fn)
    def wrapper(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, **kwargs):
        if "sampler_pre_cfg_function" in model_options:
            uncond, cond, cond_scale = model_options["sampler_pre_cfg_function"](
                sigma=timestep, uncond=uncond, cond=cond, cond_scale=cond_scale
            )
        return fn(
            model=model,
            x=x,
            timestep=timestep,
            uncond=uncond,
            cond=cond,
            cond_scale=cond_scale,
            model_options=model_options,
            seed=seed,
            **kwargs,
        )

    wrapper._tgate_cfg_decorated = True  # type: ignore # flag to check monkey patch

    return wrapper


def monkey_patching_comfy_sampling_function():
    global original_sampling_function_ref
    from comfy import samplers

    if original_sampling_function_ref is None:
        original_sampling_function_ref = samplers.sampling_function
    # Make sure to only patch once
    if hasattr(samplers.sampling_function, "_tgate_cfg_decorated"):
        return

    samplers.sampling_function = sampling_function_wrapper(original_sampling_function_ref)


class TGateSamplerCfgRescaler:
    def __init__(self, sigma_gate=-1, sigma_gate_attn1=-1, only_cross_attention=True):
        self.sigma_gate = sigma_gate if only_cross_attention else min(sigma_gate_attn1, sigma_gate)

    def __call__(self, sigma, uncond, cond, cond_scale, **kwargs):
        sigma = sigma.detach().cpu()[0].item()
        if sigma < self.sigma_gate:
            return None, cond, 1.0
        return uncond, cond, cond_scale


class TGateProxy:
    def __init__(self, model_sampling, sigma_gate=-1) -> None:
        self.sigma_gate = sigma_gate
        self.model_sampling = model_sampling

    def __call__(self, apply_model, kwargs: dict):
        input_x = kwargs["input"]
        timestep_ = kwargs["timestep"]
        cond_or_uncond = kwargs["cond_or_uncond"]  # [0|1]
        c = kwargs["c"]
        c["transformer_options"]["tgate_enable"] = True
        sigma = timestep_[0].detach().cpu().item()

        if sigma > self.sigma_gate:
            return apply_model(input_x, timestep_, **c)

        batch_chunks = len(cond_or_uncond)
        outputs = []
        input_x_chunks = input_x.chunk(batch_chunks)
        timestep_chunks = timestep_.chunk(batch_chunks)
        c_crossattn_chunks = c["c_crossattn"].chunk(batch_chunks)
        y_chunks = None if "y" not in c else c["y"].chunk(batch_chunks)
        t = self.model_sampling.timestep(timestep_[0].detach().cpu())
        for i in cond_or_uncond:
            chunk_idx = min(i, batch_chunks - 1)
            if i == 1:
                outputs.append(torch.zeros_like(input_x_chunks[chunk_idx], dtype=input_x.dtype, device=input_x.device))
            else:
                c["c_crossattn"] = c_crossattn_chunks[chunk_idx]
                if y_chunks is not None:
                    c["y"] = y_chunks[chunk_idx]
                c["transformer_options"]["cond_or_uncond"] = [i]
                # c["transformer_options"]["sigmas"] = c["transformer_options"]["sigmas"][chunk_idx : chunk_idx + 1]
                c["transformer_options"]["tgate_clear"] = t == 0
                outputs.append(apply_model(input_x_chunks[chunk_idx], timestep_chunks[chunk_idx], **c))
        return torch.cat(outputs, dim=0)


class TGateSamplerCFG:
    def __init__(self, sigma_gate=-1):
        self.sigma_gate = sigma_gate

    def __call__(self, args: dict):
        sigma = args["sigma"][0].detach().cpu().item()
        if sigma < self.sigma_gate:
            return args["cond"]
        uncond_pred = args["uncond_denoised"]
        cond_pred = args["cond_denoised"]
        cond_scale = args["cond_scale"]
        x = args["input"]
        return x - (uncond_pred + (cond_pred - uncond_pred) * cond_scale)


class TGateApplyAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "self_attn_start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "only_cross_attention": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "use_cpu_cache": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_tgate"
    CATEGORY = "TGate"

    def apply_tgate(
        self,
        model,
        start_at=1.0,
        self_attn_start_at=1.0,
        only_cross_attention=True,
        use_cpu_cache=False,
    ):
        model_clone = model.clone()
        model_sampling = model_clone.get_model_object("model_sampling")
        sigma_gate = model_sampling.percent_to_sigma(start_at)
        sigma_gate_self_attn = model_sampling.percent_to_sigma(self_attn_start_at)

        transformer_blocks = []
        for n, m in model_clone.model.named_modules():
            if not n.endswith("transformer_blocks"):
                continue
            for tb in m:
                transformer_blocks.append(tb)

        # We inject function into the transformer_blocks instances so that
        # we can apply tgate without modifying the library source.
        for tb in transformer_blocks:
            tgate_forward = MethodType(
                make_tgate_forward(
                    sigma_gate,
                    sigma_gate_attn1=sigma_gate_self_attn,
                    only_cross_attention=only_cross_attention,
                    use_cpu_cache=use_cpu_cache,
                ),
                tb,
            )
            if hasattr(tb, "_forward"):
                tb._forward = tgate_forward
            else:
                tb.forward = tgate_forward

        sigma_gate_disable_cfg = sigma_gate if only_cross_attention else min(sigma_gate_self_attn, sigma_gate)
        model_clone.set_model_unet_function_wrapper(
            TGateProxy(
                sigma_gate=sigma_gate_disable_cfg,
                model_sampling=model_sampling,
            )
        )

        model_clone.set_model_sampler_cfg_function(
            TGateSamplerCFG(sigma_gate_disable_cfg),
            disable_cfg1_optimization=False,  # TODO: Consider whether to force enable cfg1_optimization
        )
        return (model_clone,)


class TGateApplySimple(TGateApplyAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "use_cpu_cache": ("BOOLEAN", {"default": False}),
            },
        }


class TGateApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "only_cross_attention": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "self_attn_start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_cpu_cache": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_tgate"
    CATEGORY = "TGate"

    def apply_tgate(self, model, start_at=1.0, only_cross_attention=True, self_attn_start_at=1.0, use_cpu_cache=False):
        model_clone = model.clone()
        sigma_gate = model_clone.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_gate_self_attn = model_clone.get_model_object("model_sampling").percent_to_sigma(self_attn_start_at)

        transformer_blocks = []
        for n, m in model_clone.model.named_modules():
            if not n.endswith("transformer_blocks"):
                continue
            for tb in m:
                transformer_blocks.append(tb)

        # We inject function into the transformer_blocks instances so that
        # we can apply tgate without modifying the library source.
        for tb in transformer_blocks:
            tgate_forward = MethodType(
                make_tgate_forward(
                    sigma_gate,
                    sigma_gate_attn1=sigma_gate_self_attn,
                    only_cross_attention=only_cross_attention,
                    use_cpu_cache=use_cpu_cache,
                ),
                tb,
            )
            # update_wrapper(tgate_forward, tb._forward)
            if hasattr(tb, "_forward"):
                tb._forward = tgate_forward
            else:
                tb.forward = tgate_forward
        model_clone.model_options["sampler_pre_cfg_function"] = TGateSamplerCfgRescaler(
            sigma_gate, sigma_gate_self_attn, only_cross_attention
        )
        model_clone.model_options["transformer_options"]["tgate_enable"] = True
        monkey_patching_comfy_sampling_function()
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "TGateApply": TGateApply,
    "TGateApplySimple": TGateApplySimple,
    "TGateApplyAdvanced": TGateApplyAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TGateApply": "TGate Apply(Deprecated)",
    "TGateApplySimple": "TGate Apply",
    "TGateApplyAdvanced": "TGate Apply Advanced",
}
