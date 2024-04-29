from types import MethodType


def make_tgate_forward(sigma_gate=-1, sigma_gate_attn1=-1, only_cross_attention=True):
    attn_cache = {}

    def tgate_forward(self, x, context=None, transformer_options={}):
        nonlocal attn_cache
        extra_options = {}
        tgate_enable = transformer_options.get("tgate_enable", False)
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
            if tgate_enable:
                attn_cache["attn1"] = (n, len(cond_or_uncond))  # TODO: reduce, cache times

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
                attn_cache["attn2"] = (n, len(cond_or_uncond))

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
    # model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None
    def wrapper(model, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None, **kwargs):
        model_options = model_options or {}
        if "sampler_cfg_rescaler" in model_options:
            cond_scale = model_options["sampler_cfg_rescaler"]({"cond_scale": cond_scale, "timestep": timestep})
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

    return wrapper


def monkey_patching_comfy_sampling_function():
    global original_sampling_function_ref
    from comfy import samplers

    if original_sampling_function_ref is not None and original_sampling_function_ref is not samplers.sampling_function:
        return
    original_sampling_function_ref = samplers.sampling_function
    samplers.sampling_function = sampling_function_wrapper(samplers.sampling_function)


class TGateSamplerCfgRescaler:
    def __init__(self, sigma_gate=-1, sigma_gate_attn1=-1, only_cross_attention=True):
        self.sigma_gate = sigma_gate if only_cross_attention else min(sigma_gate_attn1, sigma_gate)

    def __call__(self, kwds):
        sigma = kwds["timestep"].detach().cpu()[0].item() if "timestep" in kwds else 999999999.9
        if sigma < self.sigma_gate:
            return 1.0
        return kwds["cond_scale"]


class TGateApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "only_cross_attention": ("BOOLEAN", {"default": True}),
            },
            "optional": {"self_attn_start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_tgate"
    CATEGORY = "TGate"

    def apply_tgate(self, model, start_at=1.0, only_cross_attention=True, self_attn_start_at=1.0):
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
                ),
                tb,
            )
            # update_wrapper(tgate_forward, tb._forward)
            tb._forward = tgate_forward
        model_clone.model_options["sampler_cfg_rescaler"] = TGateSamplerCfgRescaler(
            sigma_gate, sigma_gate_self_attn, only_cross_attention
        )
        model_clone.model_options["transformer_options"]["tgate_enable"] = True
        monkey_patching_comfy_sampling_function()
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "TGateApply": TGateApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TGateApply": "TGate Apply",
}
