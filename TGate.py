import torch


class TGateTransformerWrapper:
    def __init__(self, sigma_gate_attn1=-1, sigma_gate_attn2=-1):
        self.sigma_gate_attn1 = sigma_gate_attn1
        self.sigma_gate_attn2 = sigma_gate_attn2
        self.attn1_cache = {}
        self.attn2_cache = {}

    def __call__(self, inner_block, x, context=None, transformer_options=None):
        transformer_options = transformer_options or {}
        extra_options = {}
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

        extra_options["n_heads"] = inner_block.n_heads
        extra_options["dim_head"] = inner_block.d_head
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if "sigmas" in extra_options else 999999999.9
        # transformer_block key
        transformer_block = (block[0], block[1], block_index) if block is not None else None

        # in resblock
        if inner_block.ff_in:
            x_skip = x
            x = inner_block.ff_in(inner_block.norm_in(x))
            if inner_block.is_res:
                x += x_skip

        n = inner_block.norm1(x)
        if sigma < self.sigma_gate_attn1 and transformer_block in self.attn1_cache:
            # use cache
            n = self.attn1_cache[transformer_block]
            if sigma < self.sigma_gate_attn2:
                uncond, cond = n.chunk(2)
                n = (uncond + cond) / 2
                # n = torch.mean(n, dim=0, keepdim=True)
        else:
            context_attn1 = context if inner_block.disable_self_attn else None
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

            # do attn
            if block_attn1 in attn1_replace_patch:
                if context_attn1 is None:
                    context_attn1 = n
                    value_attn1 = n
                n = inner_block.attn1.to_q(n)
                context_attn1 = inner_block.attn1.to_k(context_attn1)
                value_attn1 = inner_block.attn1.to_v(value_attn1)
                n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
                n = inner_block.attn1.to_out(n)
            else:
                n = inner_block.attn1(n, context=context_attn1, value=value_attn1)

            if "attn1_output_patch" in transformer_patches:
                patch = transformer_patches["attn1_output_patch"]
                for p in patch:
                    n = p(n, extra_options)

            self.attn1_cache[transformer_block] = n  # TODO: reduce, cache times

        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if inner_block.attn2 is not None:
            n = inner_block.norm2(x)
            context_attn2 = n if inner_block.switch_temporal_ca_to_sa else context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            if sigma < self.sigma_gate_attn2 and transformer_block in self.attn2_cache:
                n = self.attn2_cache[transformer_block]
                if sigma < self.sigma_gate_attn1:
                    uncond, cond = n.chunk(2)
                    n = (uncond + cond) / 2
                    # n = torch.mean(n, dim=0, keepdim=True)
            else:
                attn2_replace_patch = transformer_patches_replace.get("attn2", {})
                block_attn2 = transformer_block
                if block_attn2 not in attn2_replace_patch:
                    block_attn2 = block

                if block_attn2 in attn2_replace_patch:
                    if value_attn2 is None:
                        value_attn2 = context_attn2
                    n = inner_block.attn2.to_q(n)
                    context_attn2 = inner_block.attn2.to_k(context_attn2)
                    value_attn2 = inner_block.attn2.to_v(value_attn2)
                    n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                    n = inner_block.attn2.to_out(n)
                else:
                    n = inner_block.attn2(n, context=context_attn2, value=value_attn2)
                if "attn2_output_patch" in transformer_patches:
                    patch = transformer_patches["attn2_output_patch"]
                    for p in patch:
                        n = p(n, extra_options)
                self.attn2_cache[transformer_block] = n

        x += n
        if inner_block.is_res:
            x_skip = x
        x = inner_block.ff(inner_block.norm3(x))
        if inner_block.is_res:
            x += x_skip

        return x


class TGateSamplerCfgRescaler:
    def __init__(self, sigma_gate_attn1=-1, sigma_gate_attn2=-1):
        self.sigma_gate = min(sigma_gate_attn1, sigma_gate_attn2)

    def __call__(self, kwds):
        sigma = kwds["timestep"].detach().cpu()[0].item() if "timestep" in kwds else 999999999.9
        if sigma < self.sigma_gate:
            return 1.0
        return kwds["cond_scale"]


class TGateSamplerFn:
    def __init__(self, sigma_gate_attn1=-1, sigma_gate_attn2=-1):
        self.sigma_gate = min(sigma_gate_attn1, sigma_gate_attn2)

    def __call__(self, kwds):
        sigma = kwds["timestep"].detach().cpu()[0].item() if "timestep" in kwds else 999999999.9
        if sigma < self.sigma_gate:
            return kwds["cond"] * kwds["cond_scale"]
        # uncond_pred + (cond_pred - uncond_pred) * cond_scale
        # x - [x - uncond_pred + (x - uncond_pred - (x - cond_pred)) * cond_scale]
        return kwds["uncond"] + (kwds["uncond"] - kwds["cond"]) * kwds["cond_scale"]


class TGateApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "attn1_start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attn2_start_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_tgate"
    CATEGORY = "TGate"

    def apply_tgate(self, model, attn1_start_at=1.0, attn2_start_at=1.0):
        model_clone = model.clone()
        sigma_attn1 = model_clone.get_model_object("model_sampling").percent_to_sigma(attn1_start_at)
        sigma_attn2 = model_clone.get_model_object("model_sampling").percent_to_sigma(attn2_start_at)
        model_clone.set_model_transformer_function(TGateTransformerWrapper(sigma_attn1, sigma_attn2))
        model_clone.set_model_sampler_cfg_rescaler(TGateSamplerCfgRescaler(sigma_attn1, sigma_attn2))
        # model_clone.set_model_sampler_cfg_function(TGateSamplerFn(sigma_attn1, sigma_attn2))
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "TGateApply": TGateApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TGateApply": "TGate Apply",
}
