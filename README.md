# ComfyUI_TGate

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) reference implementation for [T-GATE](https://github.com/HaozheLiu-ST/T-GATE).

## Example workflows

The [examples directory](./examples/) has workflow example. There are images generated with and without T-GATE in the [assets](./assets/) folder.

![example](./examples/tgate_workflow_example.png)

Origin result

![origin_result](./assets/origin_result.png)

T-GATE result (start at 0.5, only_cross_attention)
![tgate_result](./assets/tgate_result.png)

## INSTALL
```bash
git clone https://github.com/JettHu/ComfyUI_TGate
git apply tgate.patch   # need patch yet, PR in progress
```
## Major Features

- Training-Free.
- Friendly support CNN-based U-Net, Transformer, and Consistency Model
- 10%-50% speed up for different diffusion models.


## Nodes reference

### TGateApply

#### Inputs
- **model**, model loaded by `Load Checkpoint` and other MODEL loaders.

#### Configuration parameters
- **start_at**, this is the timestepping. Defines at what percentage point of the generation to start use the T-GATE cache.
- **only_cross_attention**, default only used to cache the cross-attention output, ref to [issues](https://github.com/HaozheLiu-ST/T-GATE/issues/8#issuecomment-2061379798)


#### Optional configuration
- **self_attn_start_at**, only takes effect when `only_cross_attention` is `false`, timestepping too. Defines at what percentage point of the generation to start use the T-GATE cache.

## Performance (from [T-GATE](https://github.com/HaozheLiu-ST/T-GATE))
| Model                 | MACs     | Param     | Latency | Zero-shot 10K-FID on MS-COCO |
|-----------------------|----------|-----------|---------|---------------------------|
| SD-1.5                | 16.938T  | 859.520M  | 7.032s  | 23.927                    |
| SD-1.5 w/ TGATE       | 9.875T   | 815.557M  | 4.313s  | 20.789                    |
| SD-2.1                | 38.041T  | 865.785M  | 16.121s | 22.609                    |
| SD-2.1 w/ TGATE       | 22.208T  | 815.433 M | 9.878s  | 19.940                    |
| SD-XL                 | 149.438T | 2.570B    | 53.187s | 24.628                    |
| SD-XL w/ TGATE        | 84.438T  | 2.024B    | 27.932s | 22.738                    |
| Pixart-Alpha          | 107.031T | 611.350M  | 61.502s | 38.669                    |
| Pixart-Alpha w/ TGATE | 65.318T  | 462.585M  | 37.867s | 35.825                    |
| DeepCache (SD-XL)     | 57.888T  | -         | 19.931s | 23.755                    |
| DeepCache w/ TGATE    | 43.868T  | -         | 14.666s | 23.999                    |
| LCM (SD-XL)           | 11.955T  | 2.570B    | 3.805s  | 25.044                    |
| LCM w/ TGATE          | 11.171T  | 2.024B    | 3.533s  | 25.028                    |
| LCM (Pixart-Alpha)    | 8.563T   | 611.350M  | 4.733s  | 36.086                    |
| LCM w/ TGATE          | 7.623T   | 462.585M  | 4.543s  | 37.048                    |

The latency is tested on a 1080ti commercial card. 

The MACs and Params are calculated by [calflops](https://github.com/MrYxJ/calculate-flops.pytorch). 

The FID is calculated by [PytorchFID](https://github.com/mseitzer/pytorch-fid).

## TODO
- [x] Result image quality is inconsistent with origin. Now cache attn2 (cross_attention) only.
- [ ] Implement a native version and no longer rely on git patch
