# ComfyUI_TGate

<p align="center">
<a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a>
</p>

> 如果我的仓库对你起到了帮助，可以考虑给个 star。

[T-GATE](https://github.com/HaozheLiu-ST/T-GATE) 的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 插件实现.

T-GATE 可以在**保持原始构图**，**略微**降低生成图像的质量的情况下给你的 diffusion 模型带来 **10%-50% 的性能提升**。

> 当前实现使用了一些 monkey patch，如果出现了任何报错，请确保更新到了最新版本，还有问题的话随时提 issue。

一些其他可能会帮助到你的仓库：

- [ComfyUI-TCD](https://github.com/JettHu/ComfyUI-TCD)：TCD 采样器 ComfyUI 实现，在和 LCM 相同性能的情况下生成比 LCM 细节更丰富的结果图
- [ComfyUI_TGate](https://github.com/JettHu/ComfyUI_TGate)：T-GATE ComfyUI 实现，在接受一部分细节损失的情况下 10-50% 的性能提升。
- [ComfyUI-ELLA](https://github.com/TencentQQGYLab/ComfyUI-ELLA)：ELLA 官方 ComfyUI 实现，基于大语言模型，让你的 SD 模型更准确地理解你的描述。


## :star2: 更新日志
- **[2024.4.30]** :wrench: 修复了 animatediff 中 cond-only 情况下的一个错误。感谢 [pamparamm](https://github.com/pamparamm).
- **[2024.4.29]** :wrench: `TL,DR`: 相比上一版本提升了一些性能，并且 T-GATE 只在它该起作用的地方起作用。
  - 修复了一个即使 `TGateApply` 节点被 bypass 或者删除掉，`TGateApply` 会仍然影响其他使用模型的地方（下游）的 bug。
  - 修复了一个导致 cross attntion 结果没有被正确 cache 的错误，略微提升了一些性能。
- **[2024.4.26]** :tada: Native 版本发布(不在需要 git patch 了!)。
- **[2024.4.18]** 初始化仓库

## :books: 示例 workflow

[examples](./examples/) 文件夹下有使用节点的示例（workflow 截图里已经注入了完整的 workflow，可以直接 Load 图片或拖到 ComfyUI 空白的地方来导入 workflow）。

![example](./examples/tgate_workflow_example.png)

| 原始结果图 | 使用 T-GATE 结果图 |
| :---: | :---: |
| ![origin_result](./assets/origin_result.png) | ![tgate_result](./assets/tgate_result.png) |

T-GATE 结果图由示例 [example](./examples/tgate_workflow_example.png) 生成。


### 和另一个“类似”仓库 AutomaticCFG 的比较

[AutomaticCFG](https://github.com/Extraltodeus/ComfyUI-AutomaticCFG) 偶然间发现了另一个仓库，里面有些地方和 T-GATE 的思路有点像。

> 环境: Tesla T4-8G

| | Origin | T-GATE 0.5 | AutomaticCFG | T-GATE 0.35 |AutomaticCFG fatest |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 结果图 | ![origin_result](./assets/origin_result.png) | ![tgate_result](./assets/tgate_result.png) | ![auto_cfg_boost](./assets/auto_cfg_boost.png) | ![tgate_0_35](./assets/tgate_0_35.png) | ![auto_cfg_fatest](./assets/auto_cfg_fatest.png) |
| 速度 | 4.59it/s | **5.68it/s** | 5.62it/s| **6.13it/s** | **6.13it/s** |

在保持结果构图一致的情况下 T-GATE 有最好的性能。 不过，如果你不太关心结果的构图和不使用这两个插件的原结果图的差异，[AutomaticCFG fatest](https://github.com/Extraltodeus/ComfyUI-AutomaticCFG) 也可以带来差不多的性能提升。

## :green_book: 安装
```bash
git clone https://github.com/JettHu/ComfyUI_TGate
# Or ComfyUI-Manager, that's all!
```

## :orange_book: 主要特点

- 无需训练。
- 与基于 CNN-based U-Net 的 SD 类模型，与 Transformer 的 DiT 类模型，以及 LCM/TCD 良好兼容。
- 和 DeepCache 可以共用。
- 对于不同的模型和强度可以达到 10%-50% 的性能提升


## :book: 节点说明

### TGateApply

#### 输入
- **model**, `Load Checkpoint` 或者是其他节点加载的 SD 模型。

#### 配置参数
- **start_at**, 这个是 T-GATE 起作用的百分比，表示从哪一步开始使用 T-GATE cache，越早开始性能提升越多，不过细节会丢失更多，需要自己取舍。
- **only_cross_attention**, **[推荐]** 默认是打开的，作用是控制是否只缓存`cross attntion`, 参考这个 [issues](https://github.com/HaozheLiu-ST/T-GATE/issues/8#issuecomment-2061379798)，如果关掉会导致结果细节丢失更多，强烈不建议关掉。


#### 可选配置
- **self_attn_start_at**, 只在 `only_cross_attention` 关掉时发生作用, 表示从哪一步开始对 self attnention 使用 T-GATE cache。

## :rocket: 一些来自 ([T-GATE](https://github.com/HaozheLiu-ST/T-GATE)) 的性能数据
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

## :memo: TODO
- [x] Result image quality is inconsistent with origin. Now cache attn2 (cross_attention) only.
- [x] Implement a native version and no longer rely on git patch

## :mag: 常见问题

- 苹果 M 系列芯片可能会遇到的问题 torch 和 macos 版本兼容问题。参考 [issue comment](https://github.com/JettHu/ComfyUI_TGate/issues/4#issuecomment-2077823182)。

- 2024.4.29 已修复，无法通过删除节点/断开链接/bypass 关闭 T-GATE影响，修复前后的效果见下图。

| 2024.4.26-29 | 2024.4.29 更新 |
| :---: | :---: |
| ![before_fixed](./assets/before_fixed.png) | ![after_fixed](./assets/after_fixed.png) |
