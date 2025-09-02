这两个文件是在XAttention里的修改attention结构后的模型文件

在HELMET里是直接在mode_utils里import模型的

```python
if "llama" in model_name.lower():
    from xattn.src.load_llama import load_model, FastPrefillConfig
    cfg = FastPrefillConfig(**cfg_params)
    # use XAT's load_model instead of standard loading (like XAT model_wrappers.py)
    logger.info(f"Loading {model_name} with XAT {kwargs.get('attn_metric')} attention (Llama)")
    self.model, _ = load_model(cfg, name_or_path=model_name)
elif "qwen2.5" in model_name.lower():
    from xattn.src.load_qwen2 import load_qwen2_model, FastPrefillConfig
    cfg = FastPrefillConfig(**cfg_params)
    # use XAT's load_qwen2_model for Qwen models
    logger.info(f"Loading {model_name} with XAT {kwargs.get('attn_metric')} attention (Qwen2.5)")
    self.model, _ = load_qwen2_model(cfg, name_or_path=model_name)
```
目前这两个文件的接口不是对应最新版本的transformer库的（出于与其他库的适配考虑）

如果要支持qwen3 moe的话需要upgrade transformer库到比较新的版本，然后改一下对照最新版的接口把函数返回值数量从三个改成两个应该就好了（去掉past_key_value）