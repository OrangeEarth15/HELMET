#!/usr/bin/env python3
"""
调试脚本：验证xattn和xattn_v6的实际执行逻辑
"""

def test_xattn_logic():
    print("=== 测试 XAttention 逻辑判断 ===")
    
    # 测试不同的 use_simple_attention 值
    test_values = [False, 0, 6, "v6", True, 1]
    
    for use_simple_attention in test_values:
        print(f"\n测试值: {use_simple_attention} (类型: {type(use_simple_attention)})")
        
        # 这是 Xattention.py 中的判断逻辑
        should_use_simple = use_simple_attention not in [False, "False", "false", None, "", 0]
        print(f"  should_use_simple: {should_use_simple}")
        
        if should_use_simple:
            if use_simple_attention in ["v6", "V6", 6]:
                print(f"  → 会调用 compute_attention_rscores_v6")
            elif use_simple_attention in ["v1", "V1", True, "True", "true", 1]:
                print(f"  → 会调用 compute_attention_rscores_v1") 
            else:
                print(f"  → 会调用 compute_attention_rscores_v1 (默认)")
        else:
            print(f"  → 会使用原始复杂注意力估算")

def test_getattr_logic():
    print("\n=== 测试 getattr 逻辑 ===")
    
    class MockConfig:
        def __init__(self, use_simple_attention):
            self.use_simple_attention = use_simple_attention
    
    test_configs = [
        ("xattn配置", MockConfig(0)),
        ("xattn_v6配置", MockConfig(6)),
        ("缺失参数", MockConfig(None))  # 模拟没有这个属性
    ]
    
    for name, config in test_configs:
        print(f"\n{name}:")
        if hasattr(config, 'use_simple_attention'):
            value = getattr(config, 'use_simple_attention', 0)
            print(f"  getattr结果: {value}")
        else:
            value = getattr(config, 'use_simple_attention', 0) 
            print(f"  getattr结果 (使用默认值): {value}")
        
        # 测试实际逻辑
        should_use_simple = value not in [False, "False", "false", None, "", 0]
        print(f"  should_use_simple: {should_use_simple}")
        
        if should_use_simple and value == 6:
            print(f"  → 应该调用 v6 版本 ✅")
        elif not should_use_simple:
            print(f"  → 应该使用原始版本 ✅")
        else:
            print(f"  → 其他情况")

if __name__ == "__main__":
    test_xattn_logic()
    test_getattr_logic()
