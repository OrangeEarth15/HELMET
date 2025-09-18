#!/usr/bin/env python3
"""
测试 TriangleMix "tri" 策略是否正确集成到 HELMET 中
"""

import sys
import os
sys.path.insert(0, '/home/scratch.sarawang_ent/project/XAT')

def test_tri_import():
    """测试 tri 策略的导入"""
    print("🔍 Testing TriangleMix 'tri' strategy import...")
    
    try:
        from xattn.src.trianglemix import tri_test_forward
        print("✅ tri_test_forward imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import tri_test_forward: {e}")
        return False

def test_tri_config():
    """测试 tri 策略的配置"""
    print("\n🔧 Testing TriangleMix configuration...")
    
    try:
        from xattn.src.load_llama import FastPrefillConfig
        
        # 创建 tri 策略配置
        config = FastPrefillConfig(
            metric="tri",
            print_detail=True
        )
        
        print(f"✅ FastPrefillConfig created with metric: {config.metric}")
        print(f"   - print_detail: {config.print_detail}")
        print(f"   - stride: {config.stride}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create FastPrefillConfig: {e}")
        return False

def test_helmet_integration():
    """测试 HELMET 中的集成"""
    print("\n🏗️ Testing HELMET integration...")
    
    try:
        from arguments import parse_arguments
        
        # 模拟命令行参数
        test_args = [
            "--model_name_or_path", "dummy_model",
            "--attn_metric", "tri",
            "--output_dir", "test_output"
        ]
        
        args = parse_arguments(test_args)
        
        print(f"✅ Arguments parsed successfully")
        print(f"   - attn_metric: {args.attn_metric}")
        print(f"   - model_name_or_path: {args.model_name_or_path}")
        
        # 检查 "tri" 是否在允许的选项中
        if args.attn_metric == "tri":
            print("✅ 'tri' strategy accepted by argument parser")
            return True
        else:
            print("❌ 'tri' strategy not properly set")
            return False
            
    except Exception as e:
        print(f"❌ HELMET integration test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🎯 TriangleMix 'tri' Strategy Integration Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_tri_import),
        ("Config Test", test_tri_config), 
        ("HELMET Integration Test", test_helmet_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎉 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🚀 TriangleMix 'tri' strategy is ready to use!")
        print("\n📝 Usage:")
        print("   cd /home/scratch.sarawang_ent/project/HELMET")
        print("   ./llama_scripts/llama_tri.sh")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
