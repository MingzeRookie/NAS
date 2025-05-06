"""模型量化支持模块，提供用于4-bit和8-bit量化的工具函数"""

import torch
from transformers import BitsAndBytesConfig


def get_quantization_config(quant_type="4bit", compute_dtype=torch.float16, use_double_quant=True):
    """获取量化配置
    
    Args:
        quant_type: 量化类型，"4bit"或"8bit"
        compute_dtype: 计算数据类型
        use_double_quant: 是否使用双重量化
        
    Returns:
        BitsAndBytesConfig对象
    """
    if quant_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # 为正态分布的权重优化
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    elif quant_type == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    else:
        raise ValueError(f"不支持的量化类型: {quant_type}")


def estimate_memory_usage(model, quant_type="4bit"):
    """估计模型内存使用
    
    Args:
        model: PyTorch模型
        quant_type: 量化类型，"4bit"、"8bit"或"fp16"
        
    Returns:
        预估的内存使用量（MB）
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    if quant_type == "4bit":
        bits_per_param = 4
    elif quant_type == "8bit":
        bits_per_param = 8
    elif quant_type == "fp16":
        bits_per_param = 16
    else:
        bits_per_param = 32  # 默认为fp32
    
    # 计算内存使用量（MB）
    memory_usage = total_params * bits_per_param / 8 / 1024 / 1024
    
    return memory_usage


def print_model_memory_usage(model, quant_types=["fp32", "fp16", "8bit", "4bit"]):
    """打印不同量化配置下的模型内存使用
    
    Args:
        model: PyTorch模型
        quant_types: 要打印的量化类型列表
    """
    for quant_type in quant_types:
        memory_usage = estimate_memory_usage(model, quant_type)
        print(f"{quant_type}: {memory_usage:.2f} MB")


def get_device_info():
    """获取设备信息
    
    Returns:
        设备信息字典
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if info["cuda_available"]:
        for i in range(info["device_count"]):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / 1024**3,  # GB
                "memory_free": torch.cuda.memory_reserved(i) / 1024**3  # GB
            }
            info["devices"].append(device_info)
    
    return info


def check_hardware_compatibility(quant_type="4bit"):
    """检查硬件兼容性
    
    Args:
        quant_type: 量化类型，"4bit"或"8bit"
        
    Returns:
        是否兼容
    """
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，无法使用量化")
        return False
    
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    cuda_compute_capability = cuda_major * 10 + cuda_minor
    
    # 对于4-bit量化，建议使用Ampere或更高架构（计算能力>=8.0）
    if quant_type == "4bit" and cuda_compute_capability < 80:
        print(f"警告: 当前GPU计算能力为{cuda_major}.{cuda_minor}，4-bit量化推荐8.0或更高")
        print("可能会影响性能，但仍然可以工作")
        return True
    
    return True