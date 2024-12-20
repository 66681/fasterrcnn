import yaml
import os


def read_yaml(yaml_path):
    """
    读取YAML配置文件
    
    Args:
        yaml_path: YAML文件路径
        
    Returns:
        dict: 配置字典，如果读取失败返回None
    """
    try:
        if not isinstance(yaml_path, (str, bytes, os.PathLike)):
            raise TypeError("yaml_path必须是字符串或路径对象")
            
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"找不到配置文件: {yaml_path}")
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict):
            raise ValueError("YAML文件必须包含一个字典配置")
            
        return config
    except Exception as e:
        print(f"读取YAML文件出错: {str(e)}")
        return None


def write_yaml(path='application.yaml', data=None):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data=data, stream=f, allow_unicode=True)
    except Exception as e:
        print(e)