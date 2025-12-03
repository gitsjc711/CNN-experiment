try:
    import cupy as cp

    HAVE_GPU = True
    print("CuPy 可用，默认启用 GPU 加速模式。")
except ImportError:
    HAVE_GPU = False
    print("未找到 CuPy，将使用 NumPy 在 CPU 上运行。")
    import numpy as cp  # 回退到NumPy


class DeviceManager:
    """设备管理类，统一管理 CPU/GPU 后端和数组创建。"""
    _instance = None
    _device = 'gpu' if HAVE_GPU else 'cpu'  # 默认尝试使用GPU

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_device(cls, device: str):
        """设置计算设备 ('cpu' 或 'gpu')"""
        assert device in ['cpu', 'gpu'], "设备必须是 'cpu' 或 'gpu'"
        if device == 'gpu' and not HAVE_GPU:
            print("警告：请求使用 GPU，但 CuPy 不可用。将回退到 CPU。")
            device = 'cpu'
        cls._device = device
        print(f"计算设备已设置为: {device.upper()}")

    @classmethod
    def get_xp(cls):
        """获取当前的计算后端 (cupy 或 numpy 模块)"""
        if cls._device == 'gpu' and HAVE_GPU:
            return cp  # 返回 cupy 模块
        else:
            import numpy as np
            return np  # 返回 numpy 模块

    @classmethod
    def to_device(cls, array):
        xp = cls.get_xp()
        if xp.__name__ == 'cupy':
            # 如果已经是CuPy数组，直接返回
            if isinstance(array, cp.ndarray):
                return array
            else:
                return cp.asarray(array)
        else:
            # 如果是CuPy数组，转换为NumPy数组（CPU）
            if isinstance(array, cp.ndarray):
                return array.get()
            # 如果已经是NumPy数组，直接返回
            else:
                return array


device_manager = DeviceManager()
