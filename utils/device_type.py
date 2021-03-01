from enum import unique, Enum
import torch


@unique
class DeviceType(Enum):

    cpu = 'cpu'
    cuda = 'cuda'

    def get(self) -> torch.device:
        return torch.device(self.value)

def is_gpu_available() -> bool:
    return torch.cuda.is_available()

def clear_gpu_cache():
    torch.cuda.empty_cache()