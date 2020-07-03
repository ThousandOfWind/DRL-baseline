from .episode_memory import Memmory as Ep_Mem
from .onestep_memory import Memmory as OS_Mem

Memory = {
    'ep': Ep_Mem,
    'os': OS_Mem,
}