import ctypes
import os
from typing import List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SO_PATH = os.path.join(_THIS_DIR, "submitv2.so")

_lib = ctypes.CDLL(_DEFAULT_SO_PATH)

_lib.move_decision_c.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  
    ctypes.c_int,                   
    ctypes.c_int,                   
    ctypes.c_int,                   
    ctypes.c_int,                   
    ctypes.POINTER(ctypes.c_int32),
]
_lib.move_decision_c.restype = None

class Player:
    def __init__(self):
        self.gold = 0

    def MoveDecision(self, grid: List[List[int]], gold: int, gold_2: int) -> List[int]:
        rows, cols = 17, 17
        if rows != 17 or cols != 17:
            return [4, 4, 4]

        flat_list = [int(grid[i][j]) for i in range(rows) for j in range(cols)]
        grid_arr = (ctypes.c_int32 * (rows * cols))(*flat_list)
        out_arr = (ctypes.c_int32 * 3)()

        _lib.move_decision_c(
            ctypes.cast(grid_arr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(rows),
            ctypes.c_int(cols),
            ctypes.c_int(int(gold)),
            ctypes.c_int(int(gold_2)),
            ctypes.cast(out_arr, ctypes.POINTER(ctypes.c_int32)),
        )

        # print([int(out_arr[0]), int(out_arr[1]), int(out_arr[2])])
        return [int(out_arr[0]), int(out_arr[1]), int(out_arr[2])]