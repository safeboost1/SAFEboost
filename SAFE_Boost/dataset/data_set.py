import json
import os
from numbers import Number
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from HE_codes.utils import left_rotate_bin, right_rotate_bin
from heaan_stat.core import config
from heaan_stat.core import Block
from heaan_stat.core import Context
from heaan_stat.core import HEObject
from heaan_stat.core import HEObjectList
from pydantic import BaseModel
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from numbers import Number
import numpy as np
from heaan_stat.core import config
from heaan_stat.core import Block
from heaan_stat.core import Context
class DataList:

    def __init__(
        self,
        context: Context,
        encrypted: bool = False,
        data_list: Optional[List[Union[Block, Iterable[Number]]]] = None,
        is_identical: bool = False,
        is_complex: bool = False,
    ):
        self.context = context
        self.encrypted = encrypted
        self.is_identical = is_identical
        self.is_complex = is_complex
        self.block_list: List[Block] = []
        if data_list is not None:
            for data in data_list:
                if isinstance(data, Block):
                    self.block_list.append(data)
                else:
                    block = Block(
                        context, 
                        encrypted=encrypted, 
                        data=data, 
                        is_complex=is_complex,
                        is_identical=is_identical
                    )
                    self.block_list.append(block)
    def __len__(self) -> int:
        return len(self.block_list)
    def __setitem__(self, index, value):
        self.block_list[index] = value
    def __getitem__(self, idx: int) -> Block:
        return self.block_list[idx]
    def append(self, block: Block):
        self.block_list.append(block)
    def set_block_list(self, block_list: List[Block]):
        self.block_list = list(block_list)
    def deepcopy(self) -> "DataList":
        res = DataList(self.context, encrypted=self.encrypted, is_complex=self.is_complex)
        res.set_block_list([b.deepcopy() for b in self.block_list])
        return res
    def __add__(self, other: Union["DataList", Block, Number, np.ndarray]) -> "DataList":
        res = DataList(self.context, encrypted=self.encrypted, is_complex=self.is_complex)
        if isinstance(other, DataList):
            new_blocks = [s + o for s, o in zip(self.block_list, other.block_list)]
        else:
            new_blocks = [b + other for b in self.block_list]
        res.set_block_list(new_blocks)
        return res
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other: Union["DataList", Block, Number, np.ndarray]) -> "DataList":
        res = DataList(self.context, encrypted=self.encrypted, is_complex=self.is_complex)
        if isinstance(other, DataList):
            new_blocks = [s - o for s, o in zip(self.block_list, other.block_list)]
        else:
            new_blocks = [b - other for b in self.block_list]
        res.set_block_list(new_blocks)
        return res
    def __rsub__(self, other):
        res = DataList(self.context, encrypted=self.encrypted, is_complex=self.is_complex)
        res.set_block_list([other - b for b in self.block_list])
        return res
    def __mul__(self, other: Union["DataList", Block, Number, np.ndarray]) -> "DataList":
        res = DataList(self.context, encrypted=self.encrypted, is_complex=self.is_complex)
        if isinstance(other, DataList):
            new_blocks = [s * o for s, o in zip(self.block_list, other.block_list)]
        else:
            new_blocks = [b * other for b in self.block_list]
        res.set_block_list(new_blocks)
        return res
    def __rmul__(self, other): return self.__mul__(other)
    def inverse(self) -> "DataList":
        res = self.deepcopy()
        res.set_block_list([b.inverse() for b in res.block_list])
        return res
    def sum_block(self) -> Block:
        if not self.block_list: return Block.zeros(self.context, self.encrypted)
        s = self.block_list[0].deepcopy()
        for i in range(1, len(self.block_list)): s += self.block_list[i]
        return s
    def save(self, path: Path, target_level: Optional[int] = None):
        path = Path(path)
        os.makedirs(path, mode=0o775, exist_ok=True)
        for idx, block in enumerate(self.block_list):
            block.save(path / f"block_{idx}.bin", target_level=target_level)
    @staticmethod
    def from_path(context: Context, path: Union[Path, str], encrypted: bool) -> "DataList":
        num_blocks = 0
        for s in os.listdir(path):
            if "block" in s:
                num_blocks += 1
        hedtdata = DataList(context)
        return hedtdata.load(path, num_blocks, encrypted)
    def load(
        self,
        path: Union[Path, str],
        num_blocks: int = 0,
        encrypted: bool = False,
    ) -> "DataList":
        block_list = []
        for idx in range(num_blocks):
            block = Block.from_path(self.context, config.get_path(path, idx), encrypted=encrypted)
            block_list.append(block)
        self.block_list=block_list
        return self
    def twice_of_real_and_imag_parts(self) -> Tuple["DataList", "DataList"]:

        real_blocks = []
        imag_blocks = []
        for blk in self.block_list:
            real2x_blk, imag2x_blk = blk.twice_of_real_and_imag_parts()
            real_blocks.append(real2x_blk)
            imag_blocks.append(imag2x_blk)
        real2x_dl = DataList(self.context, encrypted=self.encrypted)
        imag2x_dl = DataList(self.context, encrypted=self.encrypted)
        real2x_dl.set_block_list(real_blocks)
        imag2x_dl.set_block_list(imag_blocks)
        return real2x_dl, imag2x_dl
    def decrypt(self):
        for i in range(len(self.block_list)):
            self.block_list[i].decrypt()
        return self
    def encrypt(self):
        for i in range(len(self.block_list)):
            self.block_list[i].encrypt()
        return self
    def level_down(self,level):
        for i in range(len(self.block_list)):
            self.block_list[i].level_down(level)
    def __lshift__(self, idx: int) -> "DataList":
        res = self.deepcopy()
        res.block_list=([res.block_list[r] << idx for r in range(len(res))])
        return res
    def __rshift__(self, idx: int) -> "DataList":
        res = self.deepcopy()
        res.block_list=([res.block_list[r] >> idx for r in range(len(res))])
        return res    
    def right_rotate_bin(self, interval: int, gs: int, encrypted: bool = True):
        res = DataList(self.context, encrypted=encrypted)
        for i in range(len(self.block_list)):
            tmp = right_rotate_bin(self.context, self.block_list[i], interval, gs)
            res.append(tmp)
        return res
    def left_rotate_bin(self, interval: int, gs: int, encrypted: bool = True):
        res = DataList(self.context, encrypted=encrypted)
        for i in range(len(self.block_list)):
            tmp = left_rotate_bin(self.context, self.block_list[i], interval, gs)
            res.append(tmp)
        return res
    @property
    def level(self):
        return self.block_list[0].level
    def compare(self, other: Union[Block, "DataList"], numiter_g: int = 8, numiter_f: int = 3):
        res = DataList(self.context, self.encrypted)
        if type(other) is Block:
            for i in range(len(self.block_list)):
                tmp = self.block_list[i].compare(other, numiter_g=numiter_g, numiter_f=numiter_f)
                res.append(tmp)
        elif len(other) == 1:
            for i in range(len(self.block_list)):
                tmp = self.block_list[i].compare(other[0], numiter_g=numiter_g, numiter_f=numiter_f)
                res.append(tmp)
        else:
            for i in range(len(self.block_list)):
                tmp = self.block_list[i].compare(other[i], numiter_g=numiter_g, numiter_f=numiter_f)
                res.append(tmp)
        return res
    def need_bootstrap(self,arg):
        res=False
        if self.block_list[0].level < arg+3:
            res=True
        return res
    def bootstrap(self):
        for i in self.block_list:
            i.bootstrap()
class DataMatrix:

    def __init__(
        self,
        context: Context,
        encrypted: bool = False,
        row_list: Optional[List[Union[DataList, List[Iterable[Number]]]]] = None,
        is_identical: bool = False,
        is_complex: bool = False,
    ):
        self.context = context
        self.encrypted = encrypted
        self.is_identical = is_identical
        self.is_complex = is_complex
        self.rows: List[DataList] = []
        if row_list is not None:
            for row_data in row_list:
                if isinstance(row_data, DataList):
                    self.rows.append(row_data)
                else:
                    dl = DataList(context, encrypted, row_data, is_identical, is_complex)
                    self.rows.append(dl)
    def load(self, path: Union[str, Path], encrypted: bool) -> None:
        if isinstance(path, str):
            path = Path(path)
        metadata_path = path / config._metadata_file_name
        with open(metadata_path, "r") as m_file:
            m_info = json.load(m_file)
        num_list = m_info["num_list"]

        self.rows = [DataList.from_path(self.context, path / str(idx), encrypted) for idx in range(num_list)]
        return self
    @staticmethod
    def from_path(context, path: Union[str, Path], encrypted: bool) -> "DataMatrix":
        return DataMatrix.load(context=context,path=path, encrypted=encrypted)
    def bootstrap(self):
        for i in self.rows:
            i.bootstrap()
    def need_bootstrap(self,arg):
        res=False
        if self.rows[0][0].level < arg+3:
            res=True
        return res            
    def __len__(self) -> int:
        return len(self.rows)
    def __setitem__(self, index, value):
        self.rows[index] = value
    def __getitem__(self, idx: int) -> DataList:
        return self.rows[idx]
    def append(self, row: DataList):
        self.rows.append(row)
    def set_rows(self, rows: List[DataList]):
        self.rows = list(rows)
    def deepcopy(self) -> "DataMatrix":
        res = DataMatrix(self.context, self.encrypted, is_complex=self.is_complex)
        res.set_rows([r.deepcopy() for r in self.rows])
        return res
    def decrypt(self):
        for i in range(len(self.rows)):
            self.rows[i].decrypt()
        return self
    def encrypt(self):
        for i in range(len(self.rows)):
            self.rows[i].encrypt()
        return self
    def level_down(self,level):
        for i in range(len(self.rows)):
            self.rows[i].level_down(level)
    def __add__(self, other: Union["DataMatrix", DataList, Block, Number, np.ndarray]) -> "DataMatrix":
        res = DataMatrix(self.context, self.encrypted, is_complex=self.is_complex)
        if isinstance(other, DataMatrix):
            new_rows = [r1 + r2 for r1, r2 in zip(self.rows, other.rows)]
        elif isinstance(other, DataList):
            new_rows = [r + other for r in self.rows]
        else:
            new_rows = [r + other for r in self.rows]
        res.set_rows(new_rows)
        return res
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other: Union["DataMatrix", DataList, Block, Number, np.ndarray]) -> "DataMatrix":
        res = DataMatrix(self.context, self.encrypted, is_complex=self.is_complex)
        if isinstance(other, DataMatrix):
            new_rows = [r1 - r2 for r1, r2 in zip(self.rows, other.rows)]
        elif isinstance(other, DataList):
            new_rows = [r - other for r in self.rows]
        else:
            new_rows = [r - other for r in self.rows]
        res.set_rows(new_rows)
        return res
    def __rsub__(self, other):
        res = DataMatrix(self.context, self.encrypted, is_complex=self.is_complex)
        new_rows = [other - r for r in self.rows]
        res.set_rows(new_rows)
        return res
    def __mul__(self, other: Union["DataMatrix", DataList, Block, Number, np.ndarray]) -> "DataMatrix":
        res = DataMatrix(self.context, self.encrypted, is_complex=self.is_complex)
        if isinstance(other, DataMatrix):
            new_rows = [r1 * r2 for r1, r2 in zip(self.rows, other.rows)]
        elif isinstance(other, DataList):
            new_rows = [r * other for r in self.rows]
        else:
            new_rows = [r * other for r in self.rows]
        res.set_rows(new_rows)
        return res
    def __rmul__(self, other): return self.__mul__(other)
    def sum_rows(self) -> DataList:
        res = DataList(self.context, self.encrypted)
        res.set_block_list([r.sum_block() for r in self.rows])
        return res
    def sum_matrix(self) -> Block:
        row_sums = [r.sum_block() for r in self.rows]
        if not row_sums: return Block.zeros(self.context, self.encrypted)
        total = row_sums[0]
        for i in range(1, len(row_sums)): total += row_sums[i]
        return total
    def inverse(self) -> "DataMatrix":
        res = self.deepcopy()
        res.set_rows([r.inverse() for r in res.rows])
        return res
    def twice_of_real_and_imag_parts(self) -> Tuple["DataMatrix", "DataMatrix"]:
        re_lists, im_lists = [], []
        for r in self.rows:
            re_dl, im_dl = r.twice_of_real_and_imag_parts()
            re_lists.append(re_dl)
            im_lists.append(im_dl)
        re_mat = DataMatrix(self.context, self.encrypted)
        im_mat = DataMatrix(self.context, self.encrypted)
        re_mat.set_rows(re_lists)
        im_mat.set_rows(im_lists)
        return re_mat, im_mat
    def compare(self, other: Union["DataMatrix", DataList, Block], numiter_g: int = 8, numiter_f: int = 3) -> "DataMatrix":
        res = DataMatrix(self.context, self.encrypted)
        new_rows = []
        if isinstance(other, DataMatrix):
            new_rows = [self.rows[i].compare(other.rows[i], numiter_g, numiter_f) for i in range(len(self))]
        else:
            new_rows = [r.compare(other, numiter_g, numiter_f) for r in self.rows]
        res.set_rows(new_rows)
        return res
    def save(self, path: Union[str, Path], target_level: Optional[int] = None):
        path = Path(path)
        os.makedirs(path, mode=0o775, exist_ok=True)
        num_rows = len(self.rows)
        metadata = {
            "num_list": num_rows
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        for i, row in enumerate(self.rows):
            row.save(path / f"{i}", target_level=target_level)
    @staticmethod
    def from_path(context: Context, path: Union[str, Path], encrypted: bool) -> "DataMatrix":
        path = Path(path)
        row_dirs = sorted([d for d in os.listdir(path) if d.startswith("row_")])
        matrix = DataMatrix(context, encrypted)
        rows = []
        for rd in row_dirs:
            row_path = path / rd
            num_blocks = len([f for f in os.listdir(row_path) if "block" in f])
            dl = DataList(context, encrypted).load(row_path, num_blocks, encrypted)
            rows.append(dl)
        matrix.set_rows(rows)
        return matrix
    def __lshift__(self, idx: int) -> "DataMatrix":
        res = self.deepcopy()
        res.set_rows([r << idx for r in res.rows])
        return res
    def __rshift__(self, idx: int) -> "DataMatrix":
        res = self.deepcopy()
        res.set_rows([r >> idx for r in res.rows])
        return res
    def right_rotate_bin(self, interval: int, gs: int, encrypted: bool = True):
        res = DataMatrix(self.context, encrypted=encrypted)
        for i in range(len(self.rows)):
            tmp = self.rows[i].right_rotate_bin(interval, gs, encrypted)
            res.append(tmp)
        return res
    def left_rotate_bin(self, interval: int, gs: int, encrypted: bool = True):
        res = DataMatrix(self.context, encrypted=encrypted)
        for i in range(len(self.rows)):
            tmp = self.rows[i].left_rotate_bin(interval, gs, encrypted)
            res.append(tmp)
        return res
    @property
    def level(self):
        return self.rows[0].level