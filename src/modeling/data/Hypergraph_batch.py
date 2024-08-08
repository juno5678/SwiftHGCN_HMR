import inspect
from collections.abc import Sequence
from typing import Any, List, Optional, Type, Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData, Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.separate import separate
from torch_geometric.data.batch import Batch, DynamicInheritance, DynamicInheritanceGetter
from src.modeling.data.Hypergraph_collate import hg_collate

class HG_Batch(metaclass=DynamicInheritance):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.
    Inherits from :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData`.
    In addition, single graphs can be identified via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.

    :pyg:`PyG` allows modification to the underlying batching procedure by
    overwriting the :meth:`~Data.__inc__` and :meth:`~Data.__cat_dim__`
    functionalities.
    The :meth:`~Data.__inc__` method defines the incremental count between two
    consecutive graph attributes.
    By default, :pyg:`PyG` increments attributes by the number of nodes
    whenever their attribute names contain the substring :obj:`index`
    (for historical reasons), which comes in handy for attributes such as
    :obj:`edge_index` or :obj:`node_index`.
    However, note that this may lead to unexpected behavior for attributes
    whose names contain the substring :obj:`index` but should not be
    incremented.
    To make sure, it is best practice to always double-check the output of
    batching.
    Furthermore, :meth:`~Data.__cat_dim__` defines in which dimension graph
    tensors of the same attribute should be concatenated together.
    """
    @classmethod
    def from_data_list(
        cls,
        data_list: List[BaseData],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> Self:
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        """
        batch, slice_dict, inc_dict = hg_collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)  # type: ignore
        batch._slice_dict = slice_dict  # type: ignore
        batch._inc_dict = inc_dict  # type: ignore

        return batch

    def get_example(self, idx: int) -> BaseData:
        r"""Gets the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object at index :obj:`idx`.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object.
        """
        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'Batch' because "
                 "'Batch' was not created via 'Batch.from_data_list()'"))

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=getattr(self, '_slice_dict'),
            inc_dict=getattr(self, '_inc_dict'),
            decrement=True,
        )

        return data

    def index_select(self, idx: IndexType) -> List[BaseData]:
        r"""Creates a subset of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from specified
        indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects.
        """
        index: Sequence[int]
        if isinstance(idx, slice):
            index = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            index = idx.flatten().tolist()

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            index = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            index = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            index = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            index = idx

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self.get_example(i) for i in index]

    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            return self.get_example(idx)  # type: ignore
        elif isinstance(idx, str) or (isinstance(idx, tuple)
                                      and isinstance(idx[0], str)):
            # Accessing attributes or node/edge types:
            return super().__getitem__(idx)  # type: ignore
        else:
            return self.index_select(idx)

    def to_data_list(self) -> List[BaseData]:
        r"""Reconstructs the list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from the
        :class:`~torch_geometric.data.Batch` object.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects.
        """
        return [self.get_example(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if hasattr(self, '_num_graphs'):
            return self._num_graphs
        elif hasattr(self, 'ptr'):
            return self.ptr.numel() - 1
        elif hasattr(self, 'batch'):
            return int(self.batch.max()) + 1
        else:
            raise ValueError("Can not infer the number of graphs")

    @property
    def batch_size(self) -> int:
        r"""Alias for :obj:`num_graphs`."""
        return self.num_graphs

    def __len__(self) -> int:
        return self.num_graphs

    def __reduce__(self) -> Any:
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state

if __name__ == "__main__":
    model_dim = (8, 16)
    batch_size = 4
    num_nodes = 431
    num_edges = 14
    in_channels, out_channels = (model_dim[0], model_dim[1])
    hyperedge_index = torch.load("../../../tensor_hg.pt").to(torch.device("cuda"))
    x = torch.randn((batch_size, num_nodes, in_channels))
    hyperedge_attr = torch.randn((batch_size, num_edges, out_channels))
    data = [Data(x=x[i], edge_index=hyperedge_index, edge_attr=hyperedge_attr[i]) for i in range(x.shape[0])]
    batch_data = HG_Batch.from_data_list(data)