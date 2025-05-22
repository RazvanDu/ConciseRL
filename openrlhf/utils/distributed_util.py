from datetime import timedelta
from typing import Any, Optional, Union

import torch
import torch.distributed
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    print("RRR0")
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."
    print("RRR1")

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"
    print("RRR2")

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout
    print("RRR3")

    # backward compatible API
    if store is None:
        print("UUU1")
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        print("UUU2")
        print(f"Before rendezvous: rank={rank}, world_size={world_size}, init_method={init_method}, timeout={timeout}")
        store, rank, world_size = next(rendezvous_iterator)
        print("UUU3")
        store.set_timeout(timeout)
        print("UUU4")

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)
        print("UUU5")
    print("RRR4")

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )
    print("RRR5")

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    print("RRR6")

    return pg
