from collections import defaultdict


# Generic bank mapping (index = element index)
def bank_idx_elem(
    index: int, elem_size: int = 4, num_banks: int = 32, bank_width: int = 4
):
    addr = index * elem_size  # convert element index -> byte address
    return (addr // bank_width) % num_banks


strides = [272, 2]
offsets = [[0, 8], [0, 1]]
distribution = [8, 4]  # [num_g, num_threads]
elems_per_thread = 2  # not actually used in this pattern


def compute_index(gidx, offset_y, tidx, offset_x, strides):
    return gidx * strides[0] + offset_y + tidx * strides[1] + offset_x


def print_expr_string(distribution, elems_per_thread, strides, offsets):
    num_axis = len(distribution)
    assert num_axis == len(strides)
    assert num_axis == len(offsets)
    for gidx in range(distribution[0]):
        for offset_y in offsets[0]:
            for lidx in range(distribution[1]):
                for offset_x in offsets[1]:
                    idx = compute_index(gidx, offset_y, lidx, offset_x, strides)
                    addr = idx * 4
                    bank = bank_idx_elem(idx)
                    tidx = gidx * distribution[1] + lidx
                    print(
                        f"{'tidx'+str(tidx):<8} "
                        f"{gidx:>2} * {strides[0]:>3} + {offset_y:>2} + "
                        f"{lidx:>2} * {strides[1]:>3} + {offset_x:>2} = {idx:>4} : "
                        f"address = {addr:>6} bank = {bank:>2}"
                    )


print_expr_string(distribution, elems_per_thread, strides, offsets)
