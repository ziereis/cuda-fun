from collections import defaultdict
from dataclasses import dataclass
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MemoryAccess:
    ld: int
    elems_per_thread: int
    elem_width: int

    def calc_idx(self, tidx):
        pass


@dataclass
class VectorizedGlobalAccess(MemoryAccess):
    def calc_idx(self, tidx):
        row = tidx // 8
        col = tidx % 8
        for i in range(self.elems_per_thread):
            yield row * self.ld + col * self.elems_per_thread + i


@dataclass
class SwizzledAccess(MemoryAccess):
    swizzle_bits: int = 2
    vector_shift: int = 2

    def calc_idx(self, tidx):
        row = tidx // 8
        col = tidx % 8

        swizzle_mask = (1 << self.swizzle_bits) - 1

        for i in range(self.elems_per_thread):
            modifier = (row & swizzle_mask) << self.vector_shift
            col_idx = col * self.elems_per_thread + i
            swizzeled_col = col_idx ^ modifier
            yield row * self.ld + swizzeled_col


@dataclass
class StoreCfragAccess(MemoryAccess):
    def __init__(self, ld=32):
        self.ld = ld
        self.elems_per_thread = 4
        self.elem_width = 4

    def calc_idx(self, tidx):
        local_row0 = tidx // 4
        row0 = local_row0
        row1 = row0 + 8
        col0 = (tidx % 4) * 2
        col1 = col0 + 1
        yield row0 * self.ld + col0
        yield row0 * self.ld + col1
        yield row1 * self.ld + col0
        yield row1 * self.ld + col1


@dataclass
class StoreCfragSwizzled(MemoryAccess):
    swizzle_bits: int = 2
    vector_shift: int = 2

    def calc_idx(self, tidx):
        swizzle_mask = (1 << self.swizzle_bits) - 1
        local_row0 = tidx // 4
        row0 = local_row0
        row1 = row0 + 8
        col0 = (tidx % 4) * 2
        col1 = col0 + 1
        modifier0 = (row0 & swizzle_mask) << self.vector_shift
        modifier1 = (row1 & swizzle_mask) << self.vector_shift
        swizzled_col0 = col0 ^ modifier0
        swizzled_col1 = col1 ^ modifier0
        swizzled_col2 = col0 ^ modifier1
        swizzled_col3 = col1 ^ modifier1
        yield row0 * self.ld + swizzled_col0
        yield row0 * self.ld + swizzled_col1
        yield row1 * self.ld + swizzled_col2
        yield row1 * self.ld + swizzled_col3


def bank_idx(addr: int, num_banks: int = 32, bank_width: int = 4):
    return (addr // bank_width) % num_banks


def print_access(access: MemoryAccess, num_threads=32):
    for tidx in range(num_threads):
        for idx in access.calc_idx(tidx):
            addr = idx * access.elem_width
            bank = bank_idx(addr)
            print(f"tidx{tidx} - idx: {idx} - addr: {addr} - bank: {bank}")


def _plot_on_axis(ax, mem_access, num_threads=32):
    num_banks = 32
    bank_width_bytes = 4

    # 1. Map Linear Index -> Thread ID
    idx_to_thread = {}
    max_idx = 0
    for tidx in range(num_threads):
        for idx in mem_access.calc_idx(tidx):
            idx_to_thread[idx] = tidx
            max_idx = max(max_idx, idx)

    # 2. Grid Dimensions
    ld = mem_access.ld
    grid_width = ld
    grid_height = max(1, (max_idx // ld) + 1)

    # 3. Color Setup (Turbo)
    cmap = plt.cm.turbo

    # 4. Draw Grid
    for y in range(grid_height):
        for x in range(grid_width):
            linear_idx = y * grid_width + x
            addr = linear_idx * mem_access.elem_width

            bank_id = (addr // bank_width_bytes) % num_banks

            # Normalize bank ID (0-31) to 0.0-1.0
            color_val = bank_id / (num_banks - 1)
            cell_color = cmap(color_val)

            # Draw top-down
            y_pos = grid_height - y - 1
            tidx = idx_to_thread.get(linear_idx, -1)

            if tidx != -1:
                rect = patches.Rectangle(
                    (x, y_pos),
                    1,
                    1,
                    facecolor=cell_color,
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

                # Text contrast
                text_color = "black" if 0.2 < color_val < 0.8 else "white"
                ax.text(
                    x + 0.5,
                    y_pos + 0.5,
                    f"T{tidx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )
            else:
                rect = patches.Rectangle(
                    (x, y_pos),
                    1,
                    1,
                    facecolor="#f5f5f5",
                    edgecolor="#ebebeb",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

    # 5. Axis Formatting
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)

    # X-Axis: Byte Offsets
    ax.set_xticks(np.arange(grid_width) + 0.5)
    ax.set_xticklabels(
        [
            str(i * mem_access.elem_width) if i % 4 == 0 else ""
            for i in range(grid_width)
        ],
        fontsize=7,
    )

    # --- Y-AXIS UPDATE: Row Start ADDRESS ---
    ax.set_yticks(np.arange(grid_height) + 0.5)

    # Logic: (Visual Row Index -> Logical Row Index) * LD * Elem_Width
    y_labels = [
        str((grid_height - 1 - i) * ld * mem_access.elem_width)
        for i in range(grid_height)
    ]

    ax.set_yticklabels(y_labels, fontsize=9)

    # Spines
    ax.tick_params(left=False, bottom=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title
    title_parts = [f"{mem_access.__class__.__name__}"]
    if hasattr(mem_access, "swizzle_bits"):
        title_parts.append(f"Swizzle={mem_access.swizzle_bits}b")

    ax.set_title(" | ".join(title_parts), fontsize=10, fontweight="bold")
    ax.set_ylabel("Row Start Addr", fontsize=9)


def visualize_access_patterns(access_list: list, num_threads=32):
    """
    Main function to plot a list of memory access patterns as subplots.
    """
    num_plots = len(access_list)

    # Determine Figure Size
    # We estimate height based on the 'tallest' access pattern roughly,
    # or just give a fixed decent height per subplot.
    subplot_height = 5
    subplot_width = 12

    fig, axes = plt.subplots(
        nrows=num_plots,
        ncols=1,
        figsize=(subplot_width, subplot_height * num_plots),
        sharex=False,  # Don't share X in case LDs are different
    )

    # Ensure axes is always iterable (even if only 1 plot)
    if num_plots == 1:
        axes = [axes]

    print(f"Visualizing {num_plots} access patterns...")

    for ax, access in zip(axes, access_list):
        _plot_on_axis(ax, access, num_threads)

    # Add a shared X-label at the bottom
    axes[-1].set_xlabel("Byte Offset (Logical)", fontsize=10)

    plt.tight_layout()
    plt.show()


vec_access_ld32 = VectorizedGlobalAccess(ld=32, elems_per_thread=4, elem_width=4)
vec_access_ld40 = VectorizedGlobalAccess(ld=36, elems_per_thread=4, elem_width=4)
store_c_frag_access = StoreCfragAccess()
store_c_frag_access_ld40 = StoreCfragAccess(ld=40)
store_c_frag_access_swizzle = StoreCfragSwizzled(
    ld=32, elem_width=4, elems_per_thread=4, swizzle_bits=3, vector_shift=2
)
vec_access_swizzled = SwizzledAccess(
    32, elems_per_thread=4, elem_width=4, swizzle_bits=3, vector_shift=2
)
# visualize_access_patterns(
#    [
#        vec_access_ld32,
#        store_c_frag_access,
#    ]
# )
# visualize_access_patterns(
#    [
#        vec_access_ld40,
#        store_c_frag_access_ld40,
#    ]
# )
visualize_access_patterns([store_c_frag_access_swizzle, vec_access_swizzled])
