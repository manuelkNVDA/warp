

import numpy as np

import warp as wp

if __name__ == "__main__":
    wp.clear_kernel_cache()
    device = wp.get_device("cuda:0")
    wp.context.runtime.core.wp_bvh_debug_device(device.context, b"d:/tmp/aabb_dump.bin");

