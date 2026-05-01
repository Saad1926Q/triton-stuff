import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def naive_softmax(x: torch.Tensor):
    row_max, _ = x.max(dim=1)

    x = x - row_max.unsqueeze(1)

    numerator = x.exp()

    denominator = numerator.sum(dim=1).unsqueeze(1)

    return numerator / denominator


@triton.jit
def softmax_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = in_ptr + row_idx * in_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)

        mask = col_offsets < n_cols

        input_ptrs = row_start_ptr + col_offsets

        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        row_max = tl.max(row, axis=0)

        row = row - row_max

        numerator = tl.exp(row)

        denominator = tl.sum(numerator, axis=0)

        output = numerator / denominator

        output_row_start = out_ptr + row_idx * out_row_stride

        output_ptrs = output_row_start + col_offsets

        tl.store(output_ptrs, output, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties[
    "max_shared_mem"
]  # total shared memory available on the entire CU
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    def is_hip():
        return triton.runtime.driver.active.get_current_target().backend == "hip"

    def is_cdna():
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
            "gfx940",
            "gfx941",
            "gfx942",
            "gfx90a",
            "gfx908",
        )

    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8  # Hardcoded value for now
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    y = torch.empty_like(x)

    kernel = softmax_kernel.warmup(
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )

    kernel._init_handles()
    n_regs = kernel.n_regs  # Registers used per thread
    size_smem = kernel.metadata.shared  # Shared memory used per program, in bytes

    # Basically the main goal here is to be able to find out how many programs we will be running
    # That will be decided by how many SMs we have and also how many programs we can run per SM
    # Based on this we will calculate occupancy which tells us how many programs can run simultaneously per SM
    # We don't necessarily have to run the program for each row but just enough to fully saturate the GPU and then each program can process multiple rows

    if is_hip():  # AMD GPU
        NUM_GPRS = NUM_REGS

        if is_cdna():
            # CDNA is different from RDNA, which is AMD's consumer gaming architecture.
            NUM_GPRS = NUM_REGS * 2

        MAX_NUM_THREADS = properties["max_threads_per_sm"]

        # Basically wave is the same thing as warp but for amd

        # The hardware has a hard limit on how many threads can be alive on one CU(basically an SM if we're talking in Nvidia terms) at the same time.
        # So we have a ceiling on how many waves we can physically have
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE

        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps

    else:  # Nvidia GPUs
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

    occupancy = min(occupancy, SIZE_SMEM // size_smem)

    # This is how many programs we have to run to fully saturate the GPU
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](
        x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages
    )

    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch", "naive_softmax"],
        line_names=["Triton", "Torch", "Naive Softmax"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))

    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)
