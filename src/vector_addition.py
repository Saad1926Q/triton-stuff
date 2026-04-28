import torch
import triton
import triton.language as tl

DEVICE=triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(ptr1,ptr2,result_ptr,n_elements,BLOCK_SIZE:tl.constexpr):

    pid=tl.program_id(axis=0) # Get the program ID as we will have multiple programs running on different blocks of the vectors

    block_start=pid*BLOCK_SIZE # Starting index of block

    offsets=block_start+tl.arange(0,BLOCK_SIZE)

    mask=offsets<n_elements # To prevent out of bound access

    x=tl.load(ptr1+offsets,mask=mask)
    y=tl.load(ptr2+offsets,mask=mask)

    result=x+y

    tl.store(result_ptr+offsets,result,mask=mask)



def add(x1:torch.Tensor,x2:torch.Tensor):
    output=torch.empty_like(x1)

    assert x1.device == DEVICE and x2.device==DEVICE and output.device == DEVICE

    n_elements=output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements,meta["BLOCK_SIZE"]),)

    add_kernel[grid](x1,x2,output,n_elements,BLOCK_SIZE=1024)

    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
