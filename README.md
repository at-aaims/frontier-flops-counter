# frontier-flops-counter

## rocprof_wrapper.sh 
This is used to wrap around rocprof so that it can create separate output file for each of the ranks running rocprof.

## Providing hardware counter lists to the wrapper
inputs-fp16.txt has the list of required counters. This file is already used in the wrapper script, nothing else needs to be done.
```
pmc : SQ_INSTS_VALU_ADD_F16 SQ_INSTS_VALU_MUL_F16 SQ_INSTS_VALU_FMA_F16 SQ_INSTS_VALU_TRANS_F16 SQ_INSTS_VALU_MFMA_MOPS_F16 
pmc : SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_TRANS_F32 SQ_INSTS_VALU_MFMA_MOPS_F32 
pmc : SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64 SQ_INSTS_VALU_TRANS_F64 SQ_INSTS_VALU_MFMA_MOPS_F64 
pmc : SQ_INSTS_VALU_MFMA_BF16 SQ_INSTS_VALU_MFMA_MOPS_BF16
```

## Using the wrapper from the launch script
```
srun -u -n$ranks_total -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest ./rocprof_wrapper_22b.sh python pretrain_gpt_deepspeed.py $GPT_ARGS
```

## count_flops.py counts flops from the hardware counters
