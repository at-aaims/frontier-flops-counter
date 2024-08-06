import pandas as pd

input_file = "results-407082_0.csv"
input_file = "results-408205_0.csv"
input_file = "profs175b/results_0.csv"

'''
'Index', 'KernelName', 'gpu-id', 'queue-id', 'queue-index', 'pid',
       'tid', 'grd', 'wgr', 'lds', 'scr', 'arch_vgpr', 'accum_vgpr', 'sgpr',
       'wave_size', 'sig', 'obj', 'SQ_INSTS_VALU_ADD_F16',
       'SQ_INSTS_VALU_MUL_F16', 'SQ_INSTS_VALU_FMA_F16',
       'SQ_INSTS_VALU_TRANS_F16', 'SQ_INSTS_VALU_MFMA_MOPS_F16',
       'SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32',
       'SQ_INSTS_VALU_FMA_F32', 'SQ_INSTS_VALU_TRANS_F32',
       'SQ_INSTS_VALU_MFMA_MOPS_F32', 'SQ_INSTS_VALU_ADD_F64',
       'SQ_INSTS_VALU_MUL_F64', 'SQ_INSTS_VALU_FMA_F64',
       'SQ_INSTS_VALU_TRANS_F64', 'SQ_INSTS_VALU_MFMA_MOPS_F64',
       'SQ_INSTS_VALU_MFMA_BF16', 'SQ_INSTS_VALU_MFMA_MOPS_BF16', 'DispatchNs',
       'BeginNs', 'EndNs', 'CompleteNs']
'''

def measure_time(time_file):
    total_time = 0

    reader = open(time_file, "r")

    for line in reader:
        tokens = line.split("|")
        elapsed = float(tokens[3].strip().split()[-1])
        total_time += elapsed
        print(elapsed / 1E3)

    total_time = total_time / 1E3
    return total_time

def count_flops_fp16(df):
    tflops_fp16_arr = []
    total_tflops = 0
    total_duration = 0
    for idx, row in df.iterrows():
        flops_fp16 = 64 * (row['SQ_INSTS_VALU_MUL_F16'] + 
                           row['SQ_INSTS_VALU_ADD_F16'] + 
                           2 * row['SQ_INSTS_VALU_FMA_F16'] + 
                           row['SQ_INSTS_VALU_TRANS_F16']
                           ) 
        flops_fp16 = flops_fp16 + 1024 * row['SQ_INSTS_VALU_MFMA_MOPS_F16'] #+ 1024 * row['SQ_INSTS_VALU_MFMA_MOPS_BF16']
        tflops_fp16 = 1.0 * flops_fp16 / (10**12)
        duration = row['EndNs'] - row['BeginNs']
        tflops_fp16_arr.append([row['KernelName'], tflops_fp16, duration])
        total_duration = total_duration + (duration / 1E9)
         
        total_tflops = total_tflops + tflops_fp16 
        #print(tflops_fp16, duration / 1E9)
    print(total_tflops / total_duration)
    print(int(1E9))

def count_flops_mixed(df):
    tflops_arr = []
    total_tflops = 0
    for idx, row in df.iterrows():
         
        flops_fp16 = 64 * (row['SQ_INSTS_VALU_MUL_F16'] + 
                           row['SQ_INSTS_VALU_ADD_F16'] + 
                           2 * row['SQ_INSTS_VALU_FMA_F16'] + 
                           row['SQ_INSTS_VALU_TRANS_F16']
                           ) 
    
        flops_mfma = 1024 * row['SQ_INSTS_VALU_MFMA_MOPS_F16'] + 1024 * row['SQ_INSTS_VALU_MFMA_MOPS_BF16'] + 256 * row['SQ_INSTS_VALU_MFMA_MOPS_F32'] + 256 * row['SQ_INSTS_VALU_MFMA_MOPS_F64']
    
        flops_fp32 = 64 * (row['SQ_INSTS_VALU_MUL_F32'] + 
                           row['SQ_INSTS_VALU_ADD_F32'] + 
                           2 * row['SQ_INSTS_VALU_FMA_F32'] + 
                           row['SQ_INSTS_VALU_TRANS_F32']
                           ) 
        
        flops_fp64 = 64 * (row['SQ_INSTS_VALU_MUL_F64'] + 
                           row['SQ_INSTS_VALU_ADD_F64'] + 
                           2 * row['SQ_INSTS_VALU_FMA_F64'] + 
                           row['SQ_INSTS_VALU_TRANS_F64']
                           )
        
        mixed_flops = flops_mfma + flops_fp16 + flops_fp32 + flops_fp64
        
        
        mixed_tflops = 1.0 * mixed_flops / (1E12)
        total_tflops = total_tflops + mixed_tflops
        tflops_arr.append([row['KernelName'], mixed_tflops])

return total_tflops

def count_flops(input_file):
    df = pd.read_csv(input_file)
    return count_flops_mixed(df)    

def count_model_flops(time_file):
    total_tflops, total_elapsed = 0, 0
    with open(time_file, "r") as reader:
        for line in reader:
            #print(line)
            tokens = line.split("|")
            flops_token = tokens[-2]
            print("Flops token:", tokens[-2])
            tflops = float(flops_token.split()[1])
            print("Elapsed token:", tokens[3])
            elapsed_sec = float(tokens[3].strip().split()[-1]) / 1E3
            print("Flops per iteration =", tflops * elapsed_sec)
            total_tflops += (tflops * elapsed_sec)
            total_elapsed += elapsed_sec
    print("[Model] Flops per iteration:",  total_tflops / 5) 
    print("[Model] Total tflops and total elapsed", total_tflops, total_elapsed)
    print("[Model] TFLOPS per gcd per second=", total_tflops / total_elapsed)
    return total_tflops / total_elapsed

def count_hardware_flops(time_file, result_dir):
    total_tflops = 0
    import glob
    result_files = glob.glob(f"{result_dir}/*.csv")
    for i in range(len(result_files)):
        input_file = f"{result_dir}/results_{i}.csv"
        tflops = count_flops(input_file) 
        total_tflops += tflops
        print(f"GCD {i} has total TFLOPS {tflops}")
    
    print("[HW] Total TFLOPS =", total_tflops)
    print("[HW] TFLOPs per GCD=", total_tflops / len(result_files))
    
import sys

if __name__ == "__main__":
    args = sys.argv
    result_dir = args[1]    
    count_hardware_flops(result_dir)
