import os, psutil
import glob
import collections
import pprint
#import tinydb as tdb
import json
import subprocess
#from get_ior_bandwidth import get_benchmark

def bytes_converter(b, unit='GB'):
    """
    Returns the b bytes in the specified unit
    parms: b <float>, unit <str> default='GB'
    Notes: to extend prev_num+1 if unit is 'XB' or unit is 'xb' else None  <- replace None with, to extend bytes_converter as needed
    """
    multiplier = (1 if unit is 'KB'or unit is 'kb' else 2 if unit is 'MB' or unit is 'mb' else 3 if unit is 'GB' or unit is 'gb' else None)
    return b/(1024**multiplier)

def get_mounted_partitions(unit='GB'):

    res_d=collections.defaultdict(list)
    for dev,mount_point,fs,opts in psutil.disk_partitions():
        res = {}
        a = psutil.disk_usage(path=mount_point)
#         print(mount_point)
        res['dev'] = dev
        res['mount_point'] = mount_point
        res['fs'] = fs
        res['opts']=opts
        res['memory_details_'+unit]={label_me(i):bytes_converter(j,unit) if j != a[-1] else j  for i, j in enumerate(a)}
        res_d[dev.split('/')[-1].rstrip('123456789')].append(res)
    return res_d

def label_me(x):
    return 'total' if x == 0 else 'used' if x == 1 else 'free' if x ==2 else 'percent' if x ==3 else 'Unknown'

def get_RAM(unit='GB'):
    a=psutil.virtual_memory()
    res = {'total_'+unit:bytes_converter(a[0],unit),
           'available_'+unit:bytes_converter(a[1],unit),
           'percent':a[2],
           'used_'+unit:bytes_converter(a[3],unit),
           'free_'+unit:bytes_converter(a[4],unit),
           'active_'+unit:bytes_converter(a[5],unit),
           'inactive_'+unit:bytes_converter(a[6],unit),
           'buffers_'+unit:bytes_converter(a[7],unit),
           'cached_'+unit:bytes_converter(a[8],unit),
           'shared_'+unit:bytes_converter(a[9],unit),
           'slab_'+unit:bytes_converter(a[10],unit),
          }
    return res

def get_swap_memory(unit='GB'):
    a=psutil.swap_memory()
    swap_res={}
    swap_res['total_'+unit]=bytes_converter(a[0],unit=unit)
    swap_res['used_'+unit]=bytes_converter(a[1],unit=unit)
    swap_res['free_'+unit]=bytes_converter(a[2],unit=unit)
    swap_res['percent_'+unit]=a[3]
    swap_res['sin_'+unit]=bytes_converter(a[4],unit=unit)
    swap_res['sout_'+unit]=bytes_converter(a[5],unit=unit)
    return swap_res

def get_block_info():
    cmd = 'lsblk -JS -io name,model,rota,WWN,HCTL,vendor,hotplug,subsystems,tran,min-io,opt-io,sched'
    blocks = json.loads(subprocess.check_output(cmd, shell=True).decode('utf-8'))
    c_block=collections.defaultdict(list)
    for idx, block in enumerate(blocks['blockdevices']):
        t_d={}
        for i,j in block.items():
            if i == 'rota':
                t_d['disk_type']= 'SSD' if j == False else 'HDD'
            elif i == 'wwn':
                t_d['unique_strorage_id']=j
            elif i =='hctl':
                t_d['Host:Channel:Target:Lun']= j
            else:
                t_d[i]=j
        c_block[blocks['blockdevices'][idx]['name']].append(t_d)
    return c_block
    
    

def print_hashlines(n):
    print('#'*n)
    
    
def main():
    print_hashlines(80)
    print("Block Information")
    print_hashlines(80)
    pprint.pprint(get_block_info())

    print_hashlines(80)
    print("Mounted Partition Information")
    print_hashlines(80)
    pprint.pprint(get_mounted_partitions())

    print_hashlines(80)
    print("RAM and Swap Information")
    print_hashlines(80)
    swap_ram={
        "swap":get_swap_memory(),
        "ram":get_RAM(),
        }
    pprint.pprint(swap_ram)
    
    #print_hashlines(80)
    #print("IOR Benchmarks")
    #print_hashlines(80)
    #pprint.pprint(get_benchmark())

if __name__ == '__main__':
    main()
