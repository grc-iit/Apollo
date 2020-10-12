#!/usr/bin/env python3
import collections
import configparser
import subprocess

def stream_op_parser(op):

    table=op.split('-------------------------------------------------------------')[6]
    head=list(filter(lambda x:x!='',table.splitlines()[1].split('  ')) )
    copy=list(filter(lambda x:x!='',table.splitlines()[2].split('  ')) )
    scale=list(filter(lambda x:x!='',table.splitlines()[3].split('  ')) )
    add=list(filter(lambda x:x!='',table.splitlines()[4].split('  ')) )
    triad=list(filter(lambda x:x!='',table.splitlines()[5].split('  ')) )
    stream=collections.defaultdict(dict)
    stream['best_rate']['copy_MBps']=float(copy[1].strip())
    stream['avg_time']['copy']=float(copy[2].strip())
    stream['min_time']['copy']=float(copy[3].strip())
    stream['max_time']['copy']=float(copy[4].strip())

    stream['best_rate']['scale_MBps']=float(scale[1].strip())
    stream['avg_time']['scale']=float(scale[2].strip())
    stream['min_time']['scale']=float(scale[3].strip())
    stream['max_time']['scale']=float(scale[4].strip())

    stream['best_rate']['add_MBps']=float(add[1].strip())
    stream['avg_time']['add']=float(add[2].strip())
    stream['min_time']['add']=float(add[3].strip())
    stream['max_time']['add']=float(add[4].strip())

    stream['best_rate']['triad_MBps']=float(triad[1].strip())
    stream['avg_time']['triad']=float(triad[2].strip())
    stream['min_time']['triad']=float(triad[3].strip())
    stream['max_time']['triad']=float(triad[4].strip())

    return stream

if __name__ == "__main__":

    conf=configparser.ConfigParser()
    conf.read('probe_config.config')
    stream_loc=conf['STREAM']['stream_install']
    op = subprocess.check_output(stream_loc+'/stream_5-10_posix_memalign').decode('utf-8')

    #op="""-------------------------------------------------------------
    #STREAM version $Revision: 5.10 $
    #-------------------------------------------------------------
    #This system uses 8 bytes per array element.
    #-------------------------------------------------------------
    #Array size = 10000000 (elements), Offset = 0 (elements)
    #Memory per array = 76.3 MiB (= 0.1 GiB).
    #Total memory required = 228.9 MiB (= 0.2 GiB).
    #Each kernel will be executed 10 times.
     #The *best* time for each kernel (excluding the first iteration)
     #will be used to compute the reported bandwidth.
    #-------------------------------------------------------------
    #Your clock granularity/precision appears to be 1 microseconds.
    #Each test below will take on the order of 8555 microseconds.
       #(= 8555 clock ticks)
    #Increase the size of the arrays if this shows that
    #you are not getting at least 20 clock ticks per test.
    #-------------------------------------------------------------
    #WARNING -- The above is only a rough guideline.
    #For best results, please be sure you know the
    #precision of your system timer.
    #-------------------------------------------------------------
    #Function    Best Rate MB/s  Avg time     Min time     Max time
    #Copy:       11786.5121       0.0136       0.0136       0.0137
    #Scale:      11992.2916       0.0135       0.0133       0.0137
    #Add:        12791.7371       0.0199       0.0188       0.0289
    #Triad:      12986.9691       0.0191       0.0185       0.0234
    #-------------------------------------------------------------
    #Solution Validates: avg error less than 1.000000e-13 on all three arrays
    #-------------------------------------------------------------"""

    print(stream_op_parser(op))
