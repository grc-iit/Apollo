#!/usr/bin/env python3

import json
import subprocess
import os

def all_checker(inp):
    if not inp:
        return False
    return True if len(inp.split(',')) == 1 and inp[0] == '0' else False

def gimme_choice(mnt,prompt):
    print(prompt)
    for i,j in enumerate(mnt):
        print(i+1,j)

    while True:
        try:
            inp=input('Select index (comma separated) [0 : all]:')
                
            if all_checker(inp):
                selects= list(range(len(mnt)))
            else:
                selects=list(map(lambda x: int(x)-1,inp.split(',')))
                for i in selects:
                    if i > len(mnt) or i < 0:
                        raise ValueError('Invalid index value')
            break
        except ValueError as e:
            print(e)
            pass
    return selects

def select2fio_ops(select, ops):
    for i in select:
        yield ops[i] if type(ops[i]) != tuple else ops[i][1]

def bash_jason(cmd):
    return json.loads(subprocess.check_output(cmd, shell=True).decode('utf-8'))

def mount_lookup_child(child):
    for children in child:
        if 'children' in children.keys():
            for x in mount_lookup_child(children['children']):
                yield (x)
        yield (children['name'],children['mountpoint'])
def breadthFirstFileScan( root ,mounting,depth=3) :
    dirs = [(root,0)]
    level=0
    while len(dirs) :
        nextDirs = []
        for (parent,level) in dirs :
            try:
                for f in os.listdir( parent ) :
                    ff = os.path.join( parent, f )
                    if os.path.isdir( ff ) :
                        if get_mounting_point(ff)==mounting:
                            if level < depth:
                                nextDirs.append( (ff,level+1))
                            yield ff
            except Exception as e:
                #print("Could not get into ",parent,e)
                pass
        dirs = nextDirs
def get_mounting_point(path):
    out = subprocess.check_output(("df '"+path+"'"), shell=True).decode('utf-8')
    out = str(out.split()[-1])
    return out
def findWritableDirectory( path ,mounting,depth=3) :
    for f in breadthFirstFileScan( path,mounting,depth ) :
        if os.access(f,os.W_OK) and get_mounting_point(f) == mounting:
            return f
        pass
def get_fio_command_list():
    mounts = bash_jason("lsblk -J -io name,mountpoint")

    mnt = list(filter(lambda x : x[1] is not None ,\
                      mount_lookup_child(mounts['blockdevices'])))
    fio_rw_ops= ['read',
                 'write',
                 'randread',
                 'randwrite',
                 'rw',
                 'readwrite',
                 'randrw',
                 ]
    fio_size_ops = ['4K',
                    '8K',
                    '16K',
                    '32K',
                    '1M',
                    '4M',
                    '8M',
                    '16M',
                     ]
    fio_ioengine_ops=['sync',
                      'psync',
                      'vsync',
                      'libaio',
                      'posixaio',
                      'mmap',
                      'syslet-rw',
                      'sg',
                      'guasi',
                      'rdma',
                      'falloc',
                      'e4defrag',
                      ]
    selects_mounts=gimme_choice(mnt,"Index, (name, mount point)")
    selects_rw=gimme_choice(fio_rw_ops, "Index, RW_OPTs")
    selects_size=gimme_choice(fio_size_ops,"Index, Size opts")
    selects_ioengine = gimme_choice(fio_ioengine_ops,"Index, IO Engines")

    mount_points=list(select2fio_ops(selects_mounts,mnt))
    rw_ops=list(select2fio_ops(selects_rw,fio_rw_ops))
    size_ops=list(select2fio_ops(selects_size,fio_size_ops))
    ioengines=list(select2fio_ops(selects_ioengine,fio_ioengine_ops))
    #print('this is mo',mount_points)
    writableDirectory=[]
    for m in mount_points:
        path = findWritableDirectory(m ,m)
        if path:
            writableDirectory.append((path,m))
            
    for dire , mpoint in writableDirectory:
        for rw in rw_ops:
            for size in size_ops:
                for io in ioengines:
                    strTemplate= "fio --output-format=json --name=global --ioengine={} "+\
                    "--rw={} --size={} --name=job1 --directory={} --direct=1 --fsync=1"
                    yield strTemplate.format(io,rw,size,dire),mpoint


if __name__ == '__main__':
    for fioCommand,mnt_point in get_fio_command_list():
        print(mnt_point,fioCommand)
