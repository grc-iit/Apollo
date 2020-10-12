import json, os
def bashing(command,printToScreen=True):
    os.system(command + " > QbashFile")
    with open("QbashFile") as f:
        var=f.read()
    os.remove('QbashFile')
    if printToScreen:
        print(var)
    else:
        return var
def get_benchmark():
    os.system('ior -a=POSIX -O summaryFile=sequential.json -O summaryFormat=JSON')
    os.system('ior -a=POSIX -z -O summaryFile=random.json -O summaryFormat=JSON')
    with open('sequential.json') as f :
        sequential = json.load(f)
    with open('random.json') as f :
        random = json.load(f)
    filesystem  = bashing('df -P sequential.json',False)
    filesystem=filesystem[filesystem.find('\n'):].split()[0]
    read=1
    write=0
    return {
        'filesystem': filesystem,
        sequential['tests'][0]['Options']['ordering in a file'] :
        {
            sequential['summary'][read]['operation']:{
            'bwMaxMIB' :    sequential['summary'][read]['bwMaxMIB'],
            'bwMinMIB' :    sequential['summary'][read]['bwMinMIB'],
            'bwMeanMIB' :    sequential['summary'][read]['bwMeanMIB'],
            'openTime' :    sequential['tests'][0]['Results'][0][read]['openTime'],
            'wrRdTime' :    sequential['tests'][0]['Results'][0][read]['wrRdTime'],
            'totalTime' :    sequential['tests'][0]['Results'][0][read]['totalTime']
            },
             sequential['summary'][write]['operation']:{
            'bwMaxMIB' :    sequential['summary'][write]['bwMaxMIB'],
            'bwMinMIB' :    sequential['summary'][write]['bwMinMIB'],
            'bwMeanMIB' :    sequential['summary'][write]['bwMeanMIB'],
            'openTime' :    sequential['tests'][0]['Results'][0][write]['openTime'],
            'wrRdTime' :    sequential['tests'][0]['Results'][0][write]['wrRdTime'],
            'totalTime' :    sequential['tests'][0]['Results'][0][write]['totalTime']
            }
        },
                random['tests'][0]['Options']['ordering in a file'] :
        {
            random['summary'][read]['operation']:{
            'bwMaxMIB' :    random['summary'][read]['bwMaxMIB'],
            'bwMinMIB' :    random['summary'][read]['bwMinMIB'],
            'bwMeanMIB' :    random['summary'][read]['bwMeanMIB'],
            'openTime' :    random['tests'][0]['Results'][0][read]['openTime'],
            'wrRdTime' :    random['tests'][0]['Results'][0][read]['wrRdTime'],
            'totalTime' :    random['tests'][0]['Results'][0][read]['totalTime']
            },
             random['summary'][write]['operation']:{
            'bwMaxMIB' :    random['summary'][write]['bwMaxMIB'],
            'bwMinMIB' :    random['summary'][write]['bwMinMIB'],
            'bwMeanMIB' :    random['summary'][write]['bwMeanMIB'],
            'openTime' :    random['tests'][0]['Results'][0][write]['openTime'],
            'wrRdTime' :    random['tests'][0]['Results'][0][write]['wrRdTime'],
            'totalTime' :    random['tests'][0]['Results'][0][write]['totalTime']
            }
        }
    }
sample =get_benchmark()
print(json.dumps(sample, sort_keys=True, indent=4))