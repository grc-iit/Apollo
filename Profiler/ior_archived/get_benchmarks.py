import json
import subprocess
def get_benchmark():
    sequential = json.loads(subprocess.check_output('ior -a=POSIX   -O summaryFormat=JSON', shell=True).decode('utf-8')) 
    random = json.loads(subprocess.check_output('ior -a=POSIX -z  -O summaryFormat=JSON', shell=True).decode('utf-8')) 
    filesystem  = subprocess.check_output("df -P .", shell=True).decode('utf-8')
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