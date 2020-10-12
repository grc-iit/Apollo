import configparser
import json
import subprocess
from benchmark_maker import get_fio_command_list

def bash_jason(cmd):
    return json.loads(subprocess.check_output(cmd, shell=True).decode('utf-8'))

def run_one_fio(cmd):  
    try :
        print(bash_jason(cmd))
    except subprocess.CalledProcessError as e:
        print("Run fio_setup.sh and try again")

def main():
    conf=configparser.ConfigParser()
    conf.read('probe_config.config')
    fio_loc=conf['FIO']['fio_install']
    
    for i in get_fio_command_list():
        run_one_fio(fio_loc+'/'+i)
        
if __name__=='__main__':
    main()
    
    
