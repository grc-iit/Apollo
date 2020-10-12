import configparser
import json
import subprocess

def bash_jason(cmd):
    return json.loads(subprocess.check_output(cmd, shell=True).decode('utf-8'))

if __name__ == '__main__':
    conf=configparser.ConfigParser()
    conf.read('probe_config.config')
    fio_loc=dict(conf.defaults().items())['fio_install']
    try :
        print(bash_jason(fio_loc+'/fio --output-format=json --name=global --rw=read --size=4K --name=job1 --directory=/home/neeraj --direct=1 --fsync=1 '))
    except subprocess.CalledProcessError as e:
        print("Run fio_setup.sh and try again")
