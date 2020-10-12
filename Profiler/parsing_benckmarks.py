import configparser
import subprocess,json

from benchmark_maker import get_fio_command_list, bash_jason

AllData=[]
class Benchmark:
    
    fields=[ "mnt_point",'ioengine','operation','block_size',
        'rd_bw','rd_clat','rd_slat','rd_lat','wr_bw','wr_clat','wr_slat','wr_lat']
    units=['','','','','bytes','ns','ns','ns','ns','ns','ns','ns']
    TABLE_HEADER="_"*132+"\n"+"|"+"|".join([f.center(10) for f in fields])+ "|"+"\n"+ \
        "|"+"|".join([f.center(10) for f in units])+ "|" + "\n" + \
        "|"+"|".join(["_"*10 for f in units])+ "|"
    TABLE_LINE="|"+"|".join(["_"*10 for f in units])+ "|"
    def __init__(self,json_benchmark,mount_point):
        self.mount_point=mount_point
        self.ioengine = json_benchmark['global options']['ioengine']
        self.operation = json_benchmark['global options']['rw']
        self.block_size = json_benchmark['global options']['size']
        self.read_bw = json_benchmark['jobs'][0]['read']['bw_bytes']
        self.read_clat=json_benchmark['jobs'][0]['read']['clat_ns']['max']
        self.read_slat=json_benchmark['jobs'][0]['read']['slat_ns']['max']
        self.read_lat=json_benchmark['jobs'][0]['read']['lat_ns']['max']
        self.write_bw = json_benchmark['jobs'][0]['write']['bw_bytes']
        self.write_clat=json_benchmark['jobs'][0]['write']['clat_ns']['max']
        self.write_slat=json_benchmark['jobs'][0]['write']['slat_ns']['max']
        self.write_lat=json_benchmark['jobs'][0]['write']['lat_ns']['max']
    def __str__(self):
        template ="'mnt_point' : '{}'"+ \
            ",'ioengine' : '{}'" + \
            ",'operation' : '{}'" + \
            ",'block_size' : '{}'" + \
            ",'read_bw' : '{}'" + \
            ",'read_clat' : '{}'" + \
            ",'read_slat' : '{}'" + \
            ",'read_lat' : '{}'" + \
            ",'write_bw' : '{}'" + \
            ",'write_clat' : '{}'" + \
            ",'write_slat' : '{}'" + \
            ",'write_lat' : '{}'"
        return '{ '+ template.format(self.mount_point,
                self.ioengine,
                self.operation,
                self.block_size,
                self.read_bw,
                self.read_slat,
                self.read_clat,
                self.read_lat,
                self.write_bw,
                self.write_slat,
                self.write_clat,
                self.write_lat) + '}'
    def table_print(self):
        template =  '|{:10s}|{:10s}|{:10s}|{:5s}|{:10s}|{:10s}|{:10s}|{:10s}|{:10s}|{:10s}|{:10s}|{:10s}|'
        print(template.format(self.mount_point.center(10),
        str(self.ioengine).center(10),
        str(self.operation).center(10),
        str(self.block_size).center(10),
        str(self.read_bw).center(10),
        str(self.read_slat).center(10),
        str(self.read_clat).center(10),
        str(self.read_lat).center(10),
        str(self.write_bw).center(10),
        str(self.write_slat).center(10),
        str(self.write_clat).center(10),
        str(self.write_lat).center(10) ))
    def append2file(self,filename):
        values=[str(self.mount_point),
        str(self.ioengine),
        str(self.operation),
        str(self.block_size),
        str(self.read_bw),
        str(self.read_slat),
        str(self.read_clat),
        str(self.read_lat),
        str(self.write_bw),
        str(self.write_slat),
        str(self.write_clat),
        str(self.write_lat)]
        with open(filename,'a') as ofile:
            ofile.write(",".join(values)+"\n")
        

def main_bench():
    conf=configparser.ConfigParser()
    conf.read('probe_config.config')
    fio_loc=conf['FIO']['fio_install']

    allBenchmarks=[]
    for command,mnt_point in get_fio_command_list():
        try:
            string_json = bash_jason(fio_loc+'/'+command)
            #print(json.dumps(string_json, indent=4, sort_keys=True))
            allBenchmarks.append(Benchmark(string_json,mnt_point))
        except  Exception as  e:
            print('Could not be executed',e)
    print(Benchmark.TABLE_HEADER)
    outputFile="benchmarks.csv"
    with open(outputFile,'w') as ofile:
        ofile.write(",".join(Benchmark.fields)+"\n")
    for benchmark in allBenchmarks:
        benchmark.table_print()
        benchmark.append2file(outputFile)
    print(Benchmark.TABLE_LINE)


if __name__ =='__main__':
    main_bench()
