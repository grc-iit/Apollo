#include <sys/sysinfo.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <ios>
#include <fstream>
#include <unistd.h>
#include <sys/statvfs.h>
#include <map>
#include "probing.h"
#include <thread>         // std::thread
#include <pthread.h>
#include <iomanip>
using namespace std;

int monitor(map<const char *,struct mnt_info>mountings){
  unsigned int mainMicroseconds;
  unsigned int ms=1000000;
  mainMicroseconds=1000000;
  struct statvfs fs;
  struct sysinfo ram;
  //while (mainMicroseconds < 100000000){
  while (1){
    usleep(ms);
    mainMicroseconds=mainMicroseconds+ms;
      for (auto  x : mountings)
      {
        statvfs(x.first,&fs);
        x.second.mem_total = fs.f_bsize * fs.f_blocks/1073741824.0;
        x.second.mem_used = x.second.mem_total - fs.f_bsize * fs.f_bavail/1073741824.0;
        x.second.mem_free=fs.f_bsize * fs.f_bfree/1073741824.0;
      }
      mountings["/home"].mem_used=mountings["/home"].mem_total - fs.f_bsize * fs.f_bavail/1073741824.0;
  }
}
int print_monitor(map<const char *,struct mnt_info>mountings){
  unsigned int mainMicroseconds;
  unsigned int ms=1000000;
  mainMicroseconds=1000000;
  struct statvfs fs;
  ofstream ofile (".MONITOR", ofstream::out);
  //while (mainMicroseconds < 100000000){
  while (1){
    usleep(ms);
    mainMicroseconds=mainMicroseconds+ms;
    ofile.open(".MONITOR");
      for (auto  x : mountings)
      {
        if (ofile.is_open())
        {
           statvfs(x.first,&fs);
          x.second.mem_total = fs.f_bsize * fs.f_blocks/1073741824.0;
          x.second.mem_used = x.second.mem_total - fs.f_bsize * fs.f_bavail/1073741824.0;
          x.second.mem_free=fs.f_bsize * fs.f_bfree/1073741824.0;
          ofile << x.first  // string (key)
          << "\tused:" 
          << x.second.mem_used // string's value 
          << "\t total:" 
          << x.second.mem_total
          << "\tfree:" 
          << x.second.mem_free
          << "\tfs:"
          << x.second.fs
          << "\tdev:"
          << x.second.dev
          <<  endl ;
        }
      }
      
    struct sysinfo ram;
    sysinfo(&ram);
          ofile <<"Total RAM:" << ram.totalram *ram.mem_unit /1024/1024.0 <<endl; // string (key)
          ofile <<"Total FREE RAM:" << ram.freeram *ram.mem_unit /1024/1024.0 <<endl; // string (key)
          ofile <<"Total SWAP:" << ram.totalswap *ram.mem_unit /1024/1024.0 <<endl; // string (key)
          ofile <<"Total FREE SWAP:" << ram.freeswap *ram.mem_unit /1024/1024.0 <<endl; // string (key)
          ofile <<"Avg Load 1 min:" << ram.loads[0] *ram.mem_unit /1024/1024.0 <<endl; // string (key)
    ofile.close();

  }
}
int main() 
{
  map<const char *,struct mnt_info>mountings;
  struct sysinfo ram;
  struct statvfs fs;
   const double megabyte = 1024 * 1024;
  for (int y; y<sizeof(cstrs)/sizeof(cstrs[0]);y++){
    mountings[cstrs[y]]=point[y];
  }
  for (auto & x : mountings)
  {
    statvfs(x.first,&fs);
    x.second.mem_total = fs.f_bsize * fs.f_blocks/1073741824.0;
    x.second.mem_used = x.second.mem_total - fs.f_bsize * fs.f_bavail/1073741824.0;
    x.second.mem_free=fs.f_bsize * fs.f_bfree/1073741824.0;
    cout << x.first  // string (key)
              << "\tused:" 
              << x.second.mem_used // string's value 
              << "\t total:" 
              << x.second.mem_total
              << "\tfree:" 
              << x.second.mem_free
              <<  endl ;
  }

  const char *file = "/";
  thread thread_monitor (monitor,mountings);
  usleep(500000);
  thread thread_pmonitor (print_monitor,mountings);
  usleep(10000000);
  thread_monitor.join();
  /*int statvfs(const char *path, struct statvfs *buf);*/
  return 0;
}
