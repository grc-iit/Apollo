#include <string>
using namespace std;
struct mnt_info { string mount_point;
                  string scsi;
                  string fs;
                  string dev;
                  double mem_total;
                  double mem_used;
                  double mem_free;
                  string opts; 
                };
const char * cstrs[] = {"/","/home"};
//struct mnt_info point[];
struct mnt_info  point[] = {{"/","sda","ext4","/dev/sda6"},{"/home","sda","ext4","/dev/sda8"}};