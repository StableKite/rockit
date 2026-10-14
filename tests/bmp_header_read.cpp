// Regression for checked I/O in the example BMP header reader.
#include "test_comm_bmp.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
int main() {
    char path[]="rockit-bmp-test-XXXXXX";
    int fd=mkstemp(path); if(fd<0) return 1;
    FILE *file=fdopen(fd,"wb"); if(!file){close(fd);unlink(path);return 2;}
    const RK_U16 type=0x4d42; OSD_BITMAPFILEHEADER header{}; OSD_BITMAPINFO info{};
    bool written=fwrite(&type,1,sizeof(type),file)==sizeof(type) &&
      fwrite(&header,1,sizeof(header),file)==sizeof(header) &&
      fwrite(&info,1,sizeof(info),file)==sizeof(info);
    const int closed=fclose(file);
    if(!written || closed!=0){unlink(path);return 3;}
    OSD_BITMAPFILEHEADER gotHeader{}; OSD_BITMAPINFO gotInfo{};
    const int complete=TEST_COMM_GetBmpInfo(path,&gotHeader,&gotInfo);
    file=fopen(path,"wb"); if(!file){unlink(path);return 4;} // ordinary empty-file I/O failure
    if(fclose(file)!=0){unlink(path);return 5;}
    const int empty=TEST_COMM_GetBmpInfo(path,&gotHeader,&gotInfo);
    unlink(path);
    if(complete!=0 || empty==0) return 6;
    std::puts("BMP_HEADER_READ_PASS complete=accepted empty=rejected");return 0;
}
