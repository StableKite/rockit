#include <stdio.h>
#include <string.h>
#include "rk_mpi_sys.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_cal.h"
int main(void) {
    PIC_BUF_ATTR_S attr = {0}; MB_PIC_CAL_S calc = {0};
    attr.u32Width=1920; attr.u32Height=1080; attr.enPixelFormat=RK_FMT_YUV420SP; attr.enCompMode=COMPRESS_MODE_NONE;
    RK_S32 cal=RK_MPI_CAL_COMM_GetPicBufferSize(&attr,&calc);
    printf("cal_rc=%d width=%u height=%u bytes=%u stride=%u\n",cal,calc.u32VirWidth,calc.u32VirHeight,calc.u32MBSize,RK_MPI_CAL_COMM_GetHorStride(1920,RK_FMT_YUV420SP));
    RK_S32 init=RK_MPI_SYS_Init(); printf("init_rc=%d\n",init); if(init!=RK_SUCCESS) return 2;
    MB_BLK block=MB_INVALID_HANDLE; RK_S32 alloc=RK_MPI_SYS_Malloc(&block,4096); printf("alloc_rc=%d\n",alloc);
    if(alloc!=RK_SUCCESS || block==MB_INVALID_HANDLE) {RK_MPI_SYS_Exit();return 3;}
    unsigned char *p=(unsigned char*)RK_MPI_MB_Handle2VirAddr(block); if(!p){RK_MPI_MB_ReleaseMB(block);RK_MPI_SYS_Exit();return 4;}
    memset(p,0x5a,4096); int bytes_ok=1;for(unsigned i=0;i<4096;++i)if(p[i]!=0x5a)bytes_ok=0;
    RK_S32 before=RK_MPI_MB_InquireUserCnt(block); RK_S32 added=RK_MPI_MB_AddUserCnt(block);
    printf("mb_size=%llu bytes_ok=%d before=%d add_result=%d",(unsigned long long)RK_MPI_MB_GetSize(block),bytes_ok,before,added);
    RK_S32 after=RK_MPI_MB_InquireUserCnt(block); printf(" after=%d\n",after);
    RK_S32 release1=RK_MPI_MB_ReleaseMB(block),release2=RK_MPI_MB_ReleaseMB(block); RK_S32 fini=RK_MPI_SYS_Exit();
    printf("release1=%d release2=%d exit_rc=%d\n",release1,release2,fini);
    return (!bytes_ok||before!=1||added!=RK_SUCCESS||after!=2||cal!=RK_SUCCESS||release1!=RK_SUCCESS||release2!=RK_SUCCESS||fini!=RK_SUCCESS)?5:0;
}
