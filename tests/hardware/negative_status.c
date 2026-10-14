/* TEST-ONLY negative control: successful real increment is intentionally
 * misreported as positive count. No production source or library is changed. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include "rk_mpi_mb.h"
RK_S32 RK_MPI_MB_AddUserCnt(MB_BLK b) {
    RK_S32 (*real_add)(MB_BLK)=(RK_S32 (*)(MB_BLK))dlsym(RTLD_NEXT,"RK_MPI_MB_AddUserCnt");
    if(!real_add)return -1;
    RK_S32 rc=real_add(b);
    return rc==RK_SUCCESS ? RK_MPI_MB_InquireUserCnt(b) : rc;
}
