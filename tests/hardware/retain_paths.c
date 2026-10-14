/* Candidate-only reference ownership regression. This is NOT a vendor codec/AVS
 * equivalence test: MPP is explicitly disabled and no hardware is used.
 * Run each case in an isolated process with an external timeout. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "rk_mpi_sys.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_ai.h"
#include "rk_mpi_vpss.h"
#include "rk_mpi_avs.h"
#include "rk_mpi_vdec.h"
#define OK(expr) do { RK_S32 result_ = (expr); if (result_ != RK_SUCCESS) { fprintf(stderr,"FAIL line=%d call=%s rc=%d\n",__LINE__,#expr,result_); return 11; } } while (0)
#define CHECK(expr) do { if (!(expr)) { fprintf(stderr,"FAIL line=%d check=%s\n",__LINE__,#expr); return 12; } } while (0)
static int one(const char *which) {
    MB_BLK b=MB_INVALID_HANDLE; OK(RK_MPI_SYS_Malloc(&b, 6144));
    CHECK(b != MB_INVALID_HANDLE); CHECK(RK_MPI_MB_InquireUserCnt(b)==1);
    unsigned char *p=(unsigned char*)RK_MPI_MB_Handle2VirAddr(b); CHECK(p); memset(p,0x5a,6144);
    VIDEO_FRAME_INFO_S f={0},out={0}; f.stVFrame.pMbBlk=b;
    f.stVFrame.u32Width=64; f.stVFrame.u32Height=64;
    f.stVFrame.u32VirWidth=64; f.stVFrame.u32VirHeight=64;
    f.stVFrame.enPixelFormat=RK_FMT_YUV420SP;
    f.stVFrame.enCompressMode=COMPRESS_MODE_NONE; f.stVFrame.u64PTS=123456;
    if (!strcmp(which,"ai")) {
        AIO_ATTR_S a={0}; a.enSamplerate=AUDIO_SAMPLE_RATE_48000; a.enBitwidth=AUDIO_BIT_WIDTH_16;
        a.enSoundmode=AUDIO_SOUND_MODE_MONO; a.u32FrmNum=8; a.u32PtNumPerFrm=1024; a.u32ChnCnt=1;
        AUDIO_FRAME_S af={0},got={0}; af.pMbBlk=b; af.u32Len=2048; af.u64TimeStamp=123456;
        af.enBitWidth=AUDIO_BIT_WIDTH_16; af.enSoundMode=AUDIO_SOUND_MODE_MONO;
        OK(RK_MPI_AI_SetPubAttr(0,&a)); OK(RK_MPI_AI_Enable(0)); OK(RK_MPI_AI_EnableChn(0,0));
        OK(RK_MPI_AI_SendFrame(0,0,&af,0)); CHECK(RK_MPI_MB_InquireUserCnt(b)==2);
        OK(RK_MPI_AI_GetFrame(0,0,&got,NULL,0)); CHECK(got.pMbBlk==b); CHECK(got.u32Len==2048);
        CHECK(got.u64TimeStamp==123456); CHECK(RK_MPI_MB_InquireUserCnt(b)==2);
        OK(RK_MPI_AI_ReleaseFrame(0,0,&got,NULL)); CHECK(RK_MPI_MB_InquireUserCnt(b)==1);
        OK(RK_MPI_AI_DisableChn(0,0)); OK(RK_MPI_AI_Disable(0)); OK(RK_MPI_AI_ClrPubAttr(0));
    } else if (!strcmp(which,"vpss")) {
        VPSS_GRP_ATTR_S a={0}; a.u32MaxW=64;a.u32MaxH=64;a.enPixelFormat=RK_FMT_YUV420SP;a.u32MaxQueue=2;
        OK(RK_MPI_VPSS_CreateGrp(0,&a)); OK(RK_MPI_VPSS_StartGrp(0));
        OK(RK_MPI_VPSS_SendFrame(0,0,&f,0)); CHECK(RK_MPI_MB_InquireUserCnt(b)==2);
        OK(RK_MPI_VPSS_GetGrpFrame(0,0,&out)); CHECK(out.stVFrame.pMbBlk==b); CHECK(out.stVFrame.u64PTS==123456);
        OK(RK_MPI_VPSS_ReleaseGrpFrame(0,0,&out)); CHECK(RK_MPI_MB_InquireUserCnt(b)==1);
        OK(RK_MPI_VPSS_StopGrp(0)); OK(RK_MPI_VPSS_DestroyGrp(0));
    } else if (!strcmp(which,"avs")) {
        AVS_GRP_ATTR_S a={0}; a.u32PipeNum=1;a.enMode=AVS_MODE_NOBLEND_HOR;
        OK(RK_MPI_AVS_CreateGrp(0,&a));OK(RK_MPI_AVS_StartGrp(0));
        OK(RK_MPI_AVS_SendPipeFrame(0,0,&f,0)); CHECK(RK_MPI_MB_InquireUserCnt(b)==3);
        OK(RK_MPI_AVS_GetPipeFrame(0,0,&out)); CHECK(out.stVFrame.pMbBlk==b);CHECK(out.stVFrame.u64PTS==123456);
        OK(RK_MPI_AVS_ReleasePipeFrame(0,0,&out));CHECK(RK_MPI_MB_InquireUserCnt(b)==2);
        OK(RK_MPI_AVS_StopGrp(0));OK(RK_MPI_AVS_DestroyGrp(0));CHECK(RK_MPI_MB_InquireUserCnt(b)==1);
    } else if (!strcmp(which,"vdec")) {
        VDEC_CHN_ATTR_S a={0}; a.enType=RK_VIDEO_ID_AVC;a.enMode=VIDEO_MODE_FRAME;
        a.u32PicWidth=64;a.u32PicHeight=64;a.u32FrameBufCnt=2;a.u32StreamBufSize=6144;
        VDEC_STREAM_S s={0};s.pMbBlk=b;s.u32Len=32;s.bEndOfFrame=RK_TRUE;s.u64PTS=123456;
        OK(RK_MPI_VDEC_CreateChn(0,&a));OK(RK_MPI_VDEC_StartRecvStream(0));
        OK(RK_MPI_VDEC_SendStream(0,&s,0)); CHECK(RK_MPI_MB_InquireUserCnt(b)==2);
        OK(RK_MPI_VDEC_GetFrame(0,&out,0));CHECK(out.stVFrame.pMbBlk==b);CHECK(out.stVFrame.u64PTS==123456);
        OK(RK_MPI_VDEC_ReleaseFrame(0,&out));CHECK(RK_MPI_MB_InquireUserCnt(b)==1);
        OK(RK_MPI_VDEC_StopRecvStream(0));OK(RK_MPI_VDEC_DestroyChn(0));
    } else { fprintf(stderr,"unknown case\n");return 2; }
    CHECK(RK_MPI_MB_InquireUserCnt(b)==1); for (unsigned i=0;i<6144;++i)CHECK(p[i]==0x5a);
    OK(RK_MPI_MB_ReleaseMB(b));return 0;
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    const char *provider=getenv("ROCKIT_MPP_PROVIDER");
    if(!provider||strcmp(provider,"disabled")){fprintf(stderr,"requires ROCKIT_MPP_PROVIDER=disabled (candidate model only)\n");return 3;}
    OK(RK_MPI_SYS_Init());
    for(int i=0;i<20;++i){int rc=one(argv[1]);if(rc)return rc;}
    OK(RK_MPI_SYS_Exit());printf("RETAIN_PATH_PASS case=%s cycles=20 payload_bytes=6144 hardware=false\n",argv[1]);return 0;
}
