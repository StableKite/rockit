void rockit_feature_init(void) __asm__("_init");
void rockit_feature_init(void) {}
void rockit_feature_fini(void) __asm__("_fini");
void rockit_feature_fini(void) {}
unsigned char rockit_feature_bss_start_alias __asm__("__bss_start__");
unsigned char rockit_feature_bss_start_alias;
unsigned char rockit_feature_bss_end_alias __asm__("__bss_end__");
unsigned char rockit_feature_bss_end_alias;
unsigned char rockit_feature_bss_end_alias2 __asm__("_bss_end__");
unsigned char rockit_feature_bss_end_alias2;
unsigned char rockit_feature_end_alias __asm__("__end__");
unsigned char rockit_feature_end_alias;
unsigned char rockit_feature_bss_start __asm__("__bss_start");
unsigned char rockit_feature_bss_start;
unsigned char rockit_feature_edata __asm__("_edata");
unsigned char rockit_feature_edata;
unsigned char rockit_feature_end __asm__("_end");
unsigned char rockit_feature_end;
