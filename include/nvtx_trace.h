#include "nvToolsExt.h"

static const uint32_t colors4[] = { 0x000ff00, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff }; \
static const int num_colors4 = sizeof(colors4)/sizeof(uint32_t); \

#define nvtx_trace_start(name, cid) do { \
   int color_id = cid % num_colors4; \
   nvtxEventAttributes_t eventAttrib = {0}; \
   eventAttrib.version = NVTX_VERSION; \
   eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
   eventAttrib.colorType = NVTX_COLOR_ARGB; \
   eventAttrib.color = colors4[color_id]; \
   eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
   eventAttrib.message.ascii = name; \
   nvtxRangePushEx(&eventAttrib); \
   } while(0)

#define nvtx_trace_end() do {\
   nvtxRangePop();\
} while(0)
