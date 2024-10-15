#pragma once

#include <stdint.h>

#ifdef __cplusplus
    extern "C" {
#endif


// Execution parameters taken from the command line
typedef enum {
    CPU,
    GPU
} e_device_t;


typedef struct  {    
    e_device_t device;
} Parameters;

/// Global state initialization
/// This function is called once before any other cpt_* function at the beginning of the program
void cpt_init(Parameters* params);

/// Function called by gstreamer to process the incoming frame
void cpt_process_frame(uint8_t* buffer, int width, int height, int stride);
    

#ifdef __cplusplus
    }
#endif