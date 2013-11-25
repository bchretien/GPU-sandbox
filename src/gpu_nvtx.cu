#include <cuda.h>

#include <nvToolsExt.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>

#include "cuda_utils.h"

namespace profiling {

nvtxEventAttributes_t init_event (const std::string& str)
{
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF880000;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = str.c_str ();

    return eventAttrib;
}

void add_marker (const std::string& str)
{
    nvtxEventAttributes_t eventAttrib = init_event (str);
    nvtxMarkEx (&eventAttrib);
}

void push_range (const std::string& str)
{
    nvtxEventAttributes_t eventAttrib = init_event (str);
    nvtxRangePushEx (&eventAttrib);
}

void pop_range ()
{
    nvtxRangePop();
}

} // namespace profiling

// Based on https://github.com/thrust/thrust/wiki/Quick-Start-Guide
int main ()
{
    // Add marker at start
    profiling::add_marker ("main() start");

    // Start range profiling
    profiling::push_range ("main() execution");

    // initialize all ten integers of a device_vector to 1
    profiling::add_marker ("initialize device vector");
    thrust::device_vector<int> D (10, 1);

    // set the first seven elements of a vector to 9
    profiling::add_marker ("modify device vector");
    thrust::fill (D.begin (), D.begin () + 7, 9);

    // initialize a host_vector with the first five elements of D
    profiling::add_marker ("initialize host vector");
    thrust::host_vector<int> H (D.begin (), D.begin () + 5);

    // set the elements of H to 0, 1, 2, 3, ...
    profiling::add_marker ("modify host vector");
    thrust::sequence (H.begin (), H.end ());

    // copy all of H back to the beginning of D
    profiling::add_marker ("copy host to device");
    thrust::copy (H.begin (), H.end (), D.begin ());

    // End range profiling
    profiling::pop_range ();

    return 0;
}

