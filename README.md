GPU-sandbox
===========

A sandbox for GPU development.


### Example list

* [Partitioning with Thrust][1] (based on [this Stack Overflow question][2])
* [Hamming distance with Thrust][3]
* [NT hash with PyCUDA][4] (based on [this Stack Overflow question][5])
* [Global memory vs Constant memory vs Texture memory][6] (based on [this Stack Overflow question][7])
* [Pointer type checking][8] (based on [this Stack Overflow question][9])
* [Element-wise product of complex vectors][10] (based on [this Stack Overflow question][11])
* [Counters for some specific kernel branches][12] (based on [this Stack Overflow question][13])
* [Passing local registers as arguments to device kernels][14]


[1]: src/gpu_partitioning.cu
[2]: http://stackoverflow.com/a/16602201/1043187
[3]: src/gpu_hamming_distance.cu
[4]: src/gpu_hash.py
[5]: http://stackoverflow.com/questions/16257776/pycuda-inconsistent-results-on-the-same-platform/16293077#16293077
[6]: src/gpu_texture.cu
[7]: http://stackoverflow.com/questions/14398416/convenience-of-2d-cuda-texture-memory-against-global-memory
[8]: src/gpu_pointer_type.cu
[9]: http://stackoverflow.com/questions/16684212/strange-behavior-when-detecting-global-memory
[10]: src/gpu_complex_multiplication.cu
[11]: http://stackoverflow.com/questions/16899237/element-by-element-vector-multiplication-cuda
[12]: src/gpu_counters.cu
[13]: http://stackoverflow.com/questions/17139688/counting-occurrences-of-specific-events-in-cuda-kernels
[14]: src/gpu_register_passing.cu
