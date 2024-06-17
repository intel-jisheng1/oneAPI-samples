#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>

int main(int argc, char **argv) {
    // std::ifstream hardwareData("/nfs/site/disks/swuser_work_jisheng1/hsd/14022200110_reduce/products/regtest/hld/sycl/oneapi_code_samples/fft2d_shim_kernel_hardware/default/jisheng1_agilex_n6001/ofs_n6001/seed_2/compile/ReferenceDesigns/fft2d_shim_kernel_hardware/build/shim_data2_0.txt"); 
    std::ifstream hardwareData("/nfs/site/disks/swuser_work_jisheng1/hsd/14022200110_reduce/products/regtest/hld/sycl/oneapi_code_samples/fft2d_shim_kernel_hoist/default/jisheng1_agilex_n6001/ofs_n6001/seed_1/compile/run/ReferenceDesigns/fft2d_shim_kernel_hoist/build/shim_fullfft_data_0.txt"); 
    std::ifstream emuData("/nfs/site/disks/swuser_work_jisheng1/hsd/14022200110_reduce/external/oneapi-src/oneAPI-samples/DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/fft2d_shim_kernel_hoist/build/shim_fullfft_data_0.txt"); 
    
    if (!hardwareData.is_open() || !emuData.is_open()) { 
        std::cerr << "Failed to open file for writing.\n"; 
        exit(1);
    } 

    // Reading the array elements from the file 
    float hardware, emu;
    int num_wrong= 0;
    for (int i = 0; i < 2*(1<<10)*(1<<10); ++i) { 
        hardwareData >> hardware; 
        emuData >> emu; 

        if( std::abs(hardware - emu) >= 1e-3) {
            std::cout<< "wrong results, expecting " << emu << " but got " << hardware << " at index " << i<< std::endl;
            num_wrong++;

            if(num_wrong > 10) break;
        }
    } 
  
    // Closing the file 
    hardwareData.close(); 
    emuData.close();
}