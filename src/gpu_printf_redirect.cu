// See: http://stackoverflow.com/questions/21238303/redirecting-cuda-printf-to-a-c-stream

#include <cuda.h>
#include <unistd.h> // dup

#include <cstdio>
#include <iostream>
#include <sstream> // stringstream
#include <fstream> // ofstream

#include <boost/optional.hpp>

#include "cuda_utils.h"

char* output_file = "printf_redirect.log";

class RedirectPrint {

public:
    /**
     * @brief RedirectPrint constructor.
     * @param o stream.
     */
    RedirectPrint(std::ostream& o)
        : backup_stdout_ (0),
          print_stream_ (),
          log_stream_ (o),
          std_stream_ (0x0)
    {
        memset (buf_, '\0', sizeof (buf_));
    }

    ~RedirectPrint()
    {
        flush ();
        reset ();
    }

    /**
     * @brief redirect printf to the previously given stream.
     */
    void redirect (FILE* f = stdout, std::ostream& os = std::cout,
                   bool std_stio_sync = false)
    {
        std::ios_base::sync_with_stdio (std_stio_sync);

        os << "[std::cout] Before redirection" << std::endl;
        std::printf ("[printf] Before redirection\n");

        print_stream_ = f;
        std_stream_ = &os;

        // Redirect std::cout to log stream
        backup_cout = std_stream_->rdbuf ();
        std_stream_->rdbuf (log_stream_.rdbuf ());

        // Redirect stdout to log stream
        backup_stdout_ = dup (fileno (f));

        // Flushing will actually send data to /dev/null, we will need to get
        // the data from the buffer buf_ directly.
        std::freopen ("/dev/null", "w", f);
        std::setbuf (f, buf_);

        std::printf ("[printf] After redirection\n");
        (*std_stream_) << "[std::cout] After redirection" << std::endl;
    }

    /**
     * @brief flush printf buffer to the stream.
     */
    void flush ()
    {
        if (print_stream_)
        {
            // Print buffer to stream
            log_stream_ << buf_;
            fflush (print_stream_.get ());
            std_stream_->flush ();
        }
    }

    /**
     * @brief reset the redirection.
     */
    void reset ()
    {
        if (print_stream_)
        {
            FILE* f = print_stream_.get ();

            // Redirect back to initial stdout
            setbuf (f, NULL);
            fclose (f);
            FILE *fbk = fdopen (backup_stdout_, "w");
            fclose (f);
            *f = *fbk;

            // Reset printf stream
            print_stream_.reset ();

            // Reset std stream
            std_stream_->rdbuf (backup_cout);
        }

        // Toggle synchronization with cstdio streams on
        std::ios_base::sync_with_stdio (true);
    }

    const std::ostream& stream ()
    {
        return log_stream_;
    }

private:
    char                   buf_[1024];
    int                    backup_stdout_;
    boost::optional<FILE*> print_stream_;
    std::ostream&          log_stream_;
    std::stringstream      ss_;

    std::ostream*   std_stream_;
    std::streambuf* backup_cout;
};


__global__ void printf_redirect(int* src, int* res)
{
    res[threadIdx.x] = threadIdx.x;
    printf("  [printf] %i: Hello World!\n", res[threadIdx.x]);
}

int main()
{
    using namespace std;

    const uint N = 2;

    // Note: dummy arrays are not actually used, but this should prevent the
    //       compiler from discarding the printf in the kernel.

    int *d_A, *d_B, *h_A, *h_B;
    size_t size = N * sizeof (int);
    CUDA_SAFE_CALL (cudaMalloc (&d_A, size));
    CUDA_SAFE_CALL (cudaMalloc (&d_B, size));
    h_A = (int*) malloc (size);
    h_B = (int*) malloc (size);
    CUDA_SAFE_CALL (cudaMemcpy (d_A, h_A, size, cudaMemcpyHostToDevice));

    std::cout << "[std::cout] start" << std::endl;
    printf ("[printf] start\n");

    /// REGULAR PRINT
    // Print to regular stdout
    std::cout << "[std::cout] Output to stdout:" << std::endl;
    printf_redirect<<<1,1>>> (d_A, d_B);
    CUDA_SAFE_CALL (cudaDeviceSynchronize ());

    /// REDIRECTION TO STRINGSTREAM
    std::stringstream ss;
    // Redirect stdout/std::cout to buffer/stream
    RedirectPrint rp (ss);
    rp.redirect (stdout, std::cout);
    std::cout << "[std::cout] Redirected output:" << std::endl;

    printf_redirect<<<1,N>>> (d_A, d_B);
    CUDA_SAFE_CALL (cudaDeviceSynchronize ());

    // Add CUDA buffer to a stringstream
    rp.flush ();

    // Write stringstream to file
    std::ofstream outFile;
    outFile.open (output_file);
    outFile << ss.str ();
    outFile.close ();

    /// RESET REDIRECTION
    // Redirect back to initial stdout/std::cout
    rp.reset ();

    std::printf ("[printf] After redirection reset\n");
    std::cout << "[std::cout] After redirection reset" << std::endl;

    CUDA_SAFE_CALL (cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL (cudaFree(d_A));
    CUDA_SAFE_CALL (cudaFree(d_B));
    free (h_A);
    free (h_B);
}
