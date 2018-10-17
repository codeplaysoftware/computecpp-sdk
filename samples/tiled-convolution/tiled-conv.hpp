#include "common.hpp"
// template<int col_per_thread> compute

// the tiled based convolution functor
template <typename read_accessor_t, typename write_accessor_t,
          typename mat_size_t>
class conv {
 private:
  using data_t = typename write_accessor_t::value_type;
  const read_accessor_t fil_acc;
  const read_accessor_t in_acc;
  write_accessor_t out_acc;
  const mat_size_t total_size;
  const mat_size_t mat_size;
  const int m_start_offset;
  const int n_start_offset;
  const mat_size_t num_group;
  static constexpr int row_per_work_item =
      opencl_configuration_t::row_per_work_item;
  static constexpr int col_per_work_item =
      opencl_configuration_t::cache_line - 2;
  static constexpr int fil_size_m = 3;
  static constexpr int fil_size_n = 3;

 public:
  // constructing the functor
  conv(const read_accessor_t fil_acc_, const read_accessor_t in_acc_,
       write_accessor_t out_acc_, const mat_size_t total_size_,
       const mat_size_t mat_size_, const int m_start_offset_,
       const int n_start_offset_, const mat_size_t num_group_)
      : fil_acc(fil_acc_),
        in_acc(in_acc_),
        out_acc(out_acc_),
        total_size(total_size_),
        mat_size(mat_size_),
        m_start_offset(m_start_offset_),
        n_start_offset(n_start_offset_),
        num_group(num_group_) {}
  void inline operator()(cl::sycl::nd_item<1> item_id) {
    const int total_threads_m = (item_id.get_local_range()[0]) * num_group.m;
    const int group_m = item_id.get_group(0) / num_group.n;
    const int group_n = item_id.get_group(0) - group_m * num_group.n;
    // item_id.get_group(0) % num_group.n;
    const int work_item = (group_m * (item_id.get_local_range()[0])) +
                          item_id.get_local_id(0);  // item_id.get_global(0);
    // the LWM tile of 2 * 6 for output that reads 2*8 inout
    data_t private_result[row_per_work_item][col_per_work_item] = {};
    data_t private_in[row_per_work_item + fil_size_m - 1]
                     [opencl_configuration_t::cache_line];
    // this is used to keep the filter in LWM to prevent the input zero level
    // cache to be flushed befor being used by all threads
    data_t filter[fil_size_m][fil_size_n];

// set filter to LWM to prevent the level zero cache to be modified
#pragma nounroll
    for (int p_m = 0; p_m < fil_size_m; p_m++) {
#pragma nounroll
      for (int p_n = 0; p_n < fil_size_n; p_n++) {
        filter[p_m][p_n] =
            fil_acc[p_m][p_n] / static_cast<data_t>(fil_size_m * fil_size_n);
      }
    }
    item_id.mem_fence(cl::sycl::access::fence_space::global_and_local);
    // the outer for-loop which dedicate 2 consecutive output rows per
    // thread. We dedicate two consecutive rows because we can only read 4
    // rows to calculate the 2 output rows.
    const int index_n_offset = group_n * opencl_configuration_t::col_per_thread;
    const int loop_n_check = std::min(
        mat_size.n, index_n_offset + opencl_configuration_t::col_per_thread);
    // loop over M
#pragma nounroll
    for (int index_m = (work_item * row_per_work_item); index_m < mat_size.m;
         index_m += total_threads_m * row_per_work_item) {
      const int row = index_m;
      const int in_row = row + m_start_offset;
      const bool is_external_block_m =
          (mat_size.m - index_m) < total_threads_m * row_per_work_item;
// loop over N
#pragma nounroll
      for (int index_n = index_n_offset; index_n < loop_n_check;
           index_n += col_per_work_item) {  //
        int m, f_m;
        const int base_col = index_n;
        const int in_base_col = base_col + n_start_offset;
        const bool is_external_block_n =
            (loop_n_check - index_n) < col_per_work_item;
        // loop over m to load the tile
#pragma nounroll
        for (f_m = 0, m = -(fil_size_m >> 1);
             f_m < row_per_work_item + fil_size_m - 1; m++, f_m++) {
          int in_id_m = (in_row + m >= 0) ? in_row + m : 0;
          in_id_m = (in_id_m < total_size.m) ? in_id_m : total_size.m - 1;
          int p_n, g_n;
#pragma unroll
          // loop over n to load the tile
          for (p_n = 0, g_n = -(fil_size_n >> 1);
               p_n < opencl_configuration_t::cache_line; p_n++, g_n++) {  //
            int in_id_n = (in_base_col + g_n >= 0) ? in_base_col + g_n : 0;
            in_id_n = (in_id_n < total_size.n) ? in_id_n : total_size.n - 1;
            private_in[f_m][p_n] = in_acc[in_id_m][in_id_n];
          }
        }
        item_id.mem_fence(cl::sycl::access::fence_space::global_and_local);
        // loop over the image column
        // loop over the filter column
#pragma unroll 6
        for (int private_n = 0; private_n < col_per_work_item;
             private_n++) {  //
#pragma unroll
          for (int f_n = 0; f_n < fil_size_n; f_n++) {
#pragma unroll
            for (int in_id_m = 0; in_id_m < fil_size_m + row_per_work_item - 1;
                 in_id_m++) {
              // compute both rows output for the read input element. This if
              // statement is flattened at compile time as the for loop is
              // static
              auto input = private_in[in_id_m][private_n + f_n];
              if (row_per_work_item == 2) {
                if (in_id_m == 0) {
                  private_result[0][private_n] +=
                      (input * filter[in_id_m][f_n]);
                } else if (in_id_m != 0 &&
                           in_id_m != fil_size_m + row_per_work_item - 2) {
                  private_result[0][private_n] +=
                      (input * filter[in_id_m][f_n]);
                  private_result[1][private_n] +=
                      (input * filter[in_id_m - 1][f_n]);
                } else if (in_id_m == fil_size_m + row_per_work_item - 2) {
                  private_result[1][private_n] +=
                      (input * filter[in_id_m - 1][f_n]);
                }
              } else if (row_per_work_item == 1) {
                private_result[0][private_n] += (input * filter[in_id_m][f_n]);
              }
            }
          }
        }
        // flush the partial tile of LWM to the global output memory
        // check to see if it is internal block or external block
        if (is_external_block_m == true && is_external_block_n == true) {
          write_back<true, true>(private_result, out_acc, row, base_col);
        } else if (is_external_block_m == true &&
                   is_external_block_n == false) {
          write_back<true, false>(private_result, out_acc, row, base_col);
        } else if (is_external_block_m == false &&
                   is_external_block_n == true) {
          write_back<false, true>(private_result, out_acc, row, base_col);
        } else if (is_external_block_m == false &&
                   is_external_block_n == false) {
          write_back<false, false>(private_result, out_acc, row, base_col);
        }
      }
    }
  }
  // flush the partial tile of LWM to the global output memory
  template <bool is_external_block_m, bool is_external_block_n, typename data_t,
            typename acc>
  void write_back(
      data_t (&private_result)[row_per_work_item][col_per_work_item],
      acc out_acc, int row, int col) {
#pragma unroll
    for (int p_m = 0; p_m < row_per_work_item; p_m++) {
#pragma unroll 6
      for (int p_n = 0; p_n < col_per_work_item; p_n++) {
        if (do_check<is_external_block_m>((row + p_m < mat_size.m)) &&
            do_check<is_external_block_n>((col + p_n < mat_size.n)))
          if ((row + p_m < mat_size.m) && (col + p_n < mat_size.n)) {
            out_acc[row + p_m][col + p_n] = private_result[p_m][p_n];
          }
        private_result[p_m][p_n] = data_t(0);
      }
    }
  }
};

template <typename kernel_t, typename read_buff_t, typename write_buff_t,
          typename mat_size_t>
void inline tiled_cov(
    cl::sycl::queue& sycl_queue, cl::sycl::program& sycl_program,
    read_buff_t in_buff, read_buff_t fill_buff, write_buff_t out_buff,
    mat_size_t out_range_size, mat_size_t in_range_size,
    mat_size_t fil_range_size, int i, std::vector<cl::sycl::event>& events,
    std::vector<std::chrono::time_point<std::chrono::system_clock>>& starts,
    const bool clamped_edge_m, const bool clamped_edge_n) {
  // execute the tile
  starts[i] = std::chrono::system_clock::now();
  events[i] = sycl_queue.submit([&](cl::sycl::handler& cgh) {
    // this must be constant buffer
    auto in_acc =
        in_buff.template get_access<cl::sycl::access::mode::read,
                                    cl::sycl::access::target::global_buffer>(
            cgh);
    auto fill_acc =
        fill_buff.template get_access<cl::sycl::access::mode::read,
                                      cl::sycl::access::target::global_buffer>(
            cgh);
    auto out_acc =
        out_buff.template get_access<cl::sycl::access::mode::write,
                                     cl::sycl::access::target::global_buffer>(
            cgh);

    // getting the maximum work group size
    int work_group_size =
        sycl_queue.get_device()
            .get_info<cl::sycl::info::device::max_work_group_size>();
    // building the best number of global thread
    // constructing the kernel
    int local_thread =
        work_group_size / opencl_configuration_t::work_group_reduction_factor;
    int num_group_n =
        (out_range_size.n + opencl_configuration_t::col_per_thread - 1) /
        (opencl_configuration_t::col_per_thread);
    int num_group_m =
        ((out_range_size.m +
          (local_thread * opencl_configuration_t::row_per_tread) - 1) /
         (local_thread * opencl_configuration_t::row_per_tread));
    auto num_group = matrix_size_t(num_group_m, num_group_n);
    printf(
        "work_group_size %d, local_thread %d, num_group_n %d num_group_m %d "
        "global_size %d , out_range_size_m %d, out_range_size_n %d\n",
        work_group_size, local_thread, num_group_n, num_group_m,
        num_group_n * num_group_m * local_thread, out_range_size.m,
        out_range_size.n);
    const int m_start_offset = clamped_edge_m ? 0 : fil_range_size.m / 2;
    const int n_start_offset = clamped_edge_n ? 0 : fil_range_size.n / 2;
    cgh.parallel_for(
        sycl_program.template get_kernel<kernel_t>(),
        cl::sycl::nd_range<1>(
            cl::sycl::range<1>(num_group_n * num_group_m * local_thread),
            cl::sycl::range<1>(local_thread)),
        kernel_t(fill_acc, in_acc, out_acc, in_range_size, out_range_size,
                 m_start_offset, n_start_offset, num_group));
  });
}