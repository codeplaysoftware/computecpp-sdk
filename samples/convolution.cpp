#include <sycl/sycl.hpp>

constexpr auto elems_per_thread = 8;

template <typename T>
class Convolve1D {
  sycl::accessor<T> f_;
  sycl::accessor<T> g_;
  sycl::accessor<T> out_;
  sycl::local_accessor<T> f_a_;
  sycl::local_accessor<T> f_b_;
  sycl::local_accessor<T> g_local_;

  void process_chunk(T* chunk, T* out, sycl::nd_item<> i) const {
    T accums[elems_per_thread] = {0};
    for (int g_index = 0; g_index < g_.size(); g_index++) {
      for (int k = 0; k < elems_per_thread; k++) {
        // k variable avoids bank conflicts, g_index is a broadcast
        accums[k] +=
            chunk[g_index + k * i.get_local_range(0) + i.get_local_id(0)] *
            g_local_[g_index];
      }
    }
    for (int k = 0; k < elems_per_thread; k++) {
      out[i.get_local_id(0) + i.get_local_range(0) * k] = accums[k];
    }
  }

 public:
  Convolve1D(sycl::accessor<T> f, sycl::accessor<T> g, sycl::accessor<T> out,
             unsigned int wg_size, sycl::handler& h)
      : f_(f),
        g_(g),
        out_(out),
        f_a_(sycl::range{wg_size * elems_per_thread} + g.get_range(), h),
        f_b_(sycl::range{wg_size * elems_per_thread} + g.get_range(), h),
        g_local_(g.get_range(), h) {}

  void operator()(sycl::nd_item<> i) const {
    // Offset: work-group size x number of elements per work-item x the global
    // work-group identity x two chunks per work-group
    const auto offset =
        i.get_group(0) * 2 * i.get_local_range(0) * elems_per_thread;
    auto g_ev = i.async_work_group_copy(g_local_.get_pointer(),
                                        g_.get_pointer(), g_.size());
    auto f_a_ev = i.async_work_group_copy(
        f_a_.get_pointer(), f_.get_pointer() + offset, f_a_.size());
    auto f_b_ev = i.async_work_group_copy(
        f_b_.get_pointer(),
        f_.get_pointer() + offset + i.get_local_range(0) * elems_per_thread,
        f_b_.size());
    i.wait_for(g_ev, f_a_ev);
    process_chunk(f_a_.get_pointer(), out_.get_pointer() + offset, i);
    i.wait_for(f_b_ev);
    process_chunk(
        f_b_.get_pointer(),
        out_.get_pointer() + offset + i.get_local_range(0) * elems_per_thread,
        i);
  }
};

class Init;

int main() {
  constexpr auto n_elems = 16777216;
  constexpr auto g_elems = 16;
  constexpr auto wg_size = 32;

  sycl::queue q;
  // The last element will not be accessed (only elements up to f + g - 1)
  sycl::buffer<float> lhs(n_elems + g_elems);
  sycl::buffer<float> rhs(g_elems);
  sycl::buffer<float> out(n_elems);
  auto init_buffer = [&](sycl::buffer<float> b) {
    q.submit([&](sycl::handler& h) {
      auto a = b.get_access(h);
      h.parallel_for<Init>(b.get_range(), [=](sycl::item<> i) {
        a[i.get_linear_id()] = i.get_linear_id() % 256;
      });
    });
  };
  init_buffer(lhs);
  init_buffer(rhs);
  q.submit([&](sycl::handler& h) {
    sycl::accessor<float> l(lhs, h);
    sycl::accessor<float> r(rhs, h);
    sycl::accessor<float> o(out, h);
    // each WI double-buffers, so 2x elems/thread per output
    h.parallel_for(sycl::nd_range{o.get_range() / (elems_per_thread * 2),
                                  sycl::range{wg_size}},
                   Convolve1D<float>{l, r, o, wg_size, h});
  });
  q.wait_and_throw();

  return 0;
}
