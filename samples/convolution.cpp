#include <iostream>
#include <sycl/sycl.hpp>

#include <utility>

constexpr auto elems_per_thread = 4;
constexpr auto double_buffer_iterations = 64;

template <typename T>
class Convolve1D {
  sycl::accessor<T> f_;
  sycl::accessor<T> g_;
  sycl::accessor<T> out_;
  sycl::local_accessor<T> f_a_;
  sycl::local_accessor<T> f_b_;
  sycl::local_accessor<T> out_a_;
  sycl::local_accessor<T> out_b_;
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
        out_a_(sycl::range{wg_size * elems_per_thread}, h),
        out_b_(sycl::range{wg_size * elems_per_thread}, h),
        g_local_(g.get_range(), h) {}

  void operator()(sycl::nd_item<> i) const {
    using DataEvent = std::tuple<sycl::local_ptr<T>, sycl::local_ptr<T>,
                                 sycl::device_event, sycl::device_event>;
    // Offset: work-group size x number of elements per work-item x the global
    // work-group identity
    const auto wg_elements = i.get_local_range(0) * elems_per_thread;
    const auto offset = i.get_group(0) * wg_elements * double_buffer_iterations;
    auto g_ev = i.async_work_group_copy(g_local_.get_pointer(),
                                        g_.get_pointer(), g_.size());
    auto f_a_ev = i.async_work_group_copy(
        f_a_.get_pointer(), f_.get_pointer() + offset, f_a_.size());
    auto f_b_ev = i.async_work_group_copy(
        f_b_.get_pointer(), f_.get_pointer() + offset + wg_elements,
        f_b_.size());
    // The repeated f events will be replaced with output events in the loop
    DataEvent active = {f_a_.get_pointer(), out_a_.get_pointer(), f_a_ev,
                        f_a_ev};
    DataEvent inactive = {f_b_.get_pointer(), out_b_.get_pointer(), f_b_ev,
                          f_b_ev};
    i.wait_for(g_ev);
    for (int j = 0; j < double_buffer_iterations - 1; j++) {
      auto& [in, out, ev, out_ev] = active;
      auto& [inactive_ptr, b, inactive_ev, c] = inactive;
      inactive_ev = i.async_work_group_copy(
          inactive_ptr, f_.get_pointer() + offset + (j+1) * wg_elements,
          wg_elements + g_.get_size());
      i.wait_for(ev, out_ev);
      process_chunk(in, out, i);
      out_ev = i.async_work_group_copy(
          out_.get_pointer() + offset + j * wg_elements, out, wg_elements);
      std::swap(active, inactive);
    }
    auto& [in, out, ev, out_ev] = active;
    i.wait_for(ev);
    process_chunk(in, out, i);
    out_ev = i.async_work_group_copy(out_.get_pointer() + offset + (double_buffer_iterations - 1) * wg_elements, out, wg_elements);
    //i.wait_for(out_ev, std::get<3>(inactive));
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
    h.parallel_for(sycl::nd_range{o.get_range() / (elems_per_thread *
                                                   double_buffer_iterations),
                                  sycl::range{wg_size}},
                   Convolve1D<float>{l, r, o, wg_size, h});
  });
  q.wait_and_throw();

  return 0;
}
