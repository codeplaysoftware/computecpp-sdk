/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  legacy_pointer.h
 *
 *  Description:
 *    Interface for SYCL buffers to behave as a non-deferrenciable pointer
 *
 *
 **************************************************************************/

#include <CL/sycl.hpp>
#include <iostream>

namespace codeplay {
namespace legacy {

/**
 * PointerMapper
 *  Associates fake pointers with buffers.
 *
 */
class PointerMapper {
 public:
  /* pointer information definitions
   */
  const unsigned ADDRESS_BITS = sizeof(void *) * 8;
  const unsigned BUFFER_ID_BITSIZE = 16u;
  const unsigned MAX_NUMBER_BUFFERS = 2 << BUFFER_ID_BITSIZE;
  const unsigned MAX_OFFSET = 2 << (ADDRESS_BITS - BUFFER_ID_BITSIZE);

  using base_ptr_t = uintptr_t;

  /* Fake Pointers are constructed using an integer indexing plus
   * the offset:
   *
   * |== MAX_BUFFERS ==|======== MAX_OFFSET ========|
   * |   Buffer Id     |       Offset in buffer     |
   * |=================|============================|
   */
  struct legacy_pointer_t {
    /* Type for the pointers
    */
    base_ptr_t _contents;

    /** Conversions from legacy_pointer_t to
     * the void * should just reinterpret cast the integer
     * number
     */
    operator void *() { return reinterpret_cast<void *>(_contents); }

    /**
     * Convert back to the integer number.
     */
    operator base_ptr_t() { return _contents; }

    /**
     * Converts a void * into a legacy pointer structure.
     * Note that this will only work if the void * was
     * already a legacy_pointer_t, but we have no way of
     * checking
     */
    legacy_pointer_t(void *ptr)
        : _contents(reinterpret_cast<base_ptr_t>(ptr)){};

    /**
     * Creates a legacy_pointer_t from the given integer
     * number
     */
    legacy_pointer_t(base_ptr_t u) : _contents(u){};
  };

  /* Whether if a pointer is null or not.
   *
   * A pointer is nullptr if the buffer id is 0,
   * i.e the first BUFFER_ID_BITSIZE are zero
   */
  static inline bool is_nullptr(legacy_pointer_t ptr) {
    return ((0x0000FFFFFFFFFFFFlu & ptr) == ptr);
  }

  /* Base nullptr
   */
  const legacy_pointer_t null_legacy_ptr = nullptr;

  /* Data type to create buffer of byte-size elements
   */
  using buffer_data_type = uint8_t;

  /* basic type for all buffers
   */
  using buffer_t = cl::sycl::buffer<buffer_data_type, 1>;

  /* id of a buffer in the map
   */
  using buffer_id = short;

  /* get_buffer_id
   */
  buffer_id get_buffer_id(legacy_pointer_t ptr) const {
    auto theId =
        (ptr & 0xFFFF000000000000lu) >> (ADDRESS_BITS - BUFFER_ID_BITSIZE);
    return theId;
  }

  /*
   * get_buffer_offset
   */
  off_t get_offset(legacy_pointer_t ptr) const {
    auto theOffset = (ptr & 0x0000FFFFFFFFFFFFlu);
    return theOffset;
  }

  /**
   * Constructs the PointerMapper structure.
   */
  PointerMapper()
      : __pointer_list{}, rng_(std::random_device()()), uni_(1, 256){};

  /**
   * PointerMapper cannot be copied or moved
   */
  PointerMapper(const PointerMapper &) = delete;

  /**
  *	empty the pointer list
  */
  void clear() { __pointer_list.clear(); }

  /* generate_id
   * Generates a unique id for a buffer.
   */
  buffer_id generate_id() {
    // Limit the number of attemts to half the combinations
    // just to avoid an infinite loop
    int numberOfAttempts = 1ul << (BUFFER_ID_BITSIZE / 2);
    buffer_id bId;
    do {
      bId = uni_(rng_);
    } while (__pointer_list.find(bId) != __pointer_list.end() &&
             numberOfAttempts--);
    return bId;
  }

  /* add_pointer.
   * Adds a pointer to the map and returns the fake pointer id.
   * This will be the bufferId on the most significant bytes and 0 elsewhere.
   */
  legacy_pointer_t add_pointer(buffer_t &&b) {
    auto nextNumber = __pointer_list.size();
    buffer_id bId = generate_id();
    __pointer_list.emplace(bId, b);
    if (nextNumber > MAX_NUMBER_BUFFERS) {
      return null_legacy_ptr;
    }
    base_ptr_t retVal = bId;
    retVal <<= (ADDRESS_BITS - BUFFER_ID_BITSIZE);
    return retVal;
  }

  /* get_buffer.
   * Returns a buffer from the map using the buffer id
   */
  buffer_t get_buffer(buffer_id bId) const {
    //  buffer_t retVal = __pointer_list.at(bId);
    auto it = __pointer_list.find(bId);
    if (it != __pointer_list.end()) return it->second;
    std::cerr << "No sycl buffer has been found. Make sure that you have "
                 "allocated memory for your buffer by calling malloc function."
              << std::endl;
    abort();
  }

  /* remove_pointer.
   * Removes the given pointer from the map.
   */
  void remove_pointer(void *ptr) {
    buffer_id bId = this->get_buffer_id(ptr);
    __pointer_list.erase(bId);
  }

  /* count.
   * Return the number of active pointers (i.e, pointers that
   * have been malloc but not freed).
   */
  size_t count() const { return __pointer_list.size(); }

 private:
  /* Maps the buffer id numbers to the actual buffer
   * instances.
    */
  std::map<buffer_id, buffer_t> __pointer_list;

  /* Random number generator for the buffer ids
         */
  std::mt19937 rng_;

  /* Random-number engine
   */
  std::uniform_int_distribution<short> uni_;
};

/**
 * Singleton interface to the pointer mapper to implement
 * the generic malloc/free C interface without extra
 * parameters.
 */
inline PointerMapper &getPointerMapper() {
  static PointerMapper thePointerMapper;
  return thePointerMapper;
}

/**
 * Malloc-like interface to the pointer-mapper.
 * Given a size, creates a byte-typed buffer and returns a
 * fake pointer to keep track of it.
 */
void *malloc(size_t size) {
  // Create a generic buffer of the given size
  auto thePointer = getPointerMapper().add_pointer(
      PointerMapper::buffer_t(cl::sycl::range<1>{size}));
  // Store the buffer on the global list
  return static_cast<void *>(thePointer);
}

/**
 * Free-like interface to the pointer mapper.
 * Given a fake-pointer created with the legacy-pointer malloc,
 * destroys the buffer and remove it from the list.
 */
void free(void *ptr) { getPointerMapper().remove_pointer(ptr); }

/**
 *clear the pointer list
 */
void clear() { getPointerMapper().clear(); }

}  // legacy
}  // codeplay
