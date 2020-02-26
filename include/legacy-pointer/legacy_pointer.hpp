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
 *  legacy_pointer.hpp
 *
 *  Description:
 *    Interface for SYCL buffers to behave as a non-deferrenciable pointer
 *
 * Authors:
 *
 *    Ruyman Reyes   Codeplay Software Ltd.
 *    Mehdi Goli     Codeplay Software Ltd.
 *
 **************************************************************************/

#include <CL/sycl.hpp>
#include <iostream>

#include <unordered_map>

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
  static const unsigned long ADDRESS_BITS = sizeof(void*) * 8;
  static const unsigned long BUFFER_ID_BITSIZE = 16u;
  static const unsigned long MAX_NUMBER_BUFFERS =
      (1UL << BUFFER_ID_BITSIZE) - 1;
  static const unsigned long MAX_OFFSET =
      (1UL << (ADDRESS_BITS - BUFFER_ID_BITSIZE)) - 1;

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
    base_ptr_t m_contents;

    /** Conversions from legacy_pointer_t to
     * the void * should just reinterpret_cast the integer
     * number
     */
    operator void*() const { return reinterpret_cast<void*>(m_contents); }

    /**
     * Convert back to the integer number.
     */
    operator base_ptr_t() const { return m_contents; }

    /**
     * Converts a void * into a legacy pointer structure.
     * Note that this will only work if the void * was
     * already a legacy_pointer_t, but we have no way of
     * checking
     */
    legacy_pointer_t(void* ptr)
        : m_contents(reinterpret_cast<base_ptr_t>(ptr)){};

    /**
     * Creates a legacy_pointer_t from the given integer
     * number
     */
    legacy_pointer_t(base_ptr_t u) : m_contents(u){};
  };

  /* Whether if a pointer is null or not.
   *
   * A pointer is nullptr if the buffer id is 0,
   * i.e the first BUFFER_ID_BITSIZE are zero
   */
  static inline bool is_nullptr(legacy_pointer_t ptr) {
    return ((MAX_OFFSET & ptr) == ptr);
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
  inline buffer_id get_buffer_id(legacy_pointer_t ptr) const {
    return ptr >> (ADDRESS_BITS - BUFFER_ID_BITSIZE);
  }

  /*
   * get_buffer_offset
   */
  inline off_t get_offset(legacy_pointer_t ptr) const {
    return ptr & MAX_OFFSET;
    ;
  }

  /**
   * Constructs the PointerMapper structure.
   */
  PointerMapper() : m_pointerMap{} {};

  /**
   * PointerMapper cannot be copied or moved
   */
  PointerMapper(const PointerMapper&) = delete;

  /**
   *	empty the pointer list
   */
  inline void clear() { m_pointerMap.clear(); }

  /* generate_id
   * Generates a unique id for a buffer.
   */
  buffer_id generate_id() {
    static buffer_id counter = 0;
    return ++counter;
  }

  /* add_pointer.
   * Adds a pointer to the map and returns the fake pointer id.
   * This will be the bufferId on the most significant bytes and 0 elsewhere.
   */
  legacy_pointer_t add_pointer(buffer_t&& b) {
    auto nextNumber = m_pointerMap.size();
    buffer_id bId = generate_id();
    m_pointerMap.emplace(bId, b);
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
    auto it = m_pointerMap.find(bId);
    if (it != m_pointerMap.end()) {
      return it->second;
    }

    std::cerr << "No sycl buffer has been found. Make sure that you have "
                 "allocated memory for your buffer by calling malloc function."
              << std::endl;
    std::abort();
  }

  /* remove_pointer.
   * Removes the given pointer from the map.
   */
  void remove_pointer(void* ptr) {
    buffer_id bId = this->get_buffer_id(ptr);
    m_pointerMap.erase(bId);
  }

  /* count.
   * Return the number of active pointers (i.e, pointers that
   * have been malloc but not freed).
   */
  size_t count() const { return m_pointerMap.size(); }

 private:
  /* Maps the buffer id numbers to the actual buffer
   * instances.
   */
  std::unordered_map<buffer_id, buffer_t> m_pointerMap;
};

/**
 * Singleton interface to the pointer mapper to implement
 * the generic malloc/free C interface without extra
 * parameters.
 */
inline PointerMapper& getPointerMapper() {
  static PointerMapper thePointerMapper;
  return thePointerMapper;
}

/**
 * Malloc-like interface to the pointer-mapper.
 * Given a size, creates a byte-typed buffer and returns a
 * fake pointer to keep track of it.
 */
inline void* malloc(size_t size) {
  // Create a generic buffer of the given size
  auto thePointer = getPointerMapper().add_pointer(
      PointerMapper::buffer_t(cl::sycl::range<1>{size}));
  // Store the buffer on the global list
  return static_cast<void*>(thePointer);
}

/**
 * Free-like interface to the pointer mapper.
 * Given a fake-pointer created with the legacy-pointer malloc,
 * destroys the buffer and remove it from the list.
 */
inline void free(void* ptr) { getPointerMapper().remove_pointer(ptr); }

/**
 *clear the pointer list
 */
inline void clear() { getPointerMapper().clear(); }

}  // namespace legacy
}  // namespace codeplay
