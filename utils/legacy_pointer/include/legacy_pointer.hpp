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
    const unsigned ADDRESS_BITS = sizeof(void*) * 8;
    const unsigned BUFFER_ID_BITSIZE = 16u;
    const unsigned MAX_NUMBER_BUFFERS = 2<<BUFFER_ID_BITSIZE;
    const unsigned MAX_OFFSET = 2<<(ADDRESS_BITS-BUFFER_ID_BITSIZE);

    using base_ptr_t = intptr_t;

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

      operator void*() {
        return reinterpret_cast<void*>(_contents);
      }

      operator base_ptr_t() {
        return _contents;
      }

      legacy_pointer_t(void* ptr)
        : _contents(reinterpret_cast<base_ptr_t>(ptr)) { };

      legacy_pointer_t(base_ptr_t u)
        : _contents(u) { };
    };

    /* Whether if a pointer is null or not.
     *
     * A pointer is nullptr if the buffer id is 0,
     * i.e the first BUFFER_ID_BITSIZE are zero
     */
    static inline bool is_nullptr(legacy_pointer_t ptr) {
      return ( (0x0000FFFFFFFFFFFFlu & ptr) == ptr);
    }

    /* Base nullptr
     */
    const legacy_pointer_t null_legacy_ptr = nullptr;

    /* basic type for all buffers
     */
    using buffer_t = cl::sycl::buffer<uint8_t, 1>;

    /* id of a buffer in the map
     */
    using buffer_id = short;

    /* get_buffer_id
     */
    buffer_id get_buffer_id(legacy_pointer_t ptr) const {
      auto theId = (ptr & 0xFFFF000000000000lu)>>(ADDRESS_BITS-BUFFER_ID_BITSIZE);
      return theId;
    }

    PointerMapper()
      : __pointer_list{} {};

    PointerMapper(const PointerMapper&) = delete;

    /* generate_id
     * Generates a unique id for a buffer.
     */
    buffer_id generate_id() const {
      static buffer_id bId = 0;
      // Note: When overflow, will be at 0, which is invalid ptr
      return ++bId;
    }
  
    /* add_pointer
     */
    legacy_pointer_t add_pointer(buffer_t&& b) {
      auto nextNumber = __pointer_list.size();
      buffer_id bId = generate_id();
      __pointer_list.emplace(bId, b);
      if (nextNumber > MAX_NUMBER_BUFFERS) {
        return null_legacy_ptr;
      }
      base_ptr_t retVal = bId;
      retVal <<= (ADDRESS_BITS-BUFFER_ID_BITSIZE);
      return retVal;
    }

    /* get_buffer
     */
    buffer_t get_buffer(buffer_id bId) const {
      buffer_t retVal = __pointer_list.at(bId);
      return retVal;
    }

    /* remove_pointer
     */
    void remove_pointer(void * ptr) {
      buffer_id bId = this->get_buffer_id(ptr);
      __pointer_list.erase(bId);
    }

    /* count.
     * Return the number of active pointers (i.e, pointers that 
     * have been malloc but not freed).
     *
     */
    size_t count() const {
      return __pointer_list.size();
    }

    private:
     /* Collection of pointers
       */
     std::map<buffer_id, buffer_t> __pointer_list;
  };

  inline PointerMapper& getPointerMapper() {
    static PointerMapper thePointerMapper;
    return thePointerMapper;
  }

  void * malloc(size_t size) {
    // Create a generic buffer of the given size
    auto thePointer = getPointerMapper().add_pointer(PointerMapper::buffer_t(cl::sycl::range<1>{size}));
    // Store the buffer on the global list
    return static_cast<void *>(thePointer);
  }

  void free(void * ptr) { 
    getPointerMapper().remove_pointer(ptr);
  }

}  // legacy
}  // codeplay




