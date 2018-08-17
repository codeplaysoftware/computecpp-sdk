/***************************************************************************
 *
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  double_buf.hpp
 *
 *  Description:
 *    Provides a double-buffer class.
 *
 **************************************************************************/

#pragma once

#include <utility>


// Double-buffers any kind of value.
template <typename T>
class DoubleBuf {
 public:
  template <typename... U>
  DoubleBuf(U&&... vals)
      : m_read_a_and_not_b(true),
        m_a(std::forward<U>(vals)...),
        m_b(std::forward<U>(vals)...) {}

  void swap() { m_read_a_and_not_b = !m_read_a_and_not_b; }

  T& read() {
    if (m_read_a_and_not_b) {
      return m_a;
    } else {
      return m_b;
    }
  }

  T& write() {
    if (m_read_a_and_not_b) {
      return m_b;
    } else {
      return m_a;
    }
  }

 private:
  DoubleBuf() {}

  // true -> read a;  false -> read b;
  // true -> write b; false -> write a;
  bool m_read_a_and_not_b;
  T m_a;
  T m_b;
};
