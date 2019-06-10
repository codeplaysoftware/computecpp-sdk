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
 *  images.cpp
 *
 *  Description:
 *    Sample code that uses SYCL images.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

class mod_image;

int main() {
  /* We define and populate arrays for the input and output data. */
  float4 src[256];
  float4 dest[256];
  for (int i = 0; i < 256; i++) {
    src[i] = float4(1.0f, 2.0f, 3.0f, 4.0f);
    dest[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  try {
    /* We create images for the input and output data we defined above. In SYCL
     * images are created from a host pointer like buffers, however, instead of
     * specifying a type you provide the channel order and channel type when
     * constructing them. */
    image<2> srcImage(src, image_channel_order::rgba, image_channel_type::fp32,
                      range<2>(16, 16));
    image<2> destImage(dest, image_channel_order::rgba,
                       image_channel_type::fp32, range<2>(16, 16));

    /* We can retrieve the size of the image by calling get_size(). */
    std::cout << "Image size: " << srcImage.get_size() << std::endl;

    queue myQueue;

    myQueue.submit([&](handler& cgh) {
      /* We can request access to an image by constructing an accessor with
       * the target access::target::image. When accessing images the accessor
       * element type is used to specify how the image should be read from or
       * written to. It can be either int4, uint4 or float4. */
      accessor<float4, 2, access::mode::read, access::target::image> inPtr(
          srcImage, cgh);

      /* We can also use the get_access() member function that is provided on
       * the image as a shortcut. You also have to specify the accessor element
       * type here. */
      auto outPtr = destImage.get_access<float4, access::mode::write>(cgh);

      /* Samplers are used to specify the way in which the coordinates map to
       * a particular pixel in the image. Here, we specify that the sampler
       * will not use normalised co-ordinates, that addresses outside the
       * image bounds should clamp to the edge of the image and that no
       * floating-point co-ordinates should take the nearest pixel's data,
       * rather that applying (for example) a linear filter.
       * SYCL's images and samplers map very well to the underlying OpenCL
       * constructs. OpenCL documentation has a more full discussion of
       * how images and samplers work. */
      sampler smpl(coordinate_normalization_mode::unnormalized,
                   addressing_mode::clamp, filtering_mode::nearest);

      cgh.parallel_for<mod_image>(range<2>(16, 16), [=](item<2> item) {
        /* SYCL vectors are used to specify the location of the image to
         * access. */
        auto coords = int2(item[0], item[1]);

        /* In SYCL images are read using a subscript operator to provide the
         * coordinates, with an optional function call operator preceding it
         * to provide a sampler. */
        float4 pixel = inPtr.read(coords, smpl);

        pixel *= 10.0f;

        /* Images are written to in a similar fashion, only without a
         * sampler. */
        outPtr.write(coords, pixel);
      });
    });

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 2;
  }

  cl::sycl::float4 expected = {10.f, 20.f, 30.f, 40.f};
  if (cl::sycl::all(cl::sycl::isequal(dest[0], expected))) {
    std::cout << "The output image is as expected." << std::endl;
    return 0;
  } else {
    std::cout << "The output image is incorrect." << std::endl;
    return 1;
  }
}
