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
 *  codeplay_demo.hpp
 *
 *  Description:
 *    Provides logo drawing functionality for the demos.
 *
 **************************************************************************/

#pragma once

#include <cinder/CameraUi.h>
#include <cinder/TriMesh.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>


// Adds logo drawing functionality.
class CodeplayDemoApp : public ci::app::App {
 protected:
  ci::gl::Texture2dRef m_codeplay_tex =
      ci::gl::Texture2d::create(loadImage(loadAsset("logo.png")));

  void draw_codeplay_logo() {
    float tex_ratio =
        float(m_codeplay_tex->getWidth()) / float(m_codeplay_tex->getHeight());

    float w = getWindow()->getWidth();
    float h = getWindow()->getHeight();
    ci::gl::setMatricesWindow(w, h);

    ci::Rectf logo_rect(w - 0.05f * h - 0.35f * h,
                        0.95f * h - 0.35f * h / tex_ratio, w - 0.05f * h,
                        0.95f * h);

    ci::gl::color(ci::Color(1.0f, 1.0f, 1.0f));
    ci::gl::draw(m_codeplay_tex, logo_rect);
  }
};
