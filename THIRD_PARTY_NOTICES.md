# Third-Party Licenses & Notices

This project incorporates and depends on the following third-party open-source software components. Each component is governed by its respective license terms.

---

## Bundled Components in Repository

### 1. KissFFT
- **Files**: `app/src/main/cpp/kiss_fft.c`, `app/src/main/cpp/kiss_fft.h`, `app/src/main/cpp/kiss_fft_log.h`, `app/src/main/cpp/_kiss_fft_guts.h`
- **Repository**: https://github.com/mborgerding/kissfft
- **License**: BSD-3-Clause (`SPDX-License-Identifier: BSD-3-Clause`)

```text
Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

### 2. ONNX Runtime
- **Files**: `app/src/main/cpp/include/*.h`, `app/src/main/cpp/Libs/*/libonnxruntime.so`, `app/src/main/cpp/Libs/*/libonnxruntime4j_jni.so`
- **Repository**: https://github.com/microsoft/onnxruntime
- **License**: MIT License

```text
MIT License

Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### 3. OpenCV
- **Files**: `app/src/main/cpp/Libs/*/libopencv_java4.so`
- **Repository**: https://github.com/opencv/opencv
- **License**: Apache License 2.0

```text
Copyright (C) 2000-2024, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2023, OpenCV Foundation, all rights reserved.
Copyright (C) 2008-2016, Itseez Inc., all rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Major External Dependencies (Managed via Gradle / pip)

| Component | Usage | License |
| :--- | :--- | :--- |
| AndroidX / Jetpack Compose / Room | Android App UI & Local Storage | Apache-2.0 |
| Kotlin & kotlinx.coroutines | Android App Runtime | Apache-2.0 |
| PyTorch / TorchAudio | Model Training & Audio Processing | BSD-3-Clause |
| timm (PyTorch Image Models) | MobileNetV3 Backbone | Apache-2.0 |
| librosa | Audio Feature Extraction | ISC |
| SciPy / scikit-learn | Scientific Computing & Evaluation | BSD-3-Clause |
| ONNX | Model Export Format | Apache-2.0 |
