# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    args   = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=False)    
    engine = Engine(config, mode="infer")
    #class_names = ["00_OK", "01_NG"]
    
    save_images = False
    for image_mode  in ["Eval", "Train", "Test1", "Test2", "Test3"]: #["Eval", "Train", "Test1", "Test2", "Test3"]: 
        print("\n---------------image_mode: ", image_mode)
        assert image_mode in ["Eval", "Train", "Test1", "Test2", "Test3"], "image_mode should be Eval or Test1 or Test2 or Train"
        engine.infer_tim(save_images=save_images, image_mode=image_mode, class_names=None)
        #engine.infer()
