# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants shared between gateware and C++"""


class Constants:
    # Maximum number of 8-bit channels per pixel
    MAX_CHANNEL_DEPTH = 512

    # Height (A dimension, aka activation values, aka inputs)
    SYS_ARRAY_HEIGHT = 4

    # Width (B dimension, aka filter values) of the Systolic array
    SYS_ARRAY_WIDTH = 2

    # Total Number of MACC blocks in Systolic array
    SYS_ARRAY_MACC_BLOCKS = SYS_ARRAY_HEIGHT * SYS_ARRAY_WIDTH

    # Total number of separate store
    NUM_FILTER_STORES = SYS_ARRAY_WIDTH

    # Depth of filter storage, per store 
    FILTER_WORDS_PER_STORE = 512
