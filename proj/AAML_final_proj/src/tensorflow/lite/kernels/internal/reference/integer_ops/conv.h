/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "perf.h"
#include "playground_util/print_params.h"

#include "cfu.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  perf_enable_counter(6);
  printf( "\x1B[32m ================================================================================================================================================\n \x1B[0m");
  printf( "\x1B[32m ================================================================================================================================================\n \x1B[0m");
  printf( "\x1B[32m ================================================================================================================================================\n \x1B[0m");
  printf( "\x1B[32m ================================================================================================================================================\n \x1B[0m");
  printf( "\x1B[32m ================================================================================================================================================\n \x1B[0m");

  print_conv_params(params, input_shape, filter_shape, output_shape);
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  int kernel_size = filter_height * filter_width;
  // GEMM IM2COL
  if(input_offset == 128 ){
    for (int batch = 0; batch < batches; ++batch) {

          // int32_t accs[1000][1000];
          int32_t accs[300][810] = {0};
          // int32_t accs_verif[300][810] = {0};
          // int32_t accs_each_channel[1000][1000][300];
          int im2col_row_num;
          int im2col_col_num;
          int32_t fr2row[300][600] = {0}; // output channel, kernel size = 1 * input_depth
          int32_t im2col[600][810] = {0}; // # of field, kernel size

        // For each input channel //
        for (int in_channel = 0; in_channel < input_depth; in_channel ++) {

          
          


          // field counter //
          // int field_cnt = 0;

          // For each output channel //
          for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            // For each filter
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                fr2row[out_channel][filter_y * filter_width + filter_x + in_channel*kernel_size] = filter_data[Offset(
                        filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // printf("fr2row[%d][%d] = %ld\n",out_channel ,(filter_y * filter_width + filter_x) ,fr2row[out_channel][filter_y * filter_width + filter_x]);

              }
            }
          }
          im2col_row_num = 0;
          im2col_col_num = 0;
          // printf("fr2row width = %d\n",output_depth); //
          // printf("fr2row height = %d\n",(filter_height * filter_width)); //

            for (int out_y = 0; out_y < output_height; ++out_y) {
              const int in_y_origin = (out_y * stride_height) - pad_height;
              for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                  for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                    for (int filter_x = 0; filter_x < filter_width; ++filter_x) {

                            const int in_y = in_y_origin + dilation_height_factor * filter_y;
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;
                            // const int in_y = in_y_origin + filter_y;
                            // const int in_x = in_x_origin + filter_x;
                          
                            // Zero padding by omitting the areas outside the image.
                            const bool is_point_inside_image =
                                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                (in_y < input_height);


                            // Compute im2col row and column indices directly
                            im2col_row_num = filter_y * filter_width + filter_x + in_channel*kernel_size;
                            im2col_col_num = out_y * output_width + out_x;

                            
                            // printf("input_data[%d][%d] = %d\n", in_y, in_x, input_data[Offset(input_shape, batch, in_y, in_x, in_channel)]);
                            if (!is_point_inside_image) {
                              im2col[im2col_row_num][im2col_col_num] = -input_offset;
                                continue;
                            }

                            im2col[im2col_row_num][im2col_col_num] = input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];

                            // printf("im2col[%d][%d] = %ld\n", im2col_row_num, im2col_col_num, im2col[im2col_row_num][im2col_col_num]);

                            
                        }
                    }
                }
            }
        }//channel
        // for (int i = 0; i < output_depth; ++i) {
        //     for (int j = 0; j < (output_width * output_height); ++j) {//input_depth*kernel_size
        //         for (int k = 0; k < (input_depth*kernel_size); ++k) {
        //             accs_verif[i][j] += fr2row[i][k] * (im2col[k][j] + input_offset);
        //         }
        //     }
        // }
        
        
        constexpr int TILE_SIZE = 100; // Tile size ===============================================================================================================> You can modify TILE SIZE {2,4~128}

            for (int row_tile = 0; row_tile < output_depth; row_tile += TILE_SIZE) {
                for (int col_tile = 0; col_tile < (output_width * output_height); col_tile += TILE_SIZE) {

                    // Adjust actual tile size to avoid out-of-bound access
                    // int actual_tile_height = TILE_SIZE;
                    // int actual_tile_width =  TILE_SIZE;
                    int actual_tile_height = std::min(TILE_SIZE, output_depth - row_tile);
                    int actual_tile_width = std::min(TILE_SIZE, (output_width * output_height) - col_tile);

                    // Temporary tile result
                    int32_t C_tile[TILE_SIZE][TILE_SIZE] = {0};
                    // int32_t C_tile_verif[TILE_SIZE][TILE_SIZE] = {0};

                    int32_t A_tile[TILE_SIZE][300] = {0}; // Store A_val
                    int32_t B_tile[300][TILE_SIZE] = {-input_offset}; // Store B_val + input_offset
                    // Shared dimension
                    // int shared_dim = filter_height * filter_width * input_depth;
                    // int shared_dim = std::min(filter_height * filter_width * input_depth, TILE_SIZE);
                    // int shared_dim = filter_height * filter_width * input_depth;
                    // for (int k = 0; k < shared_dim; ++k) {

                    //     for (int i = 0; i < actual_tile_height; ++i) {
                    //         int row_index = row_tile + i; // Current global row index
                    //         if (row_index >= output_depth) continue; // Out of bounds, skip

                    //         int32_t A_val = fr2row[row_index][k];
                    //         A_tile[i][k] = fr2row[row_index][k];

                    //         for (int j = 0; j < actual_tile_width; ++j) {
                    //             int col_index = col_tile + j; // Current global column index
                    //             if (col_index >= (output_width * output_height)) continue; // Out of bounds, skip

                    //             int32_t B_val = im2col[k][col_index];
                    //             B_tile[k][j] = im2col[k][col_index];

                    //             // Perform multiplication and accumulate GEMM
                    //             C_tile_verif[i][j] += A_val * (B_val + input_offset);

                    //             // Debug output for partial sum
                    //             printf("A_tile[%d][%d] = %ld\n", i, k, A_tile[i][k]);
                    //             printf("B_tile[%d][%d] = %ld\n", k, j, B_tile[k][j]);
                    //             printf("Tile[%d][%d]: A_val = %ld, B_val = %ld, C_tile_verif = %ld\n",i, j, A_val, B_val, C_tile_verif[i][j]);
                    //             // printf("Tile[%d][%d]: A_val = %d, B_val = %d\n",i, j, A_val, B_val, C_tile_verif[i][j]);
                    //             __asm volatile("NOP"); 
                    //         }
                    //     }
                    // }
                    // for (int k = 0; k < shared_dim; ++k) {
                    //     for (int i = 0; i < actual_tile_height; ++i) {
                    //         int row_index = row_tile + i;
                    //         if (row_index >= output_depth) continue;

                    //         // Fill A_tile
                    //         A_tile[i][k] = fr2row[row_index][k];
                    //         // printf("A_tile[%d][%d] = %ld\n", i, k, A_tile[i][k]);
                    //     }
                    //     for (int j = 0; j < actual_tile_width; ++j) {
                    //         int col_index = col_tile + j;
                    //         if (col_index >= (output_width * output_height)) continue;

                    //         // Fill B_tile
                    //         B_tile[k][j] = im2col[k][col_index];
                    //         // printf("B_tile[%d][%d] = %ld\n", k, j, B_tile[k][j]);
                    //     }
                    // }
                    // for (int i = 0; i < TILE_SIZE; ++i) {
                    //     for (int j = 0; j < TILE_SIZE; ++j) {
                    //         for (int k = 0; k < TILE_SIZE; ++k) {
                    //             C_tile[i][j] += A_tile[i][k] * (B_tile[k][j] + input_offset);
                    //         }
                    //     }
                    // }
                    int total_shared_dim = filter_height * filter_width * input_depth;
                    for (int k_start = 0; k_start < total_shared_dim; k_start += TILE_SIZE) {
                        int actual_shared_dim = std::min(TILE_SIZE, total_shared_dim - k_start);

                        // 填充 A_tile 和 B_tile
                        for (int k = 0; k < actual_shared_dim; ++k) {
                            for (int i = 0; i < actual_tile_height; ++i) {
                                int row_index = row_tile + i;
                                A_tile[i][k] = (row_index < output_depth) ? fr2row[row_index][k_start + k] : 0;
                            }
                            for (int j = 0; j < actual_tile_width; ++j) {
                                int col_index = col_tile + j;
                                B_tile[k][j] = (col_index < (output_width * output_height)) ? im2col[k_start + k][col_index] : -input_offset;
                            }
                        }

                        // 矩阵乘法累积
                        // for (int i = 0; i < actual_tile_height; ++i) {
                        //     for (int j = 0; j < actual_tile_width; ++j) {
                        //         for (int k = 0; k < actual_shared_dim; ++k) {
                        //             C_tile[i][j] += A_tile[i][k] * (B_tile[k][j] + input_offset);
                        //         }
                        //     }
                        // }
                        // int32_t cfu_out = cfu_op0(0, 0, 0);
                        cfu_op0(0, 0, 0);
                        uint32_t addr_a=0;
                        uint32_t addr_b=0;

                        for(int i=0; i < actual_tile_height; i+=4){ //M
                          for(int k=0; k < actual_shared_dim; k++){ //K
                            uint32_t in_a = 0;

                            for (int b = 0; b < 4; ++b) {
                                int current_row = i + b;
                                int8_t value = (current_row < output_depth) ? A_tile[current_row][k] : 0;  // 超出范围填充 0
                                in_a |= ((uint32_t)(value & 0xFF)) << ((3 - b) * 8); // 倒置字节顺序
                            }
                            // printf("addr_a  = %ld, in_a = %08lX\n", addr_a, in_a);
                            cfu_op0(1, addr_a, in_a);
                            // __asm volatile("NOP"); 
                            addr_a++;
                          }
                        }
                      
                        for (int j = 0; j < actual_tile_width; j += 4) { // 每次处理4列
                          for (int k = 0; k < actual_shared_dim; k++) { // 遍历行
                              uint32_t in_b = 0;

                              // 打包4个 int8 数据到 32-bit
                              for (int b = 0; b < 4; ++b) {
                                  int current_col = j + b; // 当前列索引
                                  int8_t value = (current_col < TILE_SIZE) ? B_tile[k][current_col] : -input_offset; // 超出范围填充
                                  in_b |= ((uint32_t)(value & 0xFF)) << ((3 - b) * 8); // 位移基于偏移量 b
                              }

                              // 发送到 CFU
                              cfu_op0(2, addr_b, in_b);
                              // printf("addr_b = %ld, in_b = %08lX\n", addr_b, in_b);
                              // __asm volatile("NOP"); 
                              addr_b++;
                          }
                      }
                      uint32_t MK_32;
                      uint32_t N_32;
                      MK_32 = (actual_tile_height << 16) | (actual_shared_dim);
                      N_32 = (actual_tile_width);
                      // printf("TILE_SIZE = %d\n", TILE_SIZE);
                      // printf("MK_32 = %08lX, N_32  = %ld\n", MK_32, N_32);
                      uint32_t cfu_out = cfu_op0(3, MK_32, N_32); // {M,K}, N
                      cfu_out = 1;
                      while(cfu_out){
                        cfu_out = cfu_op0(4, 0, 0);         
                      }
                      // printf("CFU has done calculation.\n");
                        uint32_t index, choose_C_data_out, row, col;
                        uint32_t total_index = ((actual_tile_height * actual_tile_width)/4) + (((actual_tile_height * actual_tile_width)%4)!= 0); // 總元素數量
                        uint32_t elements_per_index = 4;  // 每個 index 包含 4 個元素
                        // printf("actual_tile_height = %d\n", actual_tile_height);
                        // printf("actual_tile_width = %d\n", actual_tile_width);
                        for (index = 0; index < total_index; index++) { // 遍歷每個 index
                            for (choose_C_data_out = 0; choose_C_data_out < elements_per_index; choose_C_data_out++) {


                                // 計算 row 和 col 索引
                                row = index % actual_tile_height; 
                                col = (index / actual_tile_height) * elements_per_index + choose_C_data_out; 

                                // 從 CFU 獲取值
                                int32_t cfu_value = cfu_op0(5, index, choose_C_data_out);

                                // 更新 accs 矩陣
                                C_tile[row][col] += cfu_value;

                                // 調試輸出
                                // printf("cfu_op0(5, index = %ld, choose_C_data_out = %ld) = %ld\n", index, choose_C_data_out, cfu_value);
                                // printf("C_tile[%ld][%ld] = %ld\n", row, col, C_tile[row][col]);
                                // __asm volatile("NOP"); 
                            }
                      }
                    } /////for (int k_start 

                    // for (int k_start = 0; k_start < total_shared_dim; k_start += TILE_SIZE) {
                    //   int actual_shared_dim = std::min(TILE_SIZE, total_shared_dim - k_start);
                    //     for (int i = 0; i < actual_tile_height; ++i) {
                    //         for (int j = 0; j < actual_tile_width; ++j) {
                    //             for (int k = 0; k < actual_shared_dim; ++k) {
                    //                 C_tile[i][j] += A_tile[i][k] * (B_tile[k][j] + input_offset);
                    //             }
                    //         }
                    //     }
                    // }




                  //   int32_t cfu_out = cfu_op0(0, 0, 0);
                  //   cfu_op0(0, 0, 0);
                  //   uint32_t addr_a=0;
                  //   uint32_t addr_b=0;

                  //   for(int i=0; i < TILE_SIZE; i+=4){ //M
                  //     for(int k=0; k < TILE_SIZE; k++){ //K
                  //       uint32_t in_a = 0;

                  //       for (int b = 0; b < 4; ++b) {
                  //           int current_row = i + b;
                  //           int8_t value = (current_row < output_depth) ? A_tile[current_row][k] : 0;  // 超出范围填充 0
                  //           in_a |= ((uint32_t)(value & 0xFF)) << ((3 - b) * 8); // 倒置字节顺序
                  //       }
                  //       // printf("addr_a  = %ld, in_a = %08lX\n", addr_a, in_a);
                  //       cfu_op0(1, addr_a, in_a);
                  //       // __asm volatile("NOP"); 
                  //       addr_a++;
                  //     }
                  //   }
                  
                  //   for (int j = 0; j < TILE_SIZE; j += 4) { // 每次处理4列
                  //     for (int k = 0; k < TILE_SIZE; k++) { // 遍历行
                  //         uint32_t in_b = 0;

                  //         // 打包4个 int8 数据到 32-bit
                  //         for (int b = 0; b < 4; ++b) {
                  //             int current_col = j + b; // 当前列索引
                  //             int8_t value = (current_col < TILE_SIZE) ? B_tile[k][current_col] : -input_offset; // 超出范围填充
                  //             in_b |= ((uint32_t)(value & 0xFF)) << ((3 - b) * 8); // 位移基于偏移量 b
                  //         }

                  //         // 发送到 CFU
                  //         cfu_op0(2, addr_b, in_b);
                  //         // printf("addr_b = %ld, in_b = %08lX\n", addr_b, in_b);
                  //         // __asm volatile("NOP"); 
                  //         addr_b++;
                  //     }
                  // }
                  // uint32_t MK_32;
                  // uint32_t N_32;
                  // MK_32 = (TILE_SIZE << 16) | (TILE_SIZE);
                  // N_32 = (TILE_SIZE);
                  // // printf("TILE_SIZE = %d\n", TILE_SIZE);
                  // // printf("MK_32 = %08lX, N_32  = %ld\n", MK_32, N_32);
                  // uint32_t cfu_out = cfu_op0(3, MK_32, N_32); // {M,K}, N
                  // cfu_out = 1;
                  // while(cfu_out){
                  //   cfu_out = cfu_op0(4, 0, 0);         
                  // }
                  // // printf("CFU has done calculation.\n");
                  //   uint32_t index, choose_C_data_out, row, col;
                  //   uint32_t total_index = ((TILE_SIZE * TILE_SIZE)/4); // 總元素數量
                  //   uint32_t elements_per_index = 4;  // 每個 index 包含 4 個元素

                  //   for (index = 0; index < total_index; index++) { // 遍歷每個 index
                  //       for (choose_C_data_out = 0; choose_C_data_out < elements_per_index; choose_C_data_out++) {


                  //           // 計算 row 和 col 索引
                  //           row = index % TILE_SIZE; 
                  //           col = (index / TILE_SIZE) * elements_per_index + choose_C_data_out; 

                  //           // 從 CFU 獲取值
                  //           int32_t cfu_value = cfu_op0(5, index, choose_C_data_out);

                  //           // 更新 accs 矩陣
                  //           C_tile[row][col] += cfu_value;

                  //           // 調試輸出
                  //           // printf("cfu_op0(5, index = %ld, choose_C_data_out = %ld) = %ld\n", index, choose_C_data_out, cfu_value);
                  //           // printf("C_tile[%ld][%ld] = %ld\n", row, col, C_tile[row][col]);
                  //           // __asm volatile("NOP"); 
                  //       }
                  //   }

                
                    // __asm volatile("NOP"); 

                    for (int i = 0; i < actual_tile_height; ++i) {
                        int row_index = row_tile + i; // Current global row index
                        if (row_index >= output_depth) continue; // Out of bounds, skip

                        for (int j = 0; j < actual_tile_width; ++j) {
                            int col_index = col_tile + j; // Current global column index
                            if (col_index >= (output_width * output_height)) continue; // Out of bounds, skip

                            // accs[row_index][col_index] += C_tile_verif[i][j];
                            accs[row_index][col_index] += C_tile[i][j];

                            // Debug output for accumulated result
                            // printf("C_tile_verif[%d][%d] = %ld\n", i, j, C_tile_verif[i][j]);
                            // printf("C_tile[%d][%d] = %ld\n", i, j, C_tile[i][j]);

                        }
                    }

                }
            }






                // Output data modification //
                for (int out_y = 0; out_y < output_height; ++out_y) {
                  for (int out_x = 0; out_x < output_width; ++out_x) {
                    for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                      // int32_t temp = accs[ out_y * output_width + out_x][out_channel];
                      int32_t temp = accs[out_channel][out_y * output_width + out_x];
                      // int32_t temp = accs_verif[out_channel][out_y * output_width + out_x];
                      // if((accs_verif[out_channel][out_y * output_width + out_x])!= (accs[out_channel][out_y * output_width + out_x])){
                      //     printf("accs_verif[%d][%d]   = %ld\n",out_channel,(out_y * output_width + out_x),accs_verif[out_channel][out_y * output_width + out_x]);
                      //     printf("accs[%d][%d]   = %ld\n",out_channel,(out_y * output_width + out_x),accs[out_channel][out_y * output_width + out_x]);
                      // }
                      


                      if (bias_data) {
                        temp += bias_data[out_channel];
                      }

                      temp = MultiplyByQuantizedMultiplier(
                          temp, output_multiplier[out_channel], output_shift[out_channel]);
                      temp += output_offset;
                      temp = std::max(temp, output_activation_min);
                      temp = std::min(temp, output_activation_max);

                      output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                                static_cast<int8_t>(temp);

                      // printf("output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = %d\n",output_data[Offset(output_shape, batch, out_y, out_x, out_channel)]);

                    }
                  }
                }
              }
          }
    
  else{
  
   //Original
    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            int32_t acc; // resets acc

            auto group = out_channel / filters_per_group;
            acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;

                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);

                if (!is_point_inside_image) {
                  continue;
                }
                  
                

                for (int in_channel = 0; in_channel < filter_input_depth;
                    ++in_channel) {
                  int32_t input_val =
                      input_data[Offset(input_shape, batch, in_y, in_x,
                                        in_channel + group * filter_input_depth)];
                  int32_t filter_val = filter_data[Offset(
                      filter_shape, out_channel, filter_y, filter_x, in_channel)];
                  // printf("filter_val[%d][%d] = %d\n", filter_y, filter_x, filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)]);
                  // printf("with offset input_data[%d][%d] = %ld\n", in_y, in_x, (input_val + input_offset));
                  
                  acc += filter_val * (input_val + input_offset);
                }

                
              }
            }
            

            if (bias_data) {
              acc += bias_data[out_channel];
            }
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[out_channel], output_shift[out_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                static_cast<int8_t>(acc);
                // printf("output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = %d\n",output_data[Offset(output_shape, batch, out_y, out_x, out_channel)]);

          }
        }
      }
    }
  }

  perf_disable_counter(6);
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
