
module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

wire [6:0] op;
wire signed [ 31:0] SIMD_OUT;
wire signed [31:0] x, term1, term2, term3, term4;
wire signed [63:0] x2;
wire signed [63:0] x3;
wire signed [63:0] x4;
wire signed [31:0] half;
wire signed [63:0] half_t_x;
reg  [127:0] SIMD_input_buf;
reg  [127:0] SIMD_filter_buf;
reg signed [ 7:0] SIMD_inputOffset;
reg signed [31:0] acc;
reg signed [63:0] exp_result;

wire [7:0] filter_width, filter_depth, filter_height;
wire [7:0] in_y, in_x;
wire [7:0] in_channel, out_channel;


parameter one_over_six = 357913948;
parameter one_over_ttfour = 894784853;

assign op = cmd_payload_function_id[9:3];
assign cmd_ready = ~rsp_valid;
assign x = cmd_payload_inputs_0;
assign x2 = (x * x)  >>> 31;
assign x3 = (x2 * x) >>> 31;
assign x4 = (x2 * x2) >>> 31;
assign term1 = x;
assign term2 = (x2 >>> 1);
assign term3 = ((x3)*one_over_six) >>> 31;
assign term4 = ((x4)*one_over_ttfour) >>> 31;
assign half = cmd_payload_inputs_1;
assign half_t_x = (x * half) >>> 30;

always @(posedge clk or posedge reset) begin
  if (reset) begin
    rsp_payload_outputs_0 <= 32'b0;
    rsp_valid <= 1'b0;
  end
  else if (rsp_valid) begin
    rsp_valid <= ~rsp_ready;
  end
  else if (cmd_valid) begin
    rsp_valid <= 1'b1;
    case (op)
      7'd1 : begin
        rsp_payload_outputs_0 <= 0;
        acc <= 0;
        SIMD_input_buf <= 0;
        SIMD_filter_buf <= 0;
      end
      7'd2 : begin
        SIMD_input_buf[31:0] <= cmd_payload_inputs_0;
        SIMD_filter_buf[31:0] <= cmd_payload_inputs_1;
      end
      7'd3 : begin
        SIMD_input_buf[63:32] <= cmd_payload_inputs_0;
        SIMD_filter_buf[63:32] <= cmd_payload_inputs_1;
      end
      7'd4 : begin
        SIMD_input_buf[95:64] <= cmd_payload_inputs_0;
        SIMD_filter_buf[95:64] <= cmd_payload_inputs_1;
      end
      7'd5 : begin
        SIMD_input_buf[127:96] <= cmd_payload_inputs_0;
        SIMD_filter_buf[127:96] <= cmd_payload_inputs_1;
      end
      7'd6 : begin
        SIMD_inputOffset <= cmd_payload_inputs_0;
      end
      7'd7 : begin
        acc <= acc + SIMD_OUT;
      end
      7'd8 : begin
        rsp_payload_outputs_0 <= acc;
      end
      7'd9 : begin
        acc <= acc + $signed(cmd_payload_inputs_0);
      end
      7'd10 : begin
        exp_result = (term1+term2+term3+term4);
      end
      7'd11 : begin
        exp_result = (x*(1073741824-(half_t_x)))>>>30;
      end
      7'd12 : begin
        exp_result = (x * half) >>> 31;
      end
      7'd13 : begin
        rsp_payload_outputs_0 <= exp_result;
      end
      7'd14 : begin
        /*
        in_y <= cmd_payload_inputs_0[31:24];
        filter_width <= cmd_payload_inputs_0[23:16];
        filter_depth <= cmd_payload_inputs_0[15:8];
        filter_height <= cmd_payload_inputs_0[7:0];
        in_x <= cmd_payload_inputs_1[31:24];
        in_channel <= cmd_payload_inputs_1[23:16];
        out_channel <= cmd_payload_inputs_1[15:8];
        */
      end
      7'd15 : begin
        rsp_payload_outputs_0 <= (out_channel*filter_depth*filter_width*filter_height) + (in_y * filter_width * filter_depth) + (in_x * filter_depth) + in_channel;
      end
      default: begin
        acc <= acc;
      end
    endcase
  end
end

assign in_y = cmd_payload_inputs_0[31:24];
assign filter_width = cmd_payload_inputs_0[23:16];
assign filter_depth = cmd_payload_inputs_0[15:8];
assign filter_height = cmd_payload_inputs_0[7:0];
assign in_x = cmd_payload_inputs_1[31:24];
assign in_channel = cmd_payload_inputs_1[23:16];
assign out_channel = cmd_payload_inputs_1[15:8];

CFU_SIMD M1 (
  .SIMD_input(SIMD_input_buf),
  .SIMD_filter(SIMD_filter_buf),
  .SIMD_inputOffset(SIMD_inputOffset),
  .SIMD_OUT(SIMD_OUT)
);

endmodule

module CFU_SIMD (
  SIMD_input,
  SIMD_filter,
  SIMD_inputOffset,
  SIMD_OUT
);

input         [127:0] SIMD_input;
input         [127:0] SIMD_filter;
input  signed [  7:0] SIMD_inputOffset;
output signed [ 31:0] SIMD_OUT;  

wire signed [16:0] prod0, prod1, prod2, prod3;
wire signed [16:0] prod4, prod5, prod6, prod7;
wire signed [16:0] prod8, prod9, prod10, prod11;
wire signed [16:0] prod12, prod13, prod14, prod15;
wire signed [16:0] InputOffset;

assign InputOffset = 16'd128;

assign prod0 = ($signed(SIMD_input[7:0]) + InputOffset) * $signed(SIMD_filter[7:0]);
assign prod1 = ($signed(SIMD_input[15:8])  + InputOffset) * $signed(SIMD_filter[15:8]);
assign prod2 = ($signed(SIMD_input[23:16]) + InputOffset) * $signed(SIMD_filter[23:16]);
assign prod3 = ($signed(SIMD_input[31:24]) + InputOffset) * $signed(SIMD_filter[31:24]);

assign prod4 = ($signed(SIMD_input[39:32]) + InputOffset) * $signed(SIMD_filter[39:32]);
assign prod5 = ($signed(SIMD_input[47:40]) + InputOffset) * $signed(SIMD_filter[47:40]);
assign prod6 = ($signed(SIMD_input[55:48]) + InputOffset) * $signed(SIMD_filter[55:48]);
assign prod7 = ($signed(SIMD_input[63:56]) + InputOffset) * $signed(SIMD_filter[63:56]);

assign prod8 = ($signed(SIMD_input [71:64]) + InputOffset) * $signed(SIMD_filter[71:64]);
assign prod9 = ($signed(SIMD_input [79:72]) + InputOffset) * $signed(SIMD_filter[79:72]);
assign prod10 = ($signed(SIMD_input[87:80]) + InputOffset) * $signed(SIMD_filter[87:80]);
assign prod11 = ($signed(SIMD_input[95:88]) + InputOffset) * $signed(SIMD_filter[95:88]);

assign prod12 = ($signed(SIMD_input[103:96]) + InputOffset) * $signed(SIMD_filter[103:96]);
assign prod13 = ($signed(SIMD_input[111:104]) + InputOffset) * $signed(SIMD_filter[111:104]);
assign prod14 = ($signed(SIMD_input[119:112]) + InputOffset) * $signed(SIMD_filter[119:112]);
assign prod15 = ($signed(SIMD_input[127:120]) + InputOffset) * $signed(SIMD_filter[127:120]);

assign SIMD_OUT = prod0 + prod1 + prod2 + prod3 + 
                  prod4 + prod5 + prod6 + prod7 + 
                  prod8 + prod9 + prod10 + prod11 + 
                  prod12 + prod13 + prod14 + prod15 ;

endmodule