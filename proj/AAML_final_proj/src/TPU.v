
module TPU(
           clk,
           rst_n,

           in_valid,
           K,
           M,
           N,
           busy,

           A_wr_en,
           A_index,
           A_data_in,
           A_data_out,

           B_wr_en,
           B_index,
           B_data_in,
           B_data_out,

           C_wr_en,
           C_index,
           C_data_in,
           C_data_out,


           inputOffset
       );


input clk;
input rst_n;
input            in_valid;
input [7:0]      K;
input [7:0]      M;
input [7:0]      N;
output  reg      busy;

output           A_wr_en;
output [15:0]    A_index;
output [31:0]    A_data_in;
input  signed [31:0]    A_data_out;

output           B_wr_en;
output [15:0]    B_index;
output [31:0]    B_data_in;
input  signed [31:0]    B_data_out;

output           C_wr_en;
output [15:0]    C_index;
output signed [127:0]   C_data_in;
input  [127:0]   C_data_out;




input  [31:0] inputOffset;
/*
  printf("fmaps_num: %d\n",fmaps_num);
  printf("input_width = %d\n",input_width);
  printf("input_height = %d\n\n",input_height);

  printf("filter_num: %d\n",filter_num);
  printf("filter_input_depth = %d\n",filter_input_depth);
  printf("filter width = %d\n",filter_width);
  printf("filter height = %d\n\n",filter_height);


  printf("result_size: %d\n", result_size);
  printf("result_num: %d\n",result_num);
  printf("result width = %d\n",output_width);
  printf("result height = %d\n\n",output_height);

  printf("pad_width: %d\n",pad_width);
  printf("pad_height: %d\n",pad_height);
  printf("pad_h: %d\n",pad_h);
  printf("pad_w: %d\n",pad_w);



  printf("stride_width: %d\n",stride_width);
  printf("stride_height: %d\n",stride_height);

  printf("dilation_width_factor: %d\n",dilation_width_factor);
  printf("dilation_height_factor: %d\n",dilation_height_factor);

  printf("Input offset: %ld\n", input_offset);
  printf("output offset: %ld\n", output_offset);

  */;






integer i;
parameter IDLE = 2'b00;
parameter CALC = 2'b01;
parameter WAIT = 2'b10;
parameter WRIT = 2'b11;

reg pattern_end;
reg row_end;
reg PE_reset;
reg [7:0] K_reg, M_reg, N_reg;
reg [1:0] curr_state, next_state;
reg [7:0] col_counter;
reg [8:0] K_counter;
reg [1:0] counter4;
reg [15:0] addr_counter_A, addr_counter_B, addr_counter_C;
reg signed [127:0] out_data;
reg signed [31:0] A1_d, B1_d;
reg signed [31:0] A2_d, B2_d, A2_dd, B2_dd;
reg signed [31:0] A3_d, B3_d, A3_dd, B3_dd, A3_ddd, B3_ddd;
reg [15:0] fit_M;
reg signed [31:0] A_buf [0:3];
reg signed [31:0] B_buf [0:3];
reg [1:0] WRIT_counter;
reg [7:0] row_counter;
reg [15:0] B_offset;

wire signed [5:0] K_Quo, M_Quo, N_Quo;
wire signed [1:0] K_Rem, M_Rem, N_Rem;
wire signed [31:0] n0001, n0102, n0203;
wire signed [31:0] n1011, n1112, n1213;
wire signed [31:0] n2021, n2122, n2223;
wire signed [31:0] n3031, n3132, n3233;

wire signed [31:0] n0010, n1020, n2030;
wire signed [31:0] n0111, n1121, n2131;
wire signed [31:0] n0212, n1222, n2232;
wire signed [31:0] n0313, n1323, n2333;

wire signed [127:0] out_buf0, out_buf1, out_buf2, out_buf3;

// ==============================================================

assign K_Quo = (K_reg >>> 2) + 1;
assign M_Quo = (M_reg >>> 2) + 1;
assign N_Quo = (N_reg >>> 2) + 1;
assign K_Rem = K_reg - (K_Quo <<< 2);
assign M_Rem = K_reg - (M_Quo <<< 2);
assign N_Rem = K_reg - (N_Quo <<< 2);
assign A_index = addr_counter_A;
assign B_index = addr_counter_B;
assign C_index = addr_counter_C;
assign C_wr_en = (curr_state == WRIT) ? 1 : 0;
assign A_wr_en = 0;
assign B_wr_en = 0;
assign A_data_in = 0;
assign B_data_in = 0;
assign C_data_in = out_data;



// A_buf
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i=0;i<4;i=i+1) begin
            A_buf[i] <= 0;
            B_buf[i] <= 0;
        end
    end
    else if (K_counter >= K_reg) begin
        for (i=0;i<4;i=i+1) begin
            A_buf[i] <= 0;
            B_buf[i] <= 0;
        end
    end
    else if (curr_state == CALC) begin
        A_buf[0] <= {{24{A_data_out[31]}},{A_data_out[31:24]}}   + inputOffset[31:0];
        A_buf[1] <= {{24{A_data_out[23]}},{A_data_out[23:16]}}   + inputOffset[31:0];
        A_buf[2] <= {{24{A_data_out[15]}},{A_data_out[15: 8]}}   + inputOffset[31:0];
        A_buf[3] <= {{24{A_data_out[ 7]}},{A_data_out[ 7: 0]}}   + inputOffset[31:0];

        B_buf[0] <= {{24{B_data_out[31]}},{B_data_out[31:24]}};
        B_buf[1] <= {{24{B_data_out[23]}},{B_data_out[23:16]}};
        B_buf[2] <= {{24{B_data_out[15]}},{B_data_out[15: 8]}};
        B_buf[3] <= {{24{B_data_out[ 7]}},{B_data_out[ 7: 0]}};
    end
    else begin
        for (i=0;i<4;i=i+1) begin
            A_buf[i] <= 0;
            B_buf[i] <= 0;
        end
    end
end



// FSM
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        curr_state <= 0;
    end
    else begin
        curr_state <= next_state;
    end
end
always @(*) begin
    case (curr_state)
        IDLE : begin
            if (in_valid) begin
                next_state <= CALC;
            end
            else begin
                next_state <= IDLE;
            end
        end

        CALC : begin
            if (K_counter >= K_reg) begin
                next_state <= WAIT;
            end
            else begin
                next_state <= CALC;
            end
        end

        WAIT : begin
            if (counter4 == 2'b11) begin
                next_state <= WRIT;
            end
            else begin
                next_state <= WAIT;
            end
        end

        WRIT : begin
            if (pattern_end & WRIT_counter == 2'b11) begin
                next_state <= IDLE;
            end
            else if (WRIT_counter == 2'b11) begin
                next_state <= CALC;
            end
            else begin
                next_state <= WRIT;
            end
        end


        default: begin
            next_state <= IDLE;
        end
    endcase
end

// busy
always @(*) begin
    if (!rst_n) begin
        busy <= 0;
    end
    else if (next_state == IDLE) begin
        busy <= 0;
    end
    else begin
        busy <= 1;
    end
end

// out_data
always @(*) begin
    if (!rst_n) begin
        out_data <= 0;
    end
    else if (curr_state == WRIT) begin
        case (WRIT_counter)
            0 :
                out_data <= out_buf0;
            1 :
                out_data <= out_buf1;
            2 :
                out_data <= out_buf2;
            3 :
                out_data <= out_buf3;
            default:
                out_data <= 0;
        endcase
    end
    else begin
        out_data <= 69;
    end
end

// input reg
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        K_reg <= 0;
        M_reg <= 0;
        N_reg <= 0;
    end
    else if (in_valid) begin
        K_reg <= K;
        M_reg <= M;
        N_reg <= N;
    end
    else begin
        K_reg <= K_reg;
        M_reg <= M_reg;
        N_reg <= N_reg;
    end
end

// PE_reset
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        PE_reset <= 0;
    end
    else if (curr_state == WRIT && counter4 == 2'b10) begin
        PE_reset <= 1;
    end
    else begin
        PE_reset <= 0;
    end
end

// col_counter
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        col_counter <= 0;
    end
    else if (curr_state == IDLE) begin
        col_counter <= 0;
    end
    else if (row_end) begin
        col_counter <= col_counter + 1;
    end
    else begin
        col_counter <= col_counter;
    end
end


// row_counter
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        row_counter <= 0;
    end
    else if (curr_state == IDLE) begin
        row_counter <= 0;
    end
    else if (row_counter >= M_Quo && counter4 == 2'b11) begin
        row_counter <= 0;
    end
    else if (curr_state == WAIT && next_state == WRIT) begin
        row_counter <= row_counter + 1;
    end
    else begin
        row_counter <= row_counter;
    end
end

// row_end
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        row_end <= 0;
    end
    else if (row_counter >= M_Quo & curr_state == WRIT & counter4 == 2'b11) begin
        row_end <= 1;
    end
    else begin
        row_end <= 0;
    end
end

// WRIT_counter
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        WRIT_counter <= 0;
    end
    else if (curr_state == WRIT) begin
        WRIT_counter <= WRIT_counter + 1;
    end
    else begin
        WRIT_counter <= 0;
    end
end

// pattern_end
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pattern_end <= 0;
    end
    else if (col_counter >= N_Quo & row_counter >= M_Quo & curr_state == WRIT && counter4 == 2'b10) begin
        pattern_end <= 1;
    end

    else begin
        pattern_end <= 0;
    end
end


// B_offset
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        B_offset <= 0;
    end
    else if (curr_state == IDLE) begin
        B_offset <= 0;
    end
    else if (curr_state == WRIT & counter4 == 2'b01 & row_counter >= M_Quo) begin
        B_offset <= B_offset + K_reg;
    end
    else begin
        B_offset <= B_offset;
    end
end


// addr_counter_A
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        addr_counter_A <= 0;
    end
    else if (curr_state == IDLE) begin
        addr_counter_A <= 0;
    end

    else if (row_counter >= M_Quo & curr_state == WRIT & counter4 == 2'b11) begin
        addr_counter_A <= 0;
    end

    else if (curr_state == CALC & next_state == CALC) begin
        addr_counter_A <= addr_counter_A + 1;
    end
    else begin
        addr_counter_A <= addr_counter_A;
    end
end

// addr_counter_B
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        addr_counter_B <= 0;
    end
    else if (curr_state == IDLE) begin
        addr_counter_B <= 0;
    end
    else if (K_counter >= K_reg & next_state == CALC) begin
        addr_counter_B <= B_offset;
    end
    else if (curr_state == CALC & next_state == CALC) begin
        addr_counter_B <= addr_counter_B + 1;
    end
    else begin
        addr_counter_B <= addr_counter_B;
    end
end

// fit_M
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fit_M <= 0;
    end
    else if (curr_state == IDLE) begin
        fit_M <= 0;
    end
    else if (curr_state == WRIT && next_state == IDLE) begin
        fit_M <= 0;
    end
    else if (fit_M >= M_reg & curr_state == WRIT & counter4 == 2'b11) begin
        fit_M <= 0;
    end
    else if (curr_state == WRIT) begin
        fit_M <= fit_M + 1;
    end
    else begin
        fit_M <= fit_M;
    end
end

// addr_counter_C
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        addr_counter_C <= 0;
    end
    else if (curr_state == IDLE) begin
        addr_counter_C <= 0;
    end
    else if (curr_state == WRIT && next_state == IDLE) begin
        addr_counter_C <= 0;
    end
    else if (fit_M >= M_reg) begin
        addr_counter_C <= addr_counter_C;
    end
    else if (curr_state == WRIT) begin
        addr_counter_C <= addr_counter_C + 1;
    end
    else begin
        addr_counter_C <= addr_counter_C;
    end
end

// K_counter
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        K_counter <= 0;
    end
    else if (curr_state == WRIT & counter4 == 2'b11) begin
        K_counter <= 0;
    end
    else if (K_counter >= K_reg) begin
        K_counter <= K_counter;
    end
    else if (curr_state == CALC) begin
        K_counter <= K_counter + 1;
    end
    else begin
        K_counter <= K_counter;
    end
end

// delays
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        A1_d    <= 0;
        B1_d    <= 0;
        A2_d    <= 0;
        B2_d    <= 0;
        A2_dd   <= 0;
        B2_dd   <= 0;
        A3_d    <= 0;
        B3_d    <= 0;
        A3_dd   <= 0;
        B3_dd   <= 0;
        A3_ddd  <= 0;
        B3_ddd  <= 0;
    end
    else if (curr_state == CALC || curr_state == WAIT) begin
        A1_d <= A_buf[1];
        B1_d <= B_buf[1];

        A2_d <= A_buf[2];
        B2_d <= B_buf[2];
        A2_dd <= A2_d;
        B2_dd <= B2_d;

        A3_d <= A_buf[3];
        B3_d <= B_buf[3];
        A3_dd <= A3_d;
        B3_dd <= B3_d;
        A3_ddd <= A3_dd;
        B3_ddd <= B3_dd;
    end
    else begin
        A1_d    <= A1_d;
        B1_d    <= B1_d;
        A2_d    <= A2_d;
        B2_d    <= B2_d;
        A2_dd   <= A2_dd;
        B2_dd   <= B2_dd;
        A3_d    <= A3_d;
        B3_d    <= B3_d;
        A3_dd   <= A3_dd;
        B3_dd   <= B3_dd;
        A3_ddd  <= A3_ddd;
        B3_ddd  <= B3_ddd;
    end
end

// counter4
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        counter4 <= 0;
    end
    else if (curr_state == WAIT || curr_state == WRIT) begin
        counter4 <= counter4 + 1;
    end
    else if (curr_state == CALC) begin
        counter4 <= counter4 ;
    end
    else begin
        counter4 <= 0;
    end
end

// PEs
PE PE00 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(B_buf[0]), .left(A_buf[0]), .bot(n0010), .right(n0001), .out(out_buf0[127:96]));
PE PE01 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(B1_d  ), .left(n0001), .bot(n0111), .right(n0102), .out(out_buf0[ 95:64]));
PE PE02 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(B2_dd ), .left(n0102), .bot(n0212), .right(n0203), .out(out_buf0[ 63:32]));
PE PE03 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(B3_ddd), .left(n0203), .bot(n0313), .right(     ), .out(out_buf0[ 31: 0]));

PE PE10 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n0010), .left(A1_d),   .bot(n1020), .right(n1011), .out(out_buf1[127:96]));
PE PE11 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n0111), .left(n1011),  .bot(n1121), .right(n1112), .out(out_buf1[ 95:64]));
PE PE12 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n0212), .left(n1112),  .bot(n1222), .right(n1213), .out(out_buf1[ 63:32]));
PE PE13 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n0313), .left(n1213),  .bot(n1323), .right(     ), .out(out_buf1[ 31: 0]));

PE PE20 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n1020), .left(A2_dd),  .bot(n2030), .right(n2021), .out(out_buf2[127:96]));
PE PE21 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n1121), .left(n2021),  .bot(n2131), .right(n2122), .out(out_buf2[ 95:64]));
PE PE22 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n1222), .left(n2122),  .bot(n2232), .right(n2223), .out(out_buf2[ 63:32]));
PE PE23 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n1323), .left(n2223),  .bot(n2333), .right(     ), .out(out_buf2[ 31: 0]));

PE PE30 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n2030), .left(A3_ddd), .bot(     ), .right(n3031), .out(out_buf3[127:96]));
PE PE31 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n2131), .left(n3031),  .bot(     ), .right(n3132), .out(out_buf3[ 95:64]));
PE PE32 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n2232), .left(n3132),  .bot(     ), .right(n3233), .out(out_buf3[ 63:32]));
PE PE33 (.clk(clk), .rst_n(rst_n), .PE_reset(PE_reset), .top(n2333), .left(n3233),  .bot(     ), .right(     ), .out(out_buf3[ 31: 0]));

endmodule


    module PE (
        clk,
        rst_n,
        PE_reset,

        top,
        left,

        bot,
        right,

        out
    );

input clk, rst_n, PE_reset;
input      signed  [31:0]   top, left;
output reg signed  [31:0]   bot, right;
output reg signed  [31:0]  out;

// bot
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bot <= 0;
    end
    else if (PE_reset) begin
        bot <= 0;
    end
    else begin
        bot <= top;
    end
end

// right
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        right <= 0;
    end
    else if (PE_reset) begin
        right <= 0;
    end
    else begin
        right <= left;
    end
end

// out
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out <= 0;
    end
    else if (PE_reset) begin
        out <= 0;
    end
    else begin
        out <= out + top * left;
    end
end

endmodule
