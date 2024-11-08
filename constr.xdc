create_clock -period 10.000 -name clock [get_ports -filter { NAME =~  "*clock*" && DIRECTION == "IN" }]
set _xlnx_shared_i0 [get_ports -filter { NAME =~  "*io_in*" && DIRECTION == "IN" }]
set_input_delay -clock [get_clocks *clock*] 0.000 $_xlnx_shared_i0
set_output_delay -clock [get_clocks *clock*] 0.000 [get_ports -filter { NAME =~  "*io_out*" && DIRECTION == "OUT" }]
