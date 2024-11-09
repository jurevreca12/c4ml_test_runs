# synth.tcl is a synthesis script for Vivado
if { $argc != 2 } {
	break
}

set work_dir [lindex $argv 0]
set base_dir [lindex $argv 1]
set source_dir $work_dir/myproject_prj/solution1/syn/verilog

add_files $work_dir
set_part xcvu9p-flga2104-2L-e
synth_design -top myproject -retiming -flatten_hierarchy full
opt_design
report_timing_summary -file $work_dir/post_synth_timing_summary.rpt
report_utilization -file $work_dir/utilization.rpt
report_design_analysis -csv $work_dir/design_analysis.csv -timing
