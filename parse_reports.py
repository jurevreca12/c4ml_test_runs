from vivado_report_parser import parse_metadata, parse_vivado_report
import json
import os
import csv

def parse_reports(work_dir, util_rpt_file="utilization.rpt"):
    with open(f"{work_dir}/{util_rpt_file}", "r") as f:
        util_rpt = f.read()
    _ = parse_metadata(util_rpt)
    util_data = parse_vivado_report(util_rpt)

    with open(f"{work_dir}/info.json", "r") as f:
        info_rpt = json.load(f)

    with open(f"{work_dir}/design_analysis.csv", 'r') as f:
        dict_reader = csv.DictReader(f)
        design_data = [row for row in dict_reader][0]

    reports = {
        "util": util_data,
        "design": design_data,
        "info_rpt": info_rpt,
    }

    if os.path.exists(f"{work_dir}/memory_info_c4ml.json"):
        with open(f"{work_dir}/memory_info_c4ml.json") as f:
            mem_info_c4ml = json.load(f)
        reports["mem_info_c4ml"] = mem_info_c4ml
    return reports


def parse_finn_reports(work_dir):
    with open(f"{work_dir}/time_per_step.json", 'r') as f:
        time_per_step = json.load(f)
    with open(f"{work_dir}/info.json", 'r') as f:
        info_rpt = json.load(f)
    with open(f"{work_dir}/report/rtlsim_performance.json", 'r') as f:
        rtlsim_rpt = json.load(f)
    with open(f"{work_dir}/report/ooc_synth_and_timing.json", 'r') as f:
        synth_rpt = json.load(f)

    return {
        'time_per_step': time_per_step,
        'info_rpt': info_rpt,
        'rtlsim_performance': rtlsim_rpt,
        'ooc_synth_and_timing': synth_rpt
    }
