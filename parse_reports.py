from vivado_report_parser import parse_metadata, parse_vivado_report
import pandas as pd
import json
import os


def parse_reports(work_dir, util_rpt_file="utilization.rpt"):
    with open(f"{work_dir}/{util_rpt_file}", "r") as f:
        util_rpt = f.read()
    _ = parse_metadata(util_rpt)
    util_data = parse_vivado_report(util_rpt)

    with open(f"{work_dir}/info.json", "r") as f:
        info_rpt = json.load(f)

    df = pd.read_csv(f"{work_dir}/design_analysis.csv")
    _, drow = next(df.iterrows())
    design_data = drow.to_dict()

    with open(f"{work_dir}/memory_info.json") as f:
        mem_info = json.load(f)

    reports = {
        "util": util_data,
        "design": design_data,
        "info_rpt": info_rpt,
        "mem_info": mem_info,
    }

    if os.path.exists(f"{work_dir}/memory_info_c4ml.json"):
        with open(f"{work_dir}/memory_info_c4ml.json") as f:
            mem_info_c4ml = json.load(f)
        reports["mem_info_c4ml"] = mem_info_c4ml
    return reports
