from vivado_report_parser import parse_metadata, parse_vivado_report
import pandas as pd

def parse_reports(work_dir):
    with open(f'{work_dir}/utilization.rpt', 'r') as f:
        util_rpt = f.read()
    metadata = parse_metadata(util_rpt)
    util_data = parse_vivado_report(util_rpt)
    
    with open(f'{work_dir}/time.log', 'r') as f:
        syn_time_rpt = f.read()
    syn_time = syn_time_rpt.split('\n')[1]

    df = pd.read_csv(f'{work_dir}/design_analysis.csv')
    _, drow = next(df.iterrows())
    design_data = drow.to_dict()
    return {'util': util_data, 'design': design_data, 'syn_time': syn_time}
