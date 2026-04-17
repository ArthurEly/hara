import csv
import math

def process_sec(csv_path):
    # xc7z020 capacity
    fpga = {'LUTs': 53200, 'FFs': 106400, 'BRAM': 140, 'DSP': 220}
    targets = [10, 20, 30, 40]
    
    runs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'success':
                runs.append({
                    'name': row['hw_name'],
                    'dur': float(row['duration_in_seconds']),
                    'lut': float(row['Total LUTs']),
                    'ff': float(row['FFs']),
                    'bram': float(row['BRAM (36k)']),
                    'dsp': float(row['DSP Blocks']),
                    'fps': float(row['estimated_throughput_fps'])
                })

    print("=== SEC Scenarios (SAT6_T2W2) ===")
    for t in targets:
        max_lut = fpga['LUTs'] * (t / 100.0)
        max_ff = fpga['FFs'] * (t / 100.0)
        max_bram = fpga['BRAM'] * (t / 100.0)
        max_dsp = fpga['DSP'] * (t / 100.0)
        
        best_run = None
        best_idx = -1
        best_util = -1
        
        # In SEC, we want the highest utilization that fits within the area budget
        for i, r in enumerate(runs):
            if r['lut'] <= max_lut and r['ff'] <= max_ff and r['bram'] <= max_bram and r['dsp'] <= max_dsp:
                # We calculate the highest percentage utilized among all 4 resources
                util = max(r['lut']/fpga['LUTs'], r['ff']/fpga['FFs'], r['bram']/fpga['BRAM'], r['dsp']/fpga['DSP'])
                if best_run is None or util > best_util:
                    best_run = r
                    best_idx = i
                    best_util = util
                
        if best_run:
            total_time = sum(r['dur'] for r in runs[:best_idx+1])
            builds = best_idx + 1
            print(f"Target: {t}% Area")
            print(f"  Run: {best_run['name']} | FPS: {best_run['fps']:.2f}")
            print(f"  Total Area: "
                  f"LUT={best_run['lut']} ({best_run['lut']/fpga['LUTs']*100:.1f}%), "
                  f"FF={best_run['ff']} ({best_run['ff']/fpga['FFs']*100:.1f}%), "
                  f"BRAM={best_run['bram']} ({best_run['bram']/fpga['BRAM']*100:.1f}%), "
                  f"DSP={best_run['dsp']} ({best_run['dsp']/fpga['DSP']*100:.1f}%)")
            print(f"  Builds: {builds} | Time: {total_time/3600:.2f} h")
        else:
            close = runs[0]
            print(f"Target: {t}% Area (FAILED, showing run 1)")
            print(f"  Run: {close['name']} | FPS: {close['fps']:.2f}")
            print(f"  Total Area: "
                  f"LUT={close['lut']} ({close['lut']/fpga['LUTs']*100:.1f}%), "
                  f"FF={close['ff']} ({close['ff']/fpga['FFs']*100:.1f}%), "
                  f"BRAM={close['bram']} ({close['bram']/fpga['BRAM']*100:.1f}%), "
                  f"DSP={close['dsp']} ({close['dsp']/fpga['DSP']*100:.1f}%)")
            print(f"  Builds: 1 | Time: {close['dur']/3600:.2f} h")
        print("-" * 30)

def process_frt(csv_path):
    # xc7z045 capacity
    fpga = {'LUTs': 134600, 'FFs': 269200, 'BRAM': 365, 'DSP': 740}
    fps_targets = [3000, 6000, 9000, 12000]
    
    max_lut = fpga['LUTs'] * 0.50
    max_ff = fpga['FFs'] * 0.50
    max_bram = fpga['BRAM'] * 0.50
    max_dsp = fpga['DSP'] * 0.50
    
    runs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
             if row['status'] == 'success':
                runs.append({
                    'name': row['hw_name'],
                    'dur': float(row['duration_in_seconds']),
                    'lut': float(row['Total LUTs']),
                    'ff': float(row['FFs']),
                    'bram': float(row['BRAM (36k)']),
                    'dsp': float(row['DSP Blocks']),
                    'fps': float(row['estimated_throughput_fps'])
                })

    print("=== FRT Scenarios (CIFAR10_1W1A, 50% Area) ===")
    for fps_t in fps_targets:
        valid_run = None
        best_idx = -1
        
        # In FRT, we want the FIRST design (minimal builds) that meets FPS AND fits in 50% area
        # OR if multiple meet it, we could pick the one with lowest area? 
        # But usually we follow the greedy path.
        for i, r in enumerate(runs):
            if r['fps'] >= fps_t:
                if r['lut'] <= max_lut and r['ff'] <= max_ff and r['bram'] <= max_bram and r['dsp'] <= max_dsp:
                    valid_run = r
                    best_idx = i
                    break
        
        if valid_run:
            total_time = sum(r['dur'] for r in runs[:best_idx+1])
            builds = best_idx + 1
            print(f"Target FPS: {fps_t}")
            print(f"  Run: {valid_run['name']} | FPS: {valid_run['fps']:.2f}")
            print(f"  Resources: "
                  f"LUT={valid_run['lut']} ({valid_run['lut']/fpga['LUTs']*100:.1f}%), "
                  f"FF={valid_run['ff']} ({valid_run['ff']/fpga['FFs']*100:.1f}%), "
                  f"BRAM={valid_run['bram']} ({valid_run['bram']/fpga['BRAM']*100:.1f}%), "
                  f"DSP={valid_run['dsp']} ({valid_run['dsp']/fpga['DSP']*100:.1f}%)")
            print(f"  Builds: {builds} | Time: {total_time/3600:.2f} h")
        else:
            # If failed, pick the highest FPS design that fits in 50% area
            best_fit_run = None
            best_fit_idx = -1
            for i, r in enumerate(runs):
                if r['lut'] <= max_lut and r['ff'] <= max_ff and r['bram'] <= max_bram and r['dsp'] <= max_dsp:
                    if best_fit_run is None or r['fps'] > best_fit_run['fps']:
                        best_fit_run = r
                        best_fit_idx = i
            
            if best_fit_run:
                t_time = sum(r['dur'] for r in runs[:best_fit_idx+1])
                print(f"Target FPS: {fps_t} (FAILED, closest fit shown)")
                print(f"  Run: {best_fit_run['name']} | FPS: {best_fit_run['fps']:.2f}")
                print(f"  Resources: "
                      f"LUT={best_fit_run['lut']} ({best_fit_run['lut']/fpga['LUTs']*100:.1f}%), "
                      f"FF={best_fit_run['ff']} ({best_fit_run['ff']/fpga['FFs']*100:.1f}%), "
                      f"BRAM={best_fit_run['bram']} ({best_fit_run['bram']/fpga['BRAM']*100:.1f}%), "
                      f"DSP={best_fit_run['dsp']} ({best_fit_run['dsp']/fpga['DSP']*100:.1f}%)")
                print(f"  Builds: {best_fit_idx+1} | Time: {t_time/3600:.2f} h")
            else:
                print(f"Target FPS: {fps_t} (FAILED, nothing fits in 50%)")
        print("-" * 30)

def process_mp(csv_path, title, fpga):
    runs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
             if row['status'] == 'success':
                runs.append({
                    'name': row['hw_name'],
                    'dur': float(row['duration_in_seconds']),
                    'lut': float(row['Total LUTs']),
                    'ff': float(row['FFs']),
                    'bram': float(row['BRAM (36k)']),
                    'dsp': float(row['DSP Blocks']),
                    'fps': float(row['estimated_throughput_fps'])
                })

    print(f"=== MP Scenarios ({title}, 100% Area) ===")
    
    best_run = None
    best_idx = -1
    
    # In MP, we want the absolute highest FPS that fits within 100% area
    for i, r in enumerate(runs):
        if r['lut'] <= fpga['LUTs'] and r['ff'] <= fpga['FFs'] and r['bram'] <= fpga['BRAM'] and r['dsp'] <= fpga['DSP']:
            if best_run is None or r['fps'] > best_run['fps']:
                best_run = r
                best_idx = i
                
    if best_run:
        total_time = sum(r['dur'] for r in runs[:best_idx+1])
        builds = best_idx + 1
        print(f"Target: MAX FPS")
        print(f"  Run: {best_run['name']} | FPS: {best_run['fps']:.2f}")
        print(f"  Resources: "
              f"LUT={best_run['lut']} ({best_run['lut']/fpga['LUTs']*100:.1f}%), "
              f"FF={best_run['ff']} ({best_run['ff']/fpga['FFs']*100:.1f}%), "
              f"BRAM={best_run['bram']} ({best_run['bram']/fpga['BRAM']*100:.1f}%), "
              f"DSP={best_run['dsp']} ({best_run['dsp']/fpga['DSP']*100:.1f}%)")
        print(f"  Builds: {builds} | Time: {total_time/3600:.2f} h")
    else:
        print(f"Target: MAX FPS (FAILED, nothing fits in 100%)")
    print("-" * 30)

print("Starting analysis...\n")
summary_sat = '/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds/SAT6_T2W2_2026-04-14_02-01-55/hardware_summary.csv'
summary_cifar = '/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds/CIFAR10_1W1A_2026-04-12_02-06-38/hardware_summary.csv'

fpga_sat = {'LUTs': 53200, 'FFs': 106400, 'BRAM': 140, 'DSP': 220}
fpga_cifar = {'LUTs': 134600, 'FFs': 269200, 'BRAM': 365, 'DSP': 740}

try:
    process_sec(summary_sat)
except Exception as e:
    print("Could not process SAT6 SEC:", e)

print("\n")
try:
    process_frt(summary_cifar)
except Exception as e:
    print("Could not process CIFAR10 FRT:", e)

print("\n")
try:
    process_mp(summary_sat, "SAT6_T2W2", fpga_sat)
except Exception as e:
    print("Could not process SAT6 MP:", e)

print("\n")
try:
    process_mp(summary_cifar, "CIFAR10_1W1A", fpga_cifar)
except Exception as e:
    print("Could not process CIFAR10 MP:", e)
