"""
ReSkin MLX90393 Single Sensor Logger
────────────────────────
• Single plot for one sensor
• Solid RGB lines: X=red, Y=green, Z=blue
• Magnetic field in µT (0.161 µT/LSB @ Gain 5)
• Temperature omitted for clarity
• Auto-detects Arduino port if none given
"""

import csv, time, pathlib, argparse
import serial as pyserial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from serial.tools import list_ports

# Sensor calibration constant - µT per raw LSB for MLX90393 at Gain 5
GAIN5_LSB = 0.161  # µT per raw LSB

# Setup color palette with better visibility
COL = {'X': '#e41a1c', 'Y': '#4daf4a', 'Z': '#377eb8'}  # Colorblind-friendly palette

# ── helper ─────────────────────────────────────────────────────────────
def find_port() -> str:
    print("Available ports:")
    for p in list_ports.comports():
        print(f"  - {p.device}: {p.description}")
        if 'usb' in p.device.lower():
            return p.device
    raise SystemExit("No USB device found. Please connect your Arduino and try again.")

# ── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", help="COM7 /dev/ttyACM0 … (blank = auto)")
    ap.add_argument("--duration", type=int, default=0, help="Recording duration in seconds (0 = unlimited)")
    ap.add_argument("--cal", action="store_true", help="Enable calibration mode with wider Y-axis range")
    ap.add_argument("--sensor", type=int, default=0, help="Sensor number to monitor (0-3)")
    port = ap.parse_args().port or find_port()
    duration = ap.parse_args().duration
    cal_mode = ap.parse_args().cal
    sensor_num = ap.parse_args().sensor
    
    if not 0 <= sensor_num <= 3:
        raise SystemExit("Sensor number must be between 0 and 3")
    
    print("Opening", port)
    try:
        ser = pyserial.Serial(port, 115200, timeout=1)
        print("Serial port opened successfully")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    # Wait for sketch banner then start stream
    print("Waiting for Arduino sketch banner...")
    timeout_start = time.time()
    banner_found = False
    while time.time() - timeout_start < 5:  # 5 second timeout
        line = ser.readline()
        print(f"Received: {line}")
        if line[:1] == b'#':
            banner_found = True
            break

    if not banner_found:
        print("WARNING: Didn't receive expected banner from Arduino")
        print("Attempting to continue anyway...")
    
    print("Sending 'g' command to start data stream")
    ser.write(b'g')

    # Record start time for relative timestamps
    start_time = time.time()
    session_start_time = time.strftime("%H:%M:%S")

    # CSV output
    fn = pathlib.Path(f"log_sensor{sensor_num}_{time.strftime('%Y%m%d_%H%M%S')}_杜邦纸_05.csv")
    writer = csv.writer(fn.open("w", newline=''))
    writer.writerow(["ms", "X", "Y", "Z"])
    print(f"CSV file created: {fn}")

    # Figure set-up: single panel
    print("Setting up matplotlib figure...")
    plt.style.use('seaborn-v0_8')  # Modern, clean style
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    fig.suptitle(f"ReSkin Sensor {sensor_num} Magnetic Readings (Session started at {session_start_time})", 
                 fontsize=16, weight='bold')
    
    # Set initial y-axis limits based on mode
    if cal_mode:
        y_range = (-300, 300)  # Wider range for calibration
    else:
        y_range = (-150, 150)  # Normal range for measurements
        
    ax.set_title(f"Sensor {sensor_num}", fontsize=14, weight='bold')
    ax.set_ylabel("Magnetic flux density (µT)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_facecolor('#f9f9f9')  # Light gray background
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim(*y_range)
    ax.tick_params(axis='both', which='major', labelsize=10)

    xs, ys = [], [[], [], []]  # Single sensor × 3 axes
    lines = []
    for i, q in enumerate("XYZ"):
        (ln,) = ax.plot([], [], color=COL[q], linewidth=2.5, label=q, alpha=0.95, 
                       markersize=3, markevery=5)
        lines.append(ln)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    print("Plot setup complete")
    
    # Set initial x-axis range
    ax.set_xlim(0, 60)  # Initial view of 60 seconds

    # ── animation callback ────────────────────────────────────────────
    def update(_):
        nonlocal xs
        data_received = False
        
        # Check if we've reached the duration limit
        if duration > 0 and time.time() - start_time > duration:
            print(f"Reached duration limit of {duration} seconds")
            plt.savefig(f"plot_sensor{sensor_num}_{time.strftime('%Y%m%d_%H%M%S')}.png", dpi=150)
            ser.write(b's')
            plt.close()
            return lines

        while ser.in_waiting:
            raw = ser.readline().strip()
            if not raw or raw.startswith(b'#'):
                continue
            data_received = True
            
            # Split the data and clean it
            tok = raw.split()
            
            # Handle 20-value format
            if len(tok) == 20:
                try:
                    timestamp = float(tok[0])
                    # Get values for the selected sensor
                    base_idx = sensor_num * 5  # Each sensor takes 5 values
                    x = float(tok[base_idx + 2]) * GAIN5_LSB
                    y = float(tok[base_idx + 3]) * GAIN5_LSB
                    z = float(tok[base_idx + 4]) * GAIN5_LSB
                    
                    t_s = (time.time() - start_time)
                    xs.append(t_s)
                    
                    ys[0].append(x)
                    ys[1].append(y)
                    ys[2].append(z)
                    
                    writer.writerow([timestamp, x/GAIN5_LSB, y/GAIN5_LSB, z/GAIN5_LSB])
                    
                    # Manage data points
                    if len(xs) > 2000:
                        if len(xs) > 5000:
                            xs = xs[:1000:2] + xs[1000:]
                            for a in range(3):
                                ys[a] = ys[a][:1000:2] + ys[a][1000:]
                        else:
                            xs = xs[-2000:]
                            for a in range(3):
                                ys[a] = ys[a][-2000:]
                    
                    continue
                except (ValueError, IndexError) as e:
                    print(f"Error parsing values: {e}")
                    continue
            
            # Handle 17-value format
            if len(tok) == 17:
                try:
                    vals = [float(x) for x in tok]
                    t_s = (time.time() - start_time)
                    xs.append(t_s)
                    
                    # Get values for the selected sensor
                    ptr = 1 + sensor_num * 4  # Skip timestamp and previous sensors
                    x = vals[ptr + 1] * GAIN5_LSB
                    y = vals[ptr + 2] * GAIN5_LSB
                    z = vals[ptr + 3] * GAIN5_LSB
                    
                    ys[0].append(x)
                    ys[1].append(y)
                    ys[2].append(z)
                    
                    writer.writerow([vals[0], x/GAIN5_LSB, y/GAIN5_LSB, z/GAIN5_LSB])
                    
                    # Manage data points
                    if len(xs) > 2000:
                        if len(xs) > 5000:
                            xs = xs[:1000:2] + xs[1000:]
                            for a in range(3):
                                ys[a] = ys[a][:1000:2] + ys[a][1000:]
                        else:
                            xs = xs[-2000:]
                            for a in range(3):
                                ys[a] = ys[a][-2000:]
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing values: {e}")
                    continue
        
        # Update plot
        for i, line in enumerate(lines):
            line.set_data(xs, ys[i])
            
            # Update marker style for recent data
            if len(xs) > 20:
                if i == 0:  # X axis
                    line.set_marker('o')
                    line.set_markevery(slice(-20, None, 4))
                elif i == 1:  # Y axis
                    line.set_marker('s')
                    line.set_markevery(slice(-20, None, 4))
                else:  # Z axis
                    line.set_marker('^')
                    line.set_markevery(slice(-20, None, 4))
        
        # Update axes
        if xs:
            current_max = max(xs)
            window_size = 30  # Show last 30 seconds
            min_x = max(0, int(current_max - window_size))
            ax.set_xlim(min_x, current_max + 5)
            
            # Update y-axis limits
            if len(ys[0]) > 5:
                window_size = min(100, len(ys[0]))
                all_data = []
                for a in range(3):
                    all_data.extend(ys[a][-window_size:])
                
                if all_data:
                    mean_val = np.mean(all_data)
                    std_val = np.std(all_data)
                    filtered_data = [val for val in all_data if abs(val - mean_val) < 3 * std_val]
                    
                    if filtered_data:
                        data_min = min(filtered_data)
                        data_max = max(filtered_data)
                        padding = max(20, 0.2 * (data_max - data_min))
                        
                        if cal_mode:
                            new_ymin = max(-300, min(-50, data_min - padding))
                            new_ymax = min(300, max(50, data_max + padding))
                        else:
                            new_ymin = max(-150, min(-20, data_min - padding))
                            new_ymax = min(150, max(20, data_max + padding))
                        
                        current_ymin, current_ymax = ax.get_ylim()
                        if (new_ymin < current_ymin) or (new_ymax > current_ymax) or (current_ymax - current_ymin > 3 * (data_max - data_min + 2*padding)):
                            ax.set_ylim(new_ymin, new_ymax)
        
        # Add timestamp and recording info
        elapsed = int(time.time() - start_time)
        duration_str = f"{elapsed}s" if duration == 0 else f"{elapsed}/{duration}s"
        plt.figtext(0.01, 0.01, f"Recording: {duration_str}", 
                   horizontalalignment='left', fontsize=9, alpha=0.7)
        plt.figtext(0.99, 0.01, f"Time: {time.strftime('%H:%M:%S')}", 
                   horizontalalignment='right', fontsize=9, alpha=0.7)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return lines

    print("Starting animation...")
    global anim
    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    print("Showing plot window...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Plot closed by user")
    finally:
        plt.savefig(f"plot_sensor{sensor_num}_{time.strftime('%Y%m%d_%H%M%S')}.png", dpi=150)
        ser.write(b's')
        ser.close()
        print("Run saved to", fn)

if __name__ == "__main__":
    main() 