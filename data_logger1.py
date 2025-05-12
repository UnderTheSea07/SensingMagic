"""
ReSkin 4×MLX90393 logger
────────────────────────
• 4 subplots (sensor-0 … sensor-3) in a 2×2 grid
• Solid RGB lines:  X=red, Y=green, Z=blue
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
    port = ap.parse_args().port or find_port()
    duration = ap.parse_args().duration
    cal_mode = ap.parse_args().cal
    
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
    fn = pathlib.Path(f"log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    writer = csv.writer(fn.open("w", newline=''))
    writer.writerow(["ms"] + [f"S{s}_{q}" for s in range(4) for q in "XYZ"])
    print(f"CSV file created: {fn}")

    # Figure set-up: 4 panels sharing axes
    print("Setting up matplotlib figure...")
    plt.style.use('seaborn-v0_8')  # Modern, clean style
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8), dpi=100)
    fig.suptitle(f"ReSkin Magnetic Sensor Readings (Session started at {session_start_time})", fontsize=16, weight='bold')
    axes = axes.flatten()
    
    # Set initial y-axis limits based on mode
    if cal_mode:
        y_range = (-300, 300)  # Wider range for calibration
    else:
        y_range = (-150, 150)  # Normal range for measurements
        
    for s, ax in enumerate(axes):
        ax.set_title(f"Sensor {s}", fontsize=14, weight='bold')
        ax.set_ylabel("Magnetic flux density (µT)", fontsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_facecolor('#f9f9f9')  # Light gray background
        # Use integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Add minor ticks for more precise readings
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Set initial y-axis limits to reasonable values for magnetic sensors
        ax.set_ylim(*y_range)
    
    # Add X-axis label for time to the bottom plots
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[3].set_xlabel("Time (s)", fontsize=12)

    xs, ys = [], [[[], [], []] for _ in range(4)]         # 4 sensors × 3 axes
    lines = []
    for s, ax in enumerate(axes):
        for i, q in enumerate("XYZ"):
            (ln,) = ax.plot([], [], color=COL[q], linewidth=2.5, label=q, alpha=0.95, 
                           markersize=3, markevery=5)  # Add markers every 5 points
            lines.append(ln)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    print("Plot setup complete")
    
    # Set initial x-axis range to make it clearer it's time
    for ax in axes:
        ax.set_xlim(0, 60)  # Initial view of 60 seconds, will auto-update
        ax.tick_params(axis='both', which='major', labelsize=10)

    # ── animation callback ────────────────────────────────────────────
    def update(_):
        nonlocal xs
        data_received = False
        
        # Check if we've reached the duration limit
        if duration > 0 and time.time() - start_time > duration:
            print(f"Reached duration limit of {duration} seconds")
            plt.savefig(f"plot_{time.strftime('%Y%m%d_%H%M%S')}.png", dpi=150)
            ser.write(b's')
            plt.close()
            return lines

        print(f"Update called, buffer: {ser.in_waiting} bytes")
        while ser.in_waiting:
            raw = ser.readline().strip()
            print(f"Raw data: {raw}")
            if not raw or raw.startswith(b'#'):
                continue
            data_received = True
            
            # Split the data and clean it
            tok = raw.split()
            print(f"Got {len(tok)} values: {tok[:5]}...")
            
            # If we have 20 values instead of 17, we'll adapt
            if len(tok) == 20:
                # Use only the first timestamp and the X,Y,Z values from each sensor (skip temperature)
                try:
                    timestamp = float(tok[0])
                    x_vals = [float(tok[i]) for i in [2, 7, 12, 17]]  # X values at positions 2,7,12,17
                    y_vals = [float(tok[i]) for i in [3, 8, 13, 18]]  # Y values at positions 3,8,13,18
                    z_vals = [float(tok[i]) for i in [4, 9, 14, 19]]  # Z values at positions 4,9,14,19
                    
                    # Calculate relative time since start
                    t_s = (time.time() - start_time)
                    xs.append(t_s)
                    
                    csv_row = [timestamp]
                    for s in range(4):
                        # Apply PROPER scaling - these are raw values that need to be converted to µT
                        # using the GAIN5_LSB constant (0.161 µT per LSB)
                        x = x_vals[s] * GAIN5_LSB
                        y = y_vals[s] * GAIN5_LSB
                        z = z_vals[s] * GAIN5_LSB
                        
                        ys[s][0].append(x)
                        ys[s][1].append(y)
                        ys[s][2].append(z)
                        
                        # Store raw values in CSV
                        csv_row.extend([x_vals[s], y_vals[s], z_vals[s]])
                    
                    writer.writerow(csv_row)
                    print(f"Added data point at time {t_s:.2f}s")
                    
                    # Make sure we only keep a reasonable number of points to prevent memory issues
                    if len(xs) > 2000:  # Keep the most recent 2000 points
                        # Downsample old data to reduce memory usage
                        if len(xs) > 5000:
                            # Keep only every 2nd point for older data
                            downsampled_xs = xs[:1000:2] + xs[1000:]
                            xs = downsampled_xs
                            for s in range(4):
                                for a in range(3):
                                    ys[s][a] = ys[s][a][:1000:2] + ys[s][a][1000:]
                        else:
                            # Just trim to the most recent 2000 points
                            xs = xs[-2000:]
                            for s in range(4):
                                for a in range(3):
                                    ys[s][a] = ys[s][a][-2000:]
                                    
                    continue
                except ValueError as e:
                    print(f"Error parsing values: {e}")
                    print(f"Token values (first 5): {tok[:5]}")
                    continue
            
            # Original processing for 17 values format
            if len(tok) != 17:             # ms + 4×(T X Y Z) = 17 values
                print(f"WARNING: Expected 17 values, got {len(tok)}")
                continue
                
            try:
                vals = [float(x) for x in tok]
            except ValueError as e:
                print(f"Error parsing values: {e}")
                print(f"Token values (first 5): {tok[:5]}")
                continue

            # Calculate relative time since start
            t_s = (time.time() - start_time)
            xs.append(t_s)

            csv_row = [vals[0]]
            ptr = 1                        # points to T of sensor-0
            for s in range(4):
                # skip temperature, take X Y Z, convert → µT
                # Make sure we're properly scaling to µT
                x = vals[ptr + 1] * GAIN5_LSB  # Apply proper scaling
                y = vals[ptr + 2] * GAIN5_LSB
                z = vals[ptr + 3] * GAIN5_LSB
                
                ys[s][0].append(x); ys[s][1].append(y); ys[s][2].append(z)
                csv_row.extend([vals[ptr + 1], vals[ptr + 2], vals[ptr + 3]])
                ptr += 4                   # jump to next sensor's T

            writer.writerow(csv_row)
            print(f"Added data point at time {t_s:.2f}s")
            
            # Make sure we only keep a reasonable number of points to prevent memory issues
            if len(xs) > 2000:  # Keep the most recent 2000 points
                # Downsample old data to reduce memory usage
                if len(xs) > 5000:
                    # Keep only every 2nd point for older data
                    downsampled_xs = xs[:1000:2] + xs[1000:]
                    xs = downsampled_xs
                    for s in range(4):
                        for a in range(3):
                            ys[s][a] = ys[s][a][:1000:2] + ys[s][a][1000:]
                else:
                    # Just trim to the most recent 2000 points
                    xs = xs[-2000:]
                    for s in range(4):
                        for a in range(3):
                            ys[s][a] = ys[s][a][-2000:]
            
        if not data_received:
            print("No data received in this update")
        
        # Update the interval based on the number of data points
        global anim
        if len(xs) > 300:
            anim.event_source.interval = 500  # Slower updates when we have lots of data
        
        # redraw
        idx = 0
        for s, ax in enumerate(axes):
            for a in range(3):
                lines[idx].set_data(xs, ys[s][a])
                
                # Update marker style - show markers only on recent data for clarity
                if len(xs) > 20:  # If we have enough data points
                    # Set marker only on recent points
                    if a == 0:  # X axis
                        lines[idx].set_marker('o')
                        lines[idx].set_markevery(slice(-20, None, 4))  # Mark recent points
                    elif a == 1:  # Y axis
                        lines[idx].set_marker('s')  # Square markers
                        lines[idx].set_markevery(slice(-20, None, 4))
                    else:  # Z axis
                        lines[idx].set_marker('^')  # Triangle markers
                        lines[idx].set_markevery(slice(-20, None, 4))
                
                idx += 1
            
            # Make sure we recompute the data limits
            ax.relim()
            
            # Adjust the x-axis limit to follow the data
            if xs:
                current_max = max(xs)
                window_size = 30  # Show last 30 seconds of data
                min_x = max(0, current_max - window_size)
                # Make sure x-axis always starts from a nice round number
                min_x = max(0, int(min_x))
                ax.set_xlim(min_x, current_max + 5)
                
                # Calculate good y-limits based on actual data (only use recent data for scale)
                if len(ys[s][0]) > 5:  # Make sure we have enough data points
                    window_size = min(100, len(ys[s][0]))  # Last 100 points or all if less
                    
                    # Calculate min/max across all axes excluding outliers
                    all_data = []
                    for a in range(3):
                        all_data.extend(ys[s][a][-window_size:])
                    
                    if all_data:
                        # Filter out extreme outliers (outside 3 std deviations)
                        mean_val = np.mean(all_data)
                        std_val = np.std(all_data)
                        filtered_data = [val for val in all_data if abs(val - mean_val) < 3 * std_val]
                        
                        if filtered_data:
                            data_min = min(filtered_data)
                            data_max = max(filtered_data)
                            
                            # Add padding (at least 20 µT or 20% of range)
                            padding = max(20, 0.2 * (data_max - data_min))
                            
                            # Set limits within reasonable bounds for magnetic sensors
                            if cal_mode:
                                # Wider range for calibration
                                new_ymin = max(-300, min(-50, data_min - padding))
                                new_ymax = min(300, max(50, data_max + padding))
                            else:
                                # Normal range for measurements
                                new_ymin = max(-150, min(-20, data_min - padding))
                                new_ymax = min(150, max(20, data_max + padding))
                            
                            # Only update if needed to prevent constant rescaling
                            current_ymin, current_ymax = ax.get_ylim()
                            if (new_ymin < current_ymin) or (new_ymax > current_ymax) or (current_ymax - current_ymin > 3 * (data_max - data_min + 2*padding)):
                                ax.set_ylim(new_ymin, new_ymax)
            
            # Apply autoscale for better visualization, but only on Y axis
            # Use tight=True for better fit to data
            ax.autoscale_view(scalex=False, scaley=True, tight=True)
            
        # Add timestamp and recording info to the plot
        elapsed = int(time.time() - start_time)
        duration_str = f"{elapsed}s" if duration == 0 else f"{elapsed}/{duration}s"
        plt.figtext(0.01, 0.01, f"Recording: {duration_str}", 
                   horizontalalignment='left', fontsize=9, alpha=0.7)
        plt.figtext(0.99, 0.01, f"Time: {time.strftime('%H:%M:%S')}", 
                   horizontalalignment='right', fontsize=9, alpha=0.7)
                
        # Make sure entire figure is updated with proper spacing
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for the title
        
        return lines

    print("Starting animation...")
    global anim                           # prevent garbage-collection :contentReference[oaicite:4]{index=4}
    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    print("Showing plot window...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for the title
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Plot closed by user")
    finally:
        # Save the plot on exit
        plt.savefig(f"plot_{time.strftime('%Y%m%d_%H%M%S')}.png", dpi=150)
        ser.write(b's'); ser.close()
        print("Run saved to", fn)

if __name__ == "__main__":
    main()
