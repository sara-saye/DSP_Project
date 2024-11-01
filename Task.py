import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline
# Quantization Test
def QuantizationTest1( Your_EncodedValues, Your_QuantizedValues):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_name = filedialog.askopenfilename(title="Select Signal File")
    if not file_name:
        print("No file selected.")
        return
    expectedEncodedValues = []
    expectedQuantizedValues = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V2 = str(L[0])
                V3 = float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break

    if ((len(Your_EncodedValues) != len(expectedEncodedValues)) or (
            len(Your_QuantizedValues) != len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                "QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
            return
    print("QuantizationTest1 Test case passed successfully")
def QuantizationTest2(Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_name = filedialog.askopenfilename(title="Select Signal File")
    if not file_name:
        print("No file selected.")
        return
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 4:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if (len(Your_IntervalIndices) != len(expectedIntervalIndices)
            or len(Your_EncodedValues) != len(expectedEncodedValues)
            or len(Your_QuantizedValues) != len(expectedQuantizedValues)
            or len(Your_SampledError) != len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if (Your_IntervalIndices[i] != expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one")
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
            return
    print("QuantizationTest2 Test case passed successfully")
# read signal from signal1.txt and ignoring the first 3 rows and read only the signal values
# generate sin/cos waves and t=(n/Fs) is x-axis and signal is y-axis
def generate_signal(signal_type, amplitude, phase_shift, f_analog, f_sampling, duration=1):
    num_samples = int(f_sampling * duration)
    samples = np.arange(num_samples)
    t = samples / f_sampling
    if signal_type == "sine":
        signal = amplitude * np.sin(2 * np.pi * f_analog * t + phase_shift)
    elif signal_type == "cosine":
        signal = amplitude * np.cos(2 * np.pi * f_analog * t + phase_shift)
    return t, signal
def SignalSamplesAreEqual(indices, samples):
    try:
        # Open a file dialog to select the comparison file
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        file_path = filedialog.askopenfilename(title="Select the file to compare",filetypes=[("Text files", ".txt"), ("All files", ".*")])

        if not file_path:
            print("No file selected.")
            return

        # Read expected indices and samples from the selected file
        expected_indices = []
        expected_samples = []
        with open(file_path, 'r') as f:
            # Skipping the first 3 lines (headers)
            for _ in range(4):
                line = f.readline()

            # Reading the signal values
            while line:
                L = line.strip()
                if len(L.split()) == 2:
                    V1, V2 = L.split()
                    expected_indices.append(int(V1))
                    expected_samples.append(float(V2))
                line = f.readline()

        # Compare lengths
        if len(expected_samples) != len(samples):
            print("Test case failed, your signal has a different length from the expected one.")
            return

        # Compare values
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) >= 0.01:
                print("Test case failed, your signal has different values from the expected one.")
                return

        print("Test case passed successfully!")

    except Exception as e:
        print(f"Error: {e}")

class SignalPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Displaying Signals")

        # Get the screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate 80% of the screen size
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        # Set the geometry of the window to 80% of the screen size
        self.root.geometry(f'{window_width}x{window_height}')

        self.button_frame = tk.Frame(self.root, bg="#f2ede9")
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create buttons with specified width and padding
        button_width = 15  # Set a fixed width for buttons

        self.discrete_button = tk.Button(self.button_frame, text="Read Signal", command=self.plot_both_signals,
                                         bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.discrete_button.grid(row=0, column=0, padx=10, pady=10)

        self.generate_button = tk.Button(self.button_frame, text="Generate Signal", command=self.open_generate_window,
                                         bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.generate_button.grid(row=0, column=1, padx=10, pady=10)

        self.add_button = tk.Button(self.button_frame, text="Add Signals", command=self.add_signals, bg="#ccbeb1",
                                    fg="black", font=7, width=button_width)
        self.add_button.grid(row=0, column=2, padx=10, pady=10)

        self.subtract_button = tk.Button(self.button_frame, text="Subtract Signals", command=self.subtract_signals,
                                         bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.subtract_button.grid(row=0, column=3, padx=10, pady=10)

        self.multiply_button = tk.Button(self.button_frame, text="Multiply Signal", command=self.multiply_signal,
                                         bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.multiply_button.grid(row=0, column=4, padx=10, pady=10)

        self.square_button = tk.Button(self.button_frame, text="Square Signal", command=self.square_signal,
                                       bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.square_button.grid(row=0, column=5, padx=10, pady=10)

        self.normalize_button = tk.Button(self.button_frame, text="Normalize Signal", command=self.normalize_signal,
                                          bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.normalize_button.grid(row=1, column=0, padx=10, pady=10)

        self.accumulate_button = tk.Button(self.button_frame, text="Accumulate Signal", command=self.accumulate_signal,
                                           bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.accumulate_button.grid(row=1, column=1, padx=10, pady=10)

        self.quantize_button = tk.Button(self.button_frame, text="Quantize Signal", command=self.quantize_signal,
                                         bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.quantize_button.grid(row=1, column=2, padx=10, pady=10)

        # Add Frequency Domain button
        self.frequency_button = tk.Button(self.button_frame, text="Frequency Domain",
                                          command=self.show_frequency_options,
                                          bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.frequency_button.grid(row=1, column=3, padx=10, pady=10)

        # Ensure that the frame can grow and that buttons will stay centered
        self.button_frame.pack_propagate(False)

        self.canvas = None
    def read_signals_from_txt_files(self):
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the root window

            # Allow selection of one or more files
            file_paths = filedialog.askopenfilenames(title="Select one or more files",filetypes=[("Text files", ".txt"), ("All files", ".*")])

            if not file_paths:
                print("No file selected.")
                return None, None

            signals = []
            all_indices = []

            for file_path in file_paths:
                # Load both indices (first column) and signal values (second column)
                data = np.loadtxt(file_path, skiprows=3, usecols=(0, 1))
                indices = data[:, 0]  # First column for indices
                signal = data[:, 1]  # Second column for signal values
                all_indices.append(indices)
                signals.append(signal)

            if signals:
                # Merge all indices, ensuring that we have a union of all index points
                unified_indices = np.unique(np.concatenate(all_indices))

                # Pad and align each signal to the unified set of indices
                aligned_signals = []
                for i, indices in enumerate(all_indices):
                    # Find where the original indices fit in the unified indices
                    aligned_signal = np.zeros_like(unified_indices, dtype=float)
                    idx_in_unified = np.searchsorted(unified_indices, indices)
                    aligned_signal[idx_in_unified] = signals[i]
                    aligned_signals.append(aligned_signal)

                return unified_indices, aligned_signals  # Return the aligned signals

            return None, None

        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None

    def plot_signal(self, indices, signal, title):
        fig = plt.Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Amplitude')
        ax.plot(indices, signal, 'b')
        ax.set_title(title)
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_both_signals(self):
        indices, signals = self.read_signals_from_txt_files()
        if signals:
            fig = plt.Figure(figsize=(10, 8))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            # Plot the first signal using spline interpolation for smooth plotting
            signal1 = signals[0]
            smooth_x = np.linspace(indices.min(), indices.max(), 500)
            spl = make_interp_spline(indices, signal1, k=3)
            smooth_y = spl(smooth_x)
            ax1.plot(smooth_x, smooth_y, 'b')
            ax1.set_title('Analog Signal')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True)

            # Plot the discrete signal
            ax2.stem(indices, signal1, linefmt='r-', markerfmt='ro', basefmt='gray')
            ax2.set_title('Discrete Signal')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True)

            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            self.canvas = FigureCanvasTkAgg(fig, master=self.root)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def open_generate_window(self):
        self.generate_window = tk.Toplevel(self.root)
        self.generate_window.title("Generate Signal")
        tk.Label(self.generate_window, text="Signal Type (sine/cosine):").grid(row=0, column=0, padx=10, pady=5)
        self.signal_type_entry = tk.Entry(self.generate_window)
        self.signal_type_entry.grid(row=0, column=1, padx=10, pady=5)
        tk.Label(self.generate_window, text="Amplitude:").grid(row=1, column=0, padx=10, pady=5)
        self.amplitude_entry = tk.Entry(self.generate_window)
        self.amplitude_entry.grid(row=1, column=1, padx=10, pady=5)
        tk.Label(self.generate_window, text="Phase Shift (radians):").grid(row=2, column=0, padx=10, pady=5)
        self.phase_shift_entry = tk.Entry(self.generate_window)
        self.phase_shift_entry.grid(row=2, column=1, padx=10, pady=5)
        tk.Label(self.generate_window, text=" Analog Frequency (Hz):").grid(row=3, column=0, padx=10, pady=5)
        self.frequency_entry = tk.Entry(self.generate_window)
        self.frequency_entry.grid(row=3, column=1, padx=10, pady=5)
        tk.Label(self.generate_window, text="Sampling Frequency (Hz):").grid(row=4, column=0, padx=10, pady=5)
        self.sampling_frequency_entry = tk.Entry(self.generate_window)
        self.sampling_frequency_entry.grid(row=4, column=1, padx=10, pady=5)
        generate_button = tk.Button(self.generate_window, text="Generate", command=self.generate_user_signal)
        generate_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    def square_signal(self):
        indices, signals =self.read_signals_from_txt_files()
        # Check if signals is None or not a list or contains more than one signal
        if signals is None or not isinstance(signals, list) or len(signals) != 1:
            messagebox.showerror("Invalid", "Please select exactly one signal file.")
            return
        # Convert to NumPy array for processing
        signal = np.array(signals[0])  # Get the single signal

        # Check if the signal is numeric
        if not np.issubdtype(signal.dtype, np.number):
            messagebox.showerror("Invalid", "The selected signal is not numeric.")
            return

        try:
            result = np.square(signal)
            SignalSamplesAreEqual(indices, result)
            self.plot_signal(indices, result, title="Squared Signal")
        except ValueError as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def normalize_signal(self):
        indices, signals = self.read_signals_from_txt_files()

        # Check if we have one signal
        if signals is None or not isinstance(signals, list) or len(signals) != 1:
            messagebox.showerror("Invalid", "Please select exactly one signal file.")
            return

        signal = np.array(signals[0])  # Convert to NumPy array for processing

        # Check if the signal is numeric
        if not np.issubdtype(signal.dtype, np.number):
            messagebox.showerror("Invalid", "The selected signal is not numeric.")
            return

        try:
            normalization_type = self.get_user_input("Normalize to -1 to 1 or 0 to 1? (Enter '-1' or '0')").strip()

            # Check for zero division
            signal_min = np.min(signal)
            signal_max = np.max(signal)
            range_signal = signal_max - signal_min

            if range_signal == 0:
                messagebox.showerror("Invalid", "Cannot normalize a constant signal.")
                return

            if normalization_type == "-1":
                result = 2 * (signal - signal_min) / range_signal - 1
            elif normalization_type == "0":
                result = (signal - signal_min) / range_signal
            else:
                messagebox.showerror("Invalid", "Normalization type must be '-1' or '0'.")
                return

            SignalSamplesAreEqual(indices, result)
            self.plot_signal(indices, result, title=f"Normalized Signal ({normalization_type} to 1)")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def accumulate_signal(self):
        indices, signals =self.read_signals_from_txt_files()

        # Ensure that exactly one signal is selected
        if signals is None or len(signals) != 1:
            messagebox.showerror("Input Error", "Please select exactly one signal file for accumulation.")
            return

        signal = np.array(signals[0])  # Get the first (and only) signal

        # Compute the cumulative sum of the signal
        result = custom_cumsum(signal)

        # Save the result and plot it
        SignalSamplesAreEqual(indices, result)
        self.plot_signal(indices, result, title="Accumulated Signal")

    def generate_user_signal(self):
        try:
            signal_type = self.signal_type_entry.get()
            amplitude = float(self.amplitude_entry.get())
            phase_shift = float(self.phase_shift_entry.get())
            f_analog = float(self.frequency_entry.get())
            f_sampling = float(self.sampling_frequency_entry.get())
            t, signal = generate_signal(signal_type, amplitude, phase_shift, f_analog, f_sampling)
            if signal_type == "sine":
                SignalSamplesAreEqual(t, signal)
            elif signal_type == "cosine":
                SignalSamplesAreEqual(t, signal)
            self.plot_signal(t, signal, title="generated signal")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for the signal parameters.")

    def add_signals(self):
        indices, signals = self.read_signals_from_txt_files()
        if signals:
            result = np.sum(signals, axis=0)
            SignalSamplesAreEqual(indices, result)
            self.plot_signal(indices, result, title="Added Signals")

    def subtract_signals(self):
        # Read signals from two different files
        indices, signals = self.read_signals_from_txt_files()

        # Check if exactly two signals are selected
        if signals is None or len(signals) != 2:
            messagebox.showerror("Input Error", "Please select exactly two signal files for subtraction.")
            return

        # Print the signals to check if they are being read correctly

        # Subtract the second signal from the first
        result = np.abs(signals[0] - signals[1])

        # Optionally, compare with an expected result if applicable
        SignalSamplesAreEqual(indices, result)

        # Plot the result of subtraction
        self.plot_signal(indices, result, title="Subtracted Signals")

    def multiply_signal(self):
        indices, signals =self.read_signals_from_txt_files()

        # Ensure only one signal file is selected
        if signals is None or (isinstance(signals, list) and len(signals) != 1):
            messagebox.showerror("Input Error", "Please select exactly one signal file for multiplication.")
            return

        try:
            # Get the multiplier input from the user
            signal = np.array(signals[0])  # Use the first (and only) signal

            # Get the multiplier input from the user
            multiplier = float(self.get_user_input("Enter a multiplier: "))
            # Multiply the signal by the multiplier
            result = signal * multiplier
            # Save the result and plot it
            SignalSamplesAreEqual(indices, result)
            self.plot_signal(indices, result, title=f"Signal Multiplied by {multiplier}")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for the multiplier.")

    def get_user_input(self, prompt):
        input_window = tk.Toplevel(self.root)
        input_window.title("Input")
        input_window.geometry('300x100')

        tk.Label(input_window, text=prompt).pack(pady=10)
        user_input = tk.Entry(input_window)
        user_input.pack(pady=10)

        # Use a variable to store the user input
        result = [None]  # Using a list to allow modification inside the submit_input function

        def submit_input():
            result[0] = user_input.get()  # Get the input value
            input_window.destroy()  # Close the input window

        tk.Button(input_window, text="Submit", command=submit_input).pack()

        # Wait until the input window is closed
        self.root.wait_window(input_window)

        return result[0]  # Return the stored input value
        # Quantization function

    def quantize_signal(self):
        # Step 1: Read the input signal and indices
        indices, signals = self.read_signals_from_txt_files()
        if signals is None or len(signals) != 1:
            messagebox.showerror("Input Error", "Please select exactly one signal file for quantization.")
            return
        signal = signals[0]  # Get the first (and only) signal

        try:
            # Step 2: Ask user for quantization method (levels or bits)
            quantization_method = self.get_user_input(
                "Quantize using levels or bits? (Enter 'levels' or 'bits'):").strip().lower()
            if quantization_method == "bits":
                num_bits = int(self.get_user_input("Enter the number of bits:"))
                num_levels = 2 ** num_bits  # Calculate levels from bits
                # Perform quantization and automatically compare with QuantizationTest1
                self.quantize_and_test(signal, indices, num_levels, is_bits=True)

            elif quantization_method == "levels":
                num_levels = int(self.get_user_input("Enter the number of quantization levels:"))
                # Perform quantization and automatically compare with QuantizationTest2
                self.quantize_and_test(signal, indices, num_levels, is_bits=False)

            else:
                messagebox.showerror("Input Error", "Invalid input. Please enter 'levels' or 'bits'.")
                return

        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for the levels or bits.")

    def quantize_and_test(self, signal, indices, num_levels, is_bits):
        # Perform quantization
        quantized_signal, quantization_error, encoded_indices, interval_indices = self.perform_quantization(signal,
                                                                                                            num_levels)

        # Plot the original and quantized signals, and the error
        self.plot_quantized_signal(indices, signal, quantized_signal, quantization_error)

        if is_bits:  # TestCase 1: quantization with bits, compare encoded and quantized
            # Automatically call QuantizationTest1 to validate encoded and quantized values
            QuantizationTest1( encoded_indices, quantized_signal)

        else:  # TestCase 2: quantization with levels, compare interval index, encoded, quantized, and error
            QuantizationTest2( interval_indices, encoded_indices, quantized_signal, quantization_error)

    def perform_quantization(self, signal, levels):
        # Find the signal range (min and max values)
        min_val, max_val = np.min(signal), np.max(signal)

        delta = (max_val - min_val) / levels

        # Step 3: Quantize the signal
        quantized_signal = np.zeros_like(signal)
        quantization_error = np.zeros_like(signal)
        encoded_signal = []
        interval_indices = []  # To store the interval indices

        for i, sample in enumerate(signal):
            # Find the zone index for each sample
            zone_index = int(np.floor((sample - min_val) / delta))
            # Ensure the index stays within the valid range of levels
            zone_index = np.clip(zone_index, 0, levels - 1)
            # Assign the quantized value to the midpoint of the interval
            midpoint = min_val + (zone_index + 0.5) * delta  # Use midpoint of each interval
            quantized_signal[i] = midpoint
            # Calculate the quantization error
            quantization_error[i] = midpoint - sample
            # Encode the zone index into binary
            encoded_signal.append(f"{zone_index:0{int(np.log2(levels))}b}")
            # Store the interval index
            interval_indices.append(int(zone_index) + 1)
        # Step 4: Return the results (including interval indices)
        return quantized_signal, quantization_error, encoded_signal, interval_indices

    def plot_quantized_signal(self, indices, original_signal, quantized_signal, quantization_error):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

        # Original signal plot
        ax1.plot(indices, original_signal, 'b', label="Original Signal")
        ax1.set_title('Original Signal')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

        # Quantized signal plot
        ax2.step(indices, quantized_signal, 'r', where='mid', label="Quantized Signal")
        ax2.set_title(f'Quantized Signal (Levels: {len(np.unique(quantized_signal))})')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

        # Quantization error plot
        ax3.plot(indices, quantization_error, 'g', label="Quantization Error")
        ax3.set_title('Quantization Error')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Error')
        ax3.grid(True)

        # Display the plot in the Tkinter canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def read_reference_data(self):
        amplitude = []
        phase = []
        # Open file dialog to select the reference file
        file_path = filedialog.askopenfilename(title="Select Reference Amplitude and Phase File",
                                               filetypes=[("Text Files", ".txt"), ("All Files", ".*")])

        if not file_path:  # If no file is selected, return empty lists
            print("No file selected.")
            return amplitude, phase

        try:
            with open(file_path, 'r') as file:
                for i, line in enumerate(file):
                    if i < 3:  # Skip the first three rows
                        continue
                    values = line.split()
                    if len(values) == 2:  # Ensure there are two values per line
                        amp = float(values[0].rstrip('f'))  # First value is amplitude
                        ph = float(values[1].rstrip('f'))  # Second value is phase
                        amplitude.append(amp)
                        phase.append(ph)
                    else:
                        print(f"Warning: Line {i + 1} in {file_path} does not contain two values: {line.strip()}")
        except Exception as e:
            print(f"Error reading reference data: {e}")

        return amplitude, phase

    def fourier_transform(self, signal, inverse=False):
        N = len(signal)
        k = np.arange(N)
        n = np.arange(N)
        # Exponential factor
        if inverse:
            factor = 1 / N
            exponent = np.exp(2j * np.pi * k[:, None] * n / N)  # IDFT
        else:
            factor = 1
            exponent = np.exp(-2j * np.pi * k[:, None] * n / N)  # DFT

        return factor * np.dot(exponent, signal)  # Compute DFT or IDFT

    def show_frequency_options(self):
        # Ask user to choose between DFT and IDFT
        transform_type = simpledialog.askstring("Select Transform", "Enter 'DFT' for Discrete Fourier Transform or 'IDFT' for Inverse Discrete Fourier Transform:")
        if transform_type not in ['DFT', 'IDFT']:
            messagebox.showerror("Error", "Invalid selection. Please enter 'DFT' or 'IDFT'.")
            return

        # Read the time and signal data
        # all_times, signals = self.read_signals_from_txt_files()
        # print('hello')
        # if signals is None or len(signals) == 0:
        #     messagebox.showerror("Error", "No signals read from file.")
        #     return
        #
        # # Assuming you want to process the first signal
        # time = all_times[0]  # You may want to let the user choose which signal to analyze
        # signal = signals[0]  # Assuming we're taking the first signal for demonstration
        #
        # # Ask user for sampling frequency
        # sampling_frequency = simpledialog.askfloat("Input", "Enter the sampling frequency in Hz:", minvalue=1.0)
        # if sampling_frequency is None:
        #     return

        if transform_type == 'DFT':
            # Perform Fourier Transform
            all_times, signals = self.read_signals_from_txt_files()

            if signals is None or len(signals) == 0:
                messagebox.showerror("Error", "No signals read from file.")
                return

            # Assuming you want to process the first signal
            time = all_times[0]  # You may want to let the user choose which signal to analyze
            signal = signals[0]  # Assuming we're taking the first signal for demonstration

            # Ask user for sampling frequency
            sampling_frequency = simpledialog.askfloat("Input", "Enter the sampling frequency in Hz:", minvalue=1.0)
            if sampling_frequency is None:
                return

            spectrum = self.fourier_transform(signal)

            # Frequency calculation
            N = len(signal)
            frequencies = np.fft.fftfreq(N, d=1 / sampling_frequency)

            # Amplitude and Phase calculation
            amplitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Browse for the file containing reference amplitude and phase
            # reference_file_path = filedialog.askopenfilename(title="Select Reference Amplitude and Phase File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            # if not reference_file_path:
            #     return

            # Read the reference amplitude and phase from the file
            reference_amplitude, reference_phase = self.read_reference_data()

            # Compare amplitude and phase with the original signal's amplitude and phase
            amplitude_comparison = SignalComapreAmplitude(amplitude, reference_amplitude)
            phase_comparison = SignalComaprePhaseShift(phase, reference_phase)

            if amplitude_comparison:
                print("Amplitude comparison passed Successfully.")
            else:
                print("Amplitude comparison failed.")

            if phase_comparison:
                print("Phase comparison passed Successfully.")
            else:
                print("Phase comparison failed.")
            self.plot_frequency_response(frequencies, amplitude, phase, signal)


        elif transform_type == 'IDFT':

            # Reconstruct signal using IDFT

            amp, phase = self.read_reference_data()

            if not amp or not phase:
                print("Error: No amplitude or phase data read from the reference file.")

                return

            sampling_frequency = simpledialog.askfloat("Input", "Enter the sampling frequency in Hz:", minvalue=1.0)

            if sampling_frequency is None:
                return

            # Create complex spectrum

            complex_spectrum = np.array(amp) * np.exp(1j * np.array(phase))

            reconstructed_signal = np.fft.ifft(complex_spectrum)

            reconstructed_amplitude = np.real(reconstructed_signal).tolist()  # Convert to list

            # Read reference signal and convert it to a list for comparison

            reference_time, reference_signal = self.read_signals_from_txt_files()

            reference_signal = reference_signal[0].tolist()  # Assuming the first signal is being used

            # Perform amplitude comparison

            recon_amplitude_comparison = SignalComapreAmplitude(reconstructed_amplitude, reference_signal)

            if recon_amplitude_comparison:

                print("Reconstructed Amplitude comparison passed Successfully.")

            else:

                print("Reconstructed Amplitude comparison failed.")

            # original_signal=amp
            time_values = np.arange(len(amp)) / len(amp)  # Use discrete time values
            # Reconstructed Signal

            plt.subplot(3, 1, 3)
            plt.stem(time_values, reconstructed_amplitude, label='Reconstructed Signal', linefmt='orange',
                     markerfmt='ro', basefmt=" ")
            plt.title('Reconstructed Signal')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
            # Original Signal for comparison

            # plt.subplot(4, 1, 4)
            # plt.stem(time_values, original_signal, label='Original Signal', linefmt='green', markerfmt='go',
            #          basefmt=" ")
            # plt.title('Original Signal')
            # plt.xlabel('Sample Index')
            # plt.ylabel('Amplitude')
            # plt.grid()
            # plt.legend()


    def plot_frequency_response(self, frequencies, amplitude, phase, original_signal):
        plt.figure(figsize=(12, 8))

        # Amplitude vs Frequency
        plt.subplot(3, 1, 1)
        plt.stem(frequencies, amplitude, basefmt=" ")
        plt.title('Frequency vs Amplitude')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid()

        # Phase vs Frequency
        plt.subplot(3, 1, 2)
        plt.stem(frequencies, phase, basefmt=" ")
        plt.title('Frequency vs Phase')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.grid()
        plt.show()

def SignalComapreAmplitude(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
            return False
    return True

# Function to round phase shift
def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))

# Function to compare phase shifts
def SignalComaprePhaseShift(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        A = round(SignalInput[i])
        B = round(SignalOutput[i])
        if abs(A - B) > 0.0001:
            return False
    return True
def custom_cumsum(signal):
    csum_result = []
    running_total = 0  # Initialize the running total to 0

    for value in signal:
        running_total += value  # Add each value to the running total
        csum_result.append(running_total)  # Store the running total in the result list

    return np.array(csum_result)  # Convert the list to a NumPy array for consistency
root = tk.Tk()
app = SignalPlotApp(root)
root.mainloop()
