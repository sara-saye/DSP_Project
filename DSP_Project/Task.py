import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline
def QuantizationTest1(Your_EncodedValues, Your_QuantizedValues):
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
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        file_name = filedialog.askopenfilename(title="Select the file to compare",filetypes=[("Text files", ".txt"), ("All files", ".*")])

        if not file_name:
            print("No file selected.")
            return
        expected_indices = []
        expected_samples = []
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
                    V1 = int(L[0])
                    V2 = float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break
        if len(expected_samples) != len(samples):
            print("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal have different values from the expected one")
                return
        print("Test case passed successfully")
def Shift_Fold_Signal(Your_indices, Your_samples):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_name = filedialog.askopenfilename(title="Select Signal File")
    if not file_name:
        print("No file selected.")
        return
    expected_indices = []
    expected_samples = []
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
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Shift_Fold_Signal Test case failed, your signal have different values from the expected one")
            return
    print("Shift_Fold_Signal Test case passed successfully")
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

        self.discrete_button = tk.Button(self.button_frame, text="Read Signal", command=self.plot_both_signals,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.discrete_button.grid(row=0, column=0, padx=10, pady=10)

        self.generate_button = tk.Button(self.button_frame, text="Generate Signal", command=self.open_generate_window,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.generate_button.grid(row=0, column=1, padx=10, pady=10)

        self.add_button = tk.Button(self.button_frame, text="Add Signals", command=self.add_signals, bg="#ccbeb1",fg="black", font=7, width=button_width)
        self.add_button.grid(row=0, column=2, padx=10, pady=10)

        self.subtract_button = tk.Button(self.button_frame, text="Subtract Signals", command=self.subtract_signals,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.subtract_button.grid(row=0, column=3, padx=10, pady=10)

        self.multiply_button = tk.Button(self.button_frame, text="Multiply Signal", command=self.multiply_signal,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.multiply_button.grid(row=0, column=4, padx=10, pady=10)

        self.square_button = tk.Button(self.button_frame, text="Square Signal", command=self.square_signal,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.square_button.grid(row=0, column=5, padx=10, pady=10)

        self.normalize_button = tk.Button(self.button_frame, text="Normalize Signal", command=self.normalize_signal,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.normalize_button.grid(row=1, column=0, padx=10, pady=10)

        self.accumulate_button = tk.Button(self.button_frame, text="Accumulate Signal", command=self.accumulate_signal,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.accumulate_button.grid(row=1, column=1, padx=10, pady=10)

        self.quantize_button = tk.Button(self.button_frame, text="Quantize Signal", command=self.quantize_signal,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.quantize_button.grid(row=1, column=2, padx=10, pady=10)

        # Add Frequency Domain button
        self.frequency_button = tk.Button(self.button_frame, text="Frequency Domain",command=self.show_frequency_options,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.frequency_button.grid(row=1, column=3, padx=10, pady=10)
        self.time_domain = tk.Button(self.button_frame, text="Time Domain", command=self.time_domain,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.time_domain.grid(row=1, column=4, padx=10, pady=10)

        self.convolution = tk.Button(self.button_frame, text="Convolution", command=self.convolution_action, bg="#ccbeb1",fg="black", font=7, width=button_width)
        self.convolution.grid(row=1, column=5, padx=10, pady=10)

        self.correlation = tk.Button(self.button_frame, text="Correlation", command=self.corr_action,bg="#ccbeb1", fg="black", font=7, width=button_width)
        self.correlation.grid(row=2, column=0, padx=10, pady=10)

        self.button_frame.pack_propagate(False)
        self.canvas = None

    # def read_signals_from_txt_files(self):
    #     try:
    #         root = tk.Tk()
    #         root.withdraw()  # Hide the root window
    #
    #         # Allow selection of one or more files
    #         file_paths = filedialog.askopenfilenames(title="Select one or more files",filetypes=[("Text files", ".txt"), ("All files", ".*")])
    #
    #         if not file_paths:
    #             print("No file selected.")
    #             return None, None
    #
    #         signals = []
    #         all_indices = []
    #
    #         for file_path in file_paths:
    #             # Load both indices (first column) and signal values (second column)
    #             data = np.loadtxt(file_path, skiprows=3, usecols=(0, 1))
    #             indices = data[:, 0]  # First column for indices
    #             signal = data[:, 1]  # Second column for signal values
    #             all_indices.append(indices)
    #             signals.append(signal)
    #
    #         if signals:
    #             # Merge all indices, ensuring that we have a union of all index points
    #             unified_indices = np.unique(np.concatenate(all_indices))
    #
    #             # Pad and align each signal to the unified set of indices
    #             aligned_signals = []
    #             for i, indices in enumerate(all_indices):
    #                 # Find where the original indices fit in the unified indices
    #                 aligned_signal = np.zeros_like(unified_indices, dtype=float)
    #                 idx_in_unified = np.searchsorted(unified_indices, indices)
    #                 aligned_signal[idx_in_unified] = signals[i]
    #                 aligned_signals.append(aligned_signal)
    #
    #             return unified_indices, aligned_signals  # Return the aligned signals
    #
    #         return None, None
    #
    #     except Exception as e:
    #         print(f"Error reading file: {e}")
    #         return None, None
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

            return all_indices, signals  # Return the indices and signals for all selected files

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
            indices = indices[0]
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
        indices, signals = self.read_signals_from_txt_files()
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
            self.plot_signal(indices[0], result, title="Squared Signal")
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
            self.plot_signal(indices[0], result, title=f"Normalized Signal ({normalization_type} to 1)")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def accumulate_signal(self):
        indices, signals = self.read_signals_from_txt_files()

        # Ensure that exactly one signal is selected
        if signals is None or len(signals) != 1:
            messagebox.showerror("Input Error", "Please select exactly one signal file for accumulation.")
            return

        signal = np.array(signals[0])  # Get the first (and only) signal

        # Compute the cumulative sum of the signal
        result = custom_cumsum(signal)

        # Save the result and plot it
        SignalSamplesAreEqual(indices, result)
        self.plot_signal(indices[0], result, title="Accumulated Signal")

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
            self.plot_signal(indices[0], result, title="Added Signals")

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
        self.plot_signal(indices[0], result, title="Subtracted Signals")

    def multiply_signal(self):
        indices, signals = self.read_signals_from_txt_files()

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
            self.plot_signal(indices[0], result, title=f"Signal Multiplied by {multiplier}")

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
        signal = signals[0]
        indices=indices[0]# Get the first (and only) signal

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
            QuantizationTest1(encoded_indices, quantized_signal)

        else:  # TestCase 2: quantization with levels, compare interval index, encoded, quantized, and error
            QuantizationTest2(interval_indices, encoded_indices, quantized_signal, quantization_error)

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
        transform_type = simpledialog.askstring("Select Transform",
                                                "Enter 'DFT' for Discrete Fourier Transform "
                                                " 'IDFT' for Inverse Discrete Fourier Transform "
                                                "'DCT' for Discrete Cosine Transform "
                                                "'R-DC' for removing DC component")
        if transform_type not in ['DFT', 'IDFT', 'DCT','R-DC']:
            messagebox.showerror("Error", "Invalid selection. Please enter 'DFT' or 'IDFT' or 'DCT'.")
            return

        if transform_type == 'DFT':
            all_times, signals = self.read_signals_from_txt_files()
            if signals is None or len(signals) == 0:
                messagebox.showerror("Error", "No signals read from file.")
                return

            signal = signals[0]
            sampling_frequency = simpledialog.askfloat("Input", "Enter the sampling frequency in Hz:", minvalue=1.0)
            if sampling_frequency is None:
                return

            spectrum = self.fourier_transform(signal)

            N = len(signal)
            angular_frequencies = np.array([2 * np.pi * k * sampling_frequency / N for k in range(N)])

            amplitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            reference_amplitude, reference_phase = self.read_reference_data()

            amplitude_comparison = SignalComapreAmplitude(amplitude, reference_amplitude)
            phase_comparison = SignalComaprePhaseShift(phase, reference_phase)

            if amplitude_comparison:
                print("Amplitude comparison passed successfully.")
            else:
                print("Amplitude comparison failed.")

            if phase_comparison:
                print("Phase comparison passed successfully.")
            else:
                print("Phase comparison failed.")

            self.plot_frequency_response(angular_frequencies, amplitude, phase, signal)

        elif transform_type == 'IDFT':
            amp, phase = self.read_reference_data()
            if not amp or not phase:
                print("Error: No amplitude or phase data read from the reference file.")
                return

            sampling_frequency = simpledialog.askfloat("Input", "Enter the sampling frequency in Hz:", minvalue=1.0)
            if sampling_frequency is None:
                return

            amp = np.array(amp)
            phase = np.array(phase)

            real_part = amp * np.cos(phase)
            imaginary_part = amp * np.sin(phase)
            complex_spectrum = real_part + 1j * imaginary_part

            reconstructed_signal = self.fourier_transform(complex_spectrum, inverse=True)
            reconstructed_amplitude = np.round(np.real(reconstructed_signal), decimals=0).tolist()

            reference_time, reference_signal = self.read_signals_from_txt_files()
            reference_signal = reference_signal[0].tolist()

            recon_amplitude_comparison = SignalComapreAmplitude(reconstructed_amplitude, reference_signal)

            if recon_amplitude_comparison:
                print("Reconstructed Amplitude comparison passed successfully.")
            else:
                print("Reconstructed Amplitude comparison failed.")

            time_values = np.arange(len(amp)) / len(amp)

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
        elif transform_type == 'DCT':
            indeses, signals = self.read_signals_from_txt_files()
            if signals is None or len(signals) == 0:
                messagebox.showerror("Error", "No signals read from file.")
                return
            m = int(self.get_user_input("Enter number of DCT coefficients:"))
            signal = signals[0]
            rst = compute_dct(signal, m)
            self.plot_signal(indeses[0], rst, "DCT")
            SignalSamplesAreEqual(indeses[0], rst)


        elif transform_type == 'R-DC':

            indices, signals = self.read_signals_from_txt_files()

            if signals is None or len(signals) == 0:
                messagebox.showerror("Error", "No signals read from file.")

                return

            signal = signals[0]

            # Perform DFT to move the signal to the frequency domain

            dft_signal = self.fourier_transform(signal)

            # Set the DC component (first frequency component, corresponding to k=0) to zero

            dft_signal[0] = 0

            # Perform the inverse DFT to reconstruct the time-domain signal without the DC component
            signal_no_dc = self.fourier_transform(dft_signal, inverse=True)
            signal_no_dc = np.round(np.real(signal_no_dc), decimals=0).tolist()
            print(signal_no_dc)
            #SignalSamplesAreEqual(indices, signal_no_dc)

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

    def time_domain(self):
        operation = simpledialog.askstring("Select number of operation", """
                                   1- Sharpening Signal
                                   2- Smoothing Signal
                                   3- Shift Signal
                                   4- Fold Signal
                                   5- Shift Folding Signal
                                   6-Remove DC Component""")
        if operation not in ['1', '2', '3', '4','5','6']:
            messagebox.showerror("Error", "Invalid selection. Please select option.")
            return
        if operation == '1':
            DerivativeSignal()
        elif operation== '2':
            self.smoothing_action()
        elif operation == '3':
            all_times, signals = self.read_signals_from_txt_files()
            if signals is None or len(signals) == 0:
                messagebox.showerror("Error", "No signals read from file.")
                return
            signal = signals[0]
            k = int(self.get_user_input("Enter number of shifts 'k': "))
            shifted_times, shifted_signal = shift_signal(all_times, signal, k)
            SignalSamplesAreEqual(shifted_times, shifted_signal)
            plt.figure(figsize=(12, 6))
            # Original Signal
            plt.subplot(2, 1, 1)
            plt.plot(signal, color='blue', label='Original Signal')
            plt.title('Original Signal')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(shifted_signal, color='red', label='Shifted Signal')
            plt.title('Shift by K of the Signal')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()


        elif operation == '4':
            all_times, signals = self.read_signals_from_txt_files()
            if signals is None or len(signals) == 0:
                messagebox.showerror("Error", "No signals read from file.")
                return
            signal = signals[0]
            all_times=all_times[0]
            folded_times, folded_signal = fold_signal(all_times, signal)
            SignalSamplesAreEqual(folded_times, folded_signal)
            plt.figure(figsize=(12, 6))

            # Original Signal
            plt.subplot(2, 1, 1)
            plt.plot(signal, color='blue', label='Original Signal')
            plt.title('Original Signal')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(folded_signal, color='red', label='Folded Signal')
            plt.title('Folded Signal')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()


        elif operation == '5':
            # Fold the signal first
            all_times, signals = self.read_signals_from_txt_files()
            if signals is None or len(signals) == 0:
                messagebox.showerror("Error", "No signals read from file.")
                return
            signal = signals[0]
            all_times=all_times[0]
            # Ask user for the number of shifts
            k = int(self.get_user_input("Enter number of shifts 'k': "))
            # Shift the folded signal
            shifted_times, shifted_folded_signal = fold_and_shift_signal(all_times, signal, k)
            int_times = np.round(shifted_times).astype(int)
            int_signals = np.round(shifted_folded_signal).astype(int)
            Shift_Fold_Signal(int_times, int_signals)
            plt.figure(figsize=(12, 6))

            # Original Signal
            plt.subplot(2, 1, 1)
            plt.plot(signal, color='blue', label='Original Signal')
            plt.title('Original Signal')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(int_times, int_signals, color='red', label=' Shifted-Folded Signal')
            plt.title(' Shifted-Folded Signal')
            plt.xlabel('Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
        elif operation=='6':
            indises,signals=self.read_signals_from_txt_files()
            signal=signals[0]
            dc_removed =remove_dc_time_domain(signal)
            SignalSamplesAreEqual(indises,dc_removed)
    def smoothing_action(self):
        try:
            # Use your `reed` function to browse and read the signal
            indices, signals = self.read_signals_from_txt_files()  # Assuming this reads the signal as a list or numpy array
            signal=signals[0]
            # Ask the user for the window size
            window_size = simpledialog.askinteger("Input", "Enter window size for smoothing:")
            if window_size is None:  # If the user cancels
                return

            # Smooth the signal
            smoothed_signal = smooth_signal(signal, window_size)
            SignalSamplesAreEqual(indices,smoothed_signal)

            # Display the result (or pass it to your framework for further processing)
            messagebox.showinfo("Smoothing Complete", "The signal has been smoothed.")
            # print("Smoothed Signal:", smoothed_signal)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    def convolution_action(self):
        try:
            # Read signals from files
            all_indices, all_signals = self.read_signals_from_txt_files()
            if all_signals is None or len(all_signals) < 2:
                messagebox.showerror("Error", "Please select at least two signals.")
                return

            # Extract the first and second signals and their indices
            signal1 = all_signals[0]
            indices1 = all_indices[0]
            signal2 = all_signals[1]
            indices2 = all_indices[1]

            convolved_indices, convolved_signal = convolve_signals(indices1,signal1,indices2,signal2)
            ConvTest(convolved_indices,convolved_signal)

            # Display the results


            # Show completion message
            messagebox.showinfo("Convolution Complete", "Signals have been convolved successfully.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
    def corr_action(self):
        indices,signals=self.read_signals_from_txt_files()
        signal1=signals[0]
        signal2=signals[1]
        c_signal=normalized_cross_correlation(signal1,signal2)
        print(c_signal)
        SignalSamplesAreEqual(indices[0],c_signal)
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
def compute_dct(signal, m):
    N = len(signal)
    dct_result = [
        np.sqrt(2 / N) * sum(signal[n] * np.cos(np.pi / (4 * N) * (2 * n - 1) * (2 * k - 1)) for n in range(N)) for k in
        range(N)]
    # Save the first m coefficients to a text file
    with open('dct_coefficients.txt', 'w') as f:
        for value in dct_result[:m]:
            f.write(f"{value}\n")
    print(f"The first {m} DCT coefficients have been saved to 'dct_coefficients.txt'.")
    return dct_result
def DerivativeSignal():
    # Input signal
    InputSignal = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
        93, 94, 95, 96, 97, 98, 99, 100
    ]

    # Expected outputs
    expectedOutput_first = [1] * 99
    expectedOutput_second = [0] * 98

    """
    Start: Your Code Here
    """

    # First Derivative Calculation
    FirstDrev = [InputSignal[i] - InputSignal[i - 1] for i in range(1, len(InputSignal))]

    # Second Derivative Calculation
    SecondDrev = [InputSignal[i + 1] - 2 * InputSignal[i] + InputSignal[i - 1] for i in range(1, len(InputSignal) - 1)]

    """
    End
    """

    # Testing the Code
    if len(FirstDrev) != len(expectedOutput_first) or len(SecondDrev) != len(expectedOutput_second):
        print("Mismatch in length")
        return

    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            print("1st derivative wrong")
            return

    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            print("2nd derivative wrong")
            return

    if first and second:
        print("Derivative Test case passed successfully")
    else:
        print("Derivative Test case failed")
        return

    # Plotting the Original Signal, First Derivative, and Second Derivative
    plt.figure(figsize=(12, 6))

    # Original Signal
    plt.subplot(3, 1, 1)
    plt.plot(InputSignal, color='blue', label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    # First Derivative
    plt.subplot(3, 1, 2)
    plt.plot(FirstDrev, color='red', label='First Derivative')
    plt.title('First Derivative of the Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    # Second Derivative
    plt.subplot(3, 1, 3)
    plt.plot(SecondDrev, color='green', label='Second Derivative')
    plt.title('Second Derivative of the Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    return
def shift_signal(indices, signal, k):
    n = len(indices)
    shifted_signal = signal[:]  # Keep the signal (Y-values) the same
    shifted_indices = [0] * n  # Initialize the list with zeros

    if k < 0:  # Right shift (delay)
        for i in range(n):
            new_index = i + k
            if new_index < n and new_index >= 0:
                shifted_indices[new_index] = indices[i]  # Assign the original index to the new shifted index
            else:
                shifted_indices[i] = 0  # Or use any default value for out-of-range indices

    elif k > 0:  # Left shift (advance)
        for i in range(n):
            new_index = i - k
            if new_index >= 0 and new_index < n:
                shifted_indices[new_index] = indices[i]  # Assign the original index to the new shifted index
            else:
                shifted_indices[i] = 0  # Or use any default value for out-of-range indices

    return shifted_indices, shifted_signal  # Return the shifted indices with the original signal
def fold_signal(indices, signal):
    folded_indices = indices[::-1]  # Reverse the indices
    folded_signal = signal[::-1]  #

    return folded_indices, folded_signal  # Return both the time indices and the folded signal
def fold_and_shift_signal(indices, signal, k):
    n = len(indices)

    # Step 1: Fold the signal (reverse both signal and indices)
    folded_indices = indices[::-1]
    folded_signal = signal[::-1]

    # Step 2: Shift the folded signal (Right shift for k > 0, Left shift for k < 0)
    shifted_indices = []
    shifted_signal = []

    for i in range(n):
        new_index = folded_indices[i] + k  # Shift index right if k > 0, left if k < 0

        # Include all shifted indices, not restricting to the original min and max
        shifted_indices.append(new_index)
        shifted_signal.append(folded_signal[i])

    # Sort the result based on shifted indices for correct ordering
    sorted_pairs = sorted(zip(shifted_indices, shifted_signal))
    shifted_indices = [pair[0] for pair in sorted_pairs]

    return shifted_indices, shifted_signal
def smooth_signal(signal, window_size):
    if window_size < 2 or len(signal) <= window_size:
        raise ValueError("Window size must be greater than 1 and signal length must be greater than window size")

    smoothed_signal = []  # This will store the output signal
    half_window = window_size // 2  # Calculate half the window size

    # Iterate over the signal starting from index `half_window` to len(x) - half_window
    for n in range(half_window, len(signal) - half_window):
        # Compute the average of the window centered at x[n]
        avg = sum(signal[n - half_window:n + half_window + 1]) / window_size
        smoothed_signal.append(avg)

    return smoothed_signal
def remove_dc_time_domain(signal):

    # Manually compute the mean (DC component) of the signal
    sum_signal = sum(signal)
    n = len(signal)
    dc_component = sum_signal / n  # Calculate the mean

    # Subtract the DC component from each element in the signal
    signal_no_dc = [x - dc_component for x in signal]

    return signal_no_dc
def convolve_signals(indices1, signal1, indices2, signal2):
    # Convert indices to integers if they are numpy.float64
    indices1 = np.array(indices1, dtype=int)
    indices2 = np.array(indices2, dtype=int)

    start_index = indices1[0] + indices2[0]
    end_index = indices1[-1] + indices2[-1]

    # Initialize an empty list for the result signal
    convolved_signal = []

    # Iterate over each possible index in the output signal
    for n in range(start_index, end_index + 1):
        conv_value = 0
        # Compute the sum for the convolution at index n
        for i in range(len(indices1)):
            for j in range(len(indices2)):
                if n == indices1[i] + indices2[j]:
                    conv_value += signal1[i] * signal2[j]
        convolved_signal.append(conv_value)

    # Create the corresponding indices for the convolved signal
    convolved_indices = list(range(start_index, end_index + 1))

    return convolved_indices, convolved_signal
def ConvTest(Your_indices, Your_samples):
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]

    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """
    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Conv Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Conv Test case failed, your signal have different values from the expected one")
            return
    print("Conv Test case passed successfully")
def normalized_cross_correlation(signal1, signal2):
    N = len(signal1)
    res = []  # List to store results

    # Loop over all possible shifts in reverse order (from N-1 to 0)
    for shift in range(N):  # Starts from N-1 and ends at 0
        r12 = 0
        d1 = 0
        d2 = 0

        # Shift signal2 using np.roll
        rolled_signal2 = np.roll(signal2, shift) # x2(i+ or - n)

        # Compute r12 (cross-correlation), d1, and d2 for the current shift
        for i in range(N):
            r12 += signal1[i] * rolled_signal2[i]
            d1 += signal1[i] * signal1[i]
            d2 += rolled_signal2[i] * rolled_signal2[i]

        # Normalize r12 by N
        r12 /= N

        # Calculate denominator
        denominator = np.sqrt(d1 * d2) / N

        # Compute the normalized cross-correlation for this shift
        res.append(r12 / denominator)
    return res

root = tk.Tk()
app = SignalPlotApp(root)
root.mainloop()

