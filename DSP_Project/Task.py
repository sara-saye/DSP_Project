import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline
# read signal from signal1.txt and ignoring the first 3 rows and read only the signal values
def read_signals_from_txt_files():
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Allow selection of one or more files
        file_paths = filedialog.askopenfilenames(title="Select one or more files", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])

        if not file_paths:
            print("No file selected.")
            return None, None

        signals = []
        all_indices = []

        for file_path in file_paths:
            # Load both indices (first column) and signal values (second column)
            data = np.loadtxt(file_path, skiprows=3, usecols=(0, 1))
            indices = data[:, 0]  # First column for indices
            signal = data[:, 1]   # Second column for signal values
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
# gui
class SignalPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Displaying Signals")
        # self.root.geometry('1400x1400')
        self.button_frame = tk.Frame(self.root, bg="#f2ede9")
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.discrete_button = tk.Button(self.button_frame, text="Read Signal",command=self.plot_both_signals, bg="#ccbeb1", fg="black", font=7)
        self.discrete_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.generate_button = tk.Button(self.button_frame, text="Generate Signal", command=self.open_generate_window,bg="#ccbeb1", fg="black", font=7)
        self.generate_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.add_button = tk.Button(self.button_frame, text="Add Signals", command=self.add_signals, bg="#ccbeb1", fg="black", font=7)
        self.add_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.subtract_button = tk.Button(self.button_frame, text="Subtract Signals", command=self.subtract_signals,bg="#ccbeb1", fg="black", font=7)
        self.subtract_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.multiply_button = tk.Button(self.button_frame, text="Multiply Signal", command=self.multiply_signal,bg="#ccbeb1", fg="black", font=7)
        self.multiply_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.square_button = tk.Button(self.button_frame, text="Square Signal", command=self.square_signal,bg="#ccbeb1", fg="black", font=7)
        self.square_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.normalize_button = tk.Button(self.button_frame, text="Normalize Signal", command=self.normalize_signal,bg="#ccbeb1", fg="black", font=7)
        self.normalize_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.accumulate_button = tk.Button(self.button_frame, text="Accumulate Signal", command=self.accumulate_signal,bg="#ccbeb1", fg="black", font=7)
        self.accumulate_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas = None

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
        indices, signals = read_signals_from_txt_files()
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

    # recieving user inputs then ploting it as continous signals
    def generate_user_signal(self):
        try:
            signal_type = self.signal_type_entry.get()
            amplitude = float(self.amplitude_entry.get())
            phase_shift = float(self.phase_shift_entry.get())
            f_analog = float(self.frequency_entry.get())
            f_sampling = float(self.sampling_frequency_entry.get())
            t, signal = generate_signal(signal_type, amplitude, phase_shift, f_analog, f_sampling)
            if signal_type == "sine":
                SignalSamplesAreEqual("SinOutput.txt", t, signal)
            elif signal_type == "cosine":
                SignalSamplesAreEqual("CosOutput.txt", t, signal)
            self.plot_signal(t, signal,title="generated signal")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for the signal parameters.")
    def add_signals(self):
        indices, signals = read_signals_from_txt_files()
        if signals:
            result = np.sum(signals, axis=0)
            SignalSamplesAreEqual("Signal1+signal2.txt",indices,result)
            self.plot_signal(indices, result, title="Added Signals")
    def subtract_signals(self):
        # Read signals from two different files
        indices, signals = read_signals_from_txt_files()

        # Check if exactly two signals are selected
        if signals is None or len(signals) != 2:
            messagebox.showerror("Input Error", "Please select exactly two signal files for subtraction.")
            return

        # Print the signals to check if they are being read correctly
        print("Signal 1:", signals[0])
        print("Signal 2:", signals[1])

        # Subtract the second signal from the first
        result = np.abs(signals[0] - signals[1])
        print("Subtracted Result:", result)  # Debugging print

        # Optionally, compare with an expected result if applicable
        SignalSamplesAreEqual("signal1-signal2.txt", indices, result)

        # Plot the result of subtraction
        self.plot_signal(indices, result, title="Subtracted Signals")
    def multiply_signal(self):
        indices, signals = read_signals_from_txt_files()

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

            # Debugging: Print the original and multiplied results
            print("Original Signal:", signals)
            print("Multiplier:", multiplier)
            print("Resulting Signal:", result)

            # Save the result and plot it
            SignalSamplesAreEqual("MultiplySignalByConstant-Signal1 - by 5.txt", indices, result)
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

    def square_signal(self):
        indices, signals = read_signals_from_txt_files()
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
            SignalSamplesAreEqual("Output squaring signal 1.txt", indices, result)
            self.plot_signal(indices, result, title="Squared Signal")
        except ValueError as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def normalize_signal(self):
        indices, signals = read_signals_from_txt_files()

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

            SignalSamplesAreEqual("normalize of signal 1 (from -1 to 1)-- output.txt", indices, result)
            self.plot_signal(indices, result, title=f"Normalized Signal ({normalization_type} to 1)")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def accumulate_signal(self):
        indices, signals = read_signals_from_txt_files()

        # Check if we have one signal
        if signals is None or not isinstance(signals, list) or len(signals) != 1:
            messagebox.showerror("Invalid Input", "Please select exactly one signal file.")
            return

        signal = np.array(signals[0])  # Convert to NumPy array for processing

        # Check if the signal is numeric
        if not np.issubdtype(signal.dtype, np.number):
            messagebox.showerror("Invalid Input", "The selected signal is not numeric.")
            return

        try:
            result = np.cumsum(signal)  # Calculate the cumulative sum
            SignalSamplesAreEqual("output accumulation for signal1.txt", indices, result)
            self.plot_signal(indices, result, title="Accumulated Signal")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def SignalSamplesAreEqual(file_name, indices, samples):
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
root = tk.Tk()
app = SignalPlotApp(root)
root.mainloop()

