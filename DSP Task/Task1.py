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
        root.withdraw()
        # Allow selection of one or more files
        file_paths = filedialog.askopenfilenames(title="Select one or more files", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])

        if not file_paths:
            print("No file selected.")
            return None, None

        signals = []
        for file_path in file_paths:
            signal = np.loadtxt(file_path, skiprows=3, usecols=1)
            signals.append(signal)

        if signals:
            if len(signals) == 1:
                # If only one file is selected, no need for padding, just return the single signal
                indices = np.arange(len(signals[0]))
                return indices, signals[0]  # Return a single signal instead of a list of signals
            else:
                # If multiple files are selected, pad shorter signals to match the longest one
                max_len = max([len(signal) for signal in signals])
                signals = [np.pad(signal, (0, max_len - len(signal)), mode='constant') for signal in signals]  # Padding shorter signals with zeros
                indices = np.arange(max_len)
                return indices, signals  # Return the list of signals

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
        # self.root.geometry('1400x100')
        self.button_frame = tk.Frame(self.root, bg="#f5b9f3")
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.discrete_button = tk.Button(self.button_frame, text="Read Signal",
                                         command=self.plot_both_signals, bg="#b445b0", fg="white", font=7)
        self.discrete_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.generate_button = tk.Button(self.button_frame, text="Generate Signal", command=self.open_generate_window,
                                         bg="#b445b0", fg="white", font=7)
        self.generate_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.add_button = tk.Button(self.button_frame, text="Add Signals", command=self.add_signals, bg="#b445b0",
                                    fg="white", font=7)
        self.add_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.subtract_button = tk.Button(self.button_frame, text="Subtract Signals", command=self.subtract_signals,
                                         bg="#b445b0", fg="white", font=7)
        self.subtract_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.multiply_button = tk.Button(self.button_frame, text="Multiply Signal", command=self.multiply_signal,
                                         bg="#b445b0", fg="white", font=7)
        self.multiply_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.square_button = tk.Button(self.button_frame, text="Square Signal", command=self.square_signal,
                                       bg="#b445b0", fg="white", font=7)
        self.square_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.normalize_button = tk.Button(self.button_frame, text="Normalize Signal", command=self.normalize_signal,
                                          bg="#b445b0", fg="white", font=7)
        self.normalize_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.accumulate_button = tk.Button(self.button_frame, text="Accumulate Signal", command=self.accumulate_signal,
                                           bg="#b445b0", fg="white", font=7)
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

    def plot_both_signals(self):  # plot analog and digital signals
        fig = plt.Figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        indices,samples= read_signals_from_txt_files()
        indices = np.arange(len(samples))
        smooth_x = np.linspace(indices.min(), indices.max(), 500)
        spl = make_interp_spline(indices, samples, k=3)
        smooth_y = spl(smooth_x)
        ax1.plot(smooth_x, smooth_y, 'b')
        ax1.set_title('Analog Signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

        ax2.stem(indices, samples, linefmt='r-', markerfmt='ro', basefmt='gray')
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
        result = signals[1] - signals[0]
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
            multiplier = float(self.get_user_input("Enter a multiplier: "))

            result = signals * multiplier

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
        if signals is None or (isinstance(signals, list) and len(signals) != 1):
            messagebox.showerror("Invalid")
            return

        try:
            result = np.square(signals)
            SignalSamplesAreEqual("Output squaring signal 1.txt", indices, result)
            self.plot_signal(indices, result, title="Squared Signal")
        except ValueError:
            messagebox.showerror("invalid")
    def normalize_signal(self):
        indices, signals = read_signals_from_txt_files()
        if signals is None or (isinstance(signals, list) and len(signals) != 1):
            messagebox.showerror("Invalid")
            return
        try:
            normalization_type = self.get_user_input("Normalize to -1 to 1 or 0 to 1? (Enter '-1' or '0')").strip()
            # signal = signals[0]
            if normalization_type == "-1":
                result = 2 * (signals - np.min(signals)) / (np.max(signals) - np.min(signals)) - 1
            elif normalization_type == "0":
                result = (signals - np.min(signals)) / (np.max(signals) - np.min(signals))
            SignalSamplesAreEqual("normalize of signal 1 (from -1 to 1)-- output.txt", indices, result)
            self.plot_signal(indices, result, title=f"Normalized Signal ({normalization_type} to 1)")
        except ValueError:
            messagebox.showerror("invalid")
    def accumulate_signal(self):
        indices, signals = read_signals_from_txt_files()
        if signals is None or (isinstance(signals, list) and len(signals) != 1):
            messagebox.showerror("Invalid")
            return
        try:
            result = np.cumsum(signals)
            SignalSamplesAreEqual("output accumulation for signal1.txt", indices, result)
            self.plot_signal(indices, result, title="Accumulated Signal")
        except ValueError:
            messagebox.showerror("invalid")
# comparing generated signal by signal in  the files
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

