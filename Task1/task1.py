import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline

def read_signal_from_txt(file_path):
    try:
        signal = np.loadtxt(file_path, skiprows=3,usecols=1)
        return signal
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def generate_signal(signal_type, amplitude, phase_shift, f_analog, f_sampling, duration=1):
    num_samples = int(f_sampling * duration)
    samples = np.arange(num_samples) # indices of samples
    t = samples / f_sampling  
    if signal_type == "sine":
        signal = amplitude * np.sin(2 * np.pi * f_analog * t + phase_shift)
    elif signal_type == "cosine":
        signal = amplitude * np.cos(2 * np.pi * f_analog * t + phase_shift)
    return t, signal


class SignalPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Displaying Signals")
        self.button_frame = tk.Frame(self.root,bg="#f5b9f3")
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.discrete_button = tk.Button(self.button_frame, text="Analog & Discrete Signals", command=self.plot_both_signals, bg="#b445b0", fg="white", font=10)
        self.discrete_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.generate_button = tk.Button(self.button_frame, text="Generate Signal", command=self.open_generate_window, bg="#b445b0", fg="white", font=10)
        self.generate_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = None

    def plot_signal(self, indices, samples, signal_type):
        fig = plt.Figure(figsize=(5, 4))
        
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Amplitude')
        ax.plot(indices, samples, 'b')
        ax.set_xlim(0,0.025)
        ax.set_title('Sinusoidal Waves')
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def plot_both_signals(self):
        fig = plt.Figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211)  # Analog Signal on the top
        ax2 = fig.add_subplot(212)  # Discrete Signal on the bottom

    # Example data (replace these with your actual data points)
        samples = read_signal_from_txt("signal1.txt")
        indices = np.arange(len(samples))
    # Plotting the analog signal (continuous)
        smooth_x = np.linspace(indices.min(), indices.max(), 500)  # Generate more points
        spl = make_interp_spline(indices, samples, k=3)  # Cubic spline interpolation
        smooth_y = spl(smooth_x)

    # Plotting the smooth analog signal
        ax1.plot(smooth_x, smooth_y, 'b')  # Smooth line without points
        ax1.set_title('Analog Signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

    # Plotting the discrete signal (stem plot)
        ax2.stem(indices, samples, linefmt='r-', markerfmt='ro', basefmt='gray')
        ax2.set_title('Discrete Signal')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)

    # Handle canvas redraw and destroy previous widget
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)



    def open_generate_window(self):
        # Create a new window for user input
        self.generate_window = tk.Toplevel(self.root)
        self.generate_window.title("Generate Signal")

        # Labels and Entry widgets for user inputs
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
        # Button to generate the signal
        generate_button = tk.Button(self.generate_window, text="Generate", command=self.generate_user_signal)
        generate_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    def generate_user_signal(self):
        try:
            # Get user input values
            signal_type = self.signal_type_entry.get()
            amplitude = float(self.amplitude_entry.get())
            phase_shift = float(self.phase_shift_entry.get())
            f_analog = float(self.frequency_entry.get())
            f_sampling = float(self.sampling_frequency_entry.get())
            # Generate signal based on user input
            t, signal = generate_signal(signal_type, amplitude, phase_shift, f_analog, f_sampling)
            # smoothed_signal = savgol_filter(signal, window_length=20, polyorder=10)
        
            if signal_type=="sine":
                SignalSamplesAreEqual("SinOutput.txt",t,signal)
            elif signal_type=="cosine":
                SignalSamplesAreEqual("CosOutput.txt",t,signal)
            self.plot_signal(t, signal, signal_type="sinsudil")
        
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for the signal parameters.")

def SignalSamplesAreEqual(file_name,nsamples, signal):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
                
    if len(expected_samples)!=len(signal):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(signal[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")

root = tk.Tk()
app = SignalPlotApp(root)
root.mainloop()

