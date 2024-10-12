import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

def read_signal_from_txt(file_path):
    try:
        signal = np.loadtxt(file_path, skiprows=3,usecols=1)
        return signal
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def generate_signal(signal_type, amplitude, phase_shift, f_analog, f_sampling, duration=1):
    num_samples = int(f_sampling * duration)
    samples = np.arange(num_samples)
    t = samples / f_sampling  # Time array for internal use (if needed)
    if signal_type == "sine":
        signal = amplitude * np.sin(2 * np.pi * f_analog * t + phase_shift)
    elif signal_type == "cosine":
        signal = amplitude * np.cos(2 * np.pi * f_analog * t + phase_shift)
    return samples, signal

class SignalPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Displaying Signals")
        self.button_frame = tk.Frame(self.root,bg="#f5b9f3")
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.discrete_button = tk.Button(self.button_frame, text="Discrete Signal", command=self.plot_discrete_signal,bg="#b445b0" , fg="white",font=10)
        self.discrete_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.continuous_button = tk.Button(self.button_frame, text="Continuous Signal", command=self.plot_continuous_signal,bg="#b445b0" ,fg="white",font=10)
        self.continuous_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.generate_button = tk.Button(self.button_frame, text="Generate Signal", command=self.open_generate_window, bg="#b445b0", fg="white", font=10)
        self.generate_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = None

    def plot_signal(self, indices, samples, signal_type="discrete"):
        fig = plt.Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Amplitude')

        if len(indices) > 50:
            indices = indices[:50]
            samples = samples[:50]


        if signal_type == "discrete":
            ax.stem(indices, samples, basefmt=" ")
            ax.set_title('Discrete Signal Representation')
        elif signal_type == "continuous":
            ax.plot(indices, samples, 'r')
            ax.set_title('Continuous Signal Representation')
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def plot_discrete_signal(self):
        samples = read_signal_from_txt("signal1.txt")
        indices = np.arange(len(samples))
        self.plot_signal(indices, samples, signal_type="discrete")
    def plot_continuous_signal(self):
        t = np.linspace(0, 2 * np.pi, 50)
        samples = np.sin(t)
        self.plot_signal(t, samples, signal_type="continuous")
    


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
            # Plot the generated signal
            
            # Sample sharp wave signal
            
            self.plot_signal(t, signal, signal_type="continuous")
        
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

