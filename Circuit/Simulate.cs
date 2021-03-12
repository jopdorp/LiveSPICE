using ComputerAlgebra.LinqCompiler;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Collections.ObjectModel;
using System.ComponentModel;
using MathNet.Numerics.IntegralTransforms;
using System.Numerics;
using System.Text;
using System.Collections;

namespace Circuit
{

     class NullDevice : Device
    {
        private Guid classid;

        public override Stream Open(Stream.SampleHandler Callback, Channel[] Input, Channel[] Output)
        {
            return new NullStream(ProcessSamples);
        }

        private void ProcessSamples(int Count, SampleBuffer[] In, SampleBuffer[] Out, double Rate)
        {
            // The time covered by these samples.
            double timespan = Count / Rate;

            // Apply input gain.
            for (int i = 0; i < In.Length; ++i)
            {
                Channel ch = InputChannels[i];
                ;
                System.Console.WriteLine(In[i].Amplify(1));
            }


            // Apply output gain.
            for (int i = 0; i < Out.Length; ++i)
            {
                Channel ch = OutputChannels[i];
                System.Console.WriteLine(Out[i].Amplify(1));
            }
        }
    }
    public class Simulate
    {
        public static void Main(string[] args) {
            Schematic schema = Schematic.Load(args[0]);
            Device device = new NullDevice();
            InputChannel input = new InputChannel(1);
            Channel[] Inputs = {};
            Channel[] Outputs = {new OutputChannel(1)};
            LiveSimulation simulation = new LiveSimulation(schema, device, Inputs, Outputs);
        }
    }


    public class LiveSimulation
    {
        private object sync = new object();
        protected Circuit circuit = null;
        protected Simulation simulation = null;

        protected ObservableCollection<InputChannel> _inputChannels = new ObservableCollection<InputChannel>();
        protected ObservableCollection<OutputChannel> _outputChannels = new ObservableCollection<OutputChannel>();
        public ObservableCollection<InputChannel> InputChannels { get { return _inputChannels; } }
        public ObservableCollection<OutputChannel> OutputChannels { get { return _outputChannels; } }
        private double inputGain = 1.0;
        private double outputGain = 1.0;
        private Dictionary<ComputerAlgebra.Expression, Channel> inputs = new Dictionary<ComputerAlgebra.Expression, Channel>();
        private List<Probe> probes = new List<Probe>();
        private int clock = -1;
        protected int oversample = 8;
        protected Stream stream = null;
        protected System.Timers.Timer timer;
        public LiveSimulation(Schematic Simulate, Device Device, Channel[] Inputs, Channel[] Outputs)
        {
            System.Console.WriteLine("Creating simulation");
            try
            {
                // Make a clone of the schematic so we can mess with it.
                Schematic clone = Schematic.Deserialize(Simulate.Serialize());
                clone.Elements.ItemAdded += OnElementAdded;
                clone.Elements.ItemRemoved += OnElementRemoved;

                // Build the circuit from the schematic.
                circuit = clone.Build();
                System.Console.WriteLine("Cloned circuit");

                // Create the input and output controls.                
                IEnumerable<Component> components = circuit.Components;

                // Create audio input channels.
                for (int i = 0; i < Inputs.Length; ++i)
                    InputChannels.Add(new InputChannel(i) { Name = Inputs[i].Name });

                System.Console.WriteLine("Added inputs");

                ComputerAlgebra.Expression speakers = 0;

                foreach (Component i in components)
                {
                    Symbol S = i.Tag as Symbol;
                    if (S == null)
                        continue;

                    SymbolControl tag = (SymbolControl)S.Tag;
                    if (tag == null)
                        continue;
                    if (i is Speaker output)
                        speakers += output.Out;

                    // Create input controls.
                    if (i is Input input)
                    {

                        ComputerAlgebra.Expression In = input.In;
                        inputs[In] = new SignalChannel(0);

                    }
                }

                System.Console.WriteLine("Added components");

                // Create audio output channels.
                for (int i = 0; i < Outputs.Length; ++i)
                {
                    OutputChannel c = new OutputChannel(i) { Name = Outputs[i].Name, Signal = speakers };
                    c.PropertyChanged += (o, e) => { if (e.PropertyName == "Signal") RebuildSolution(); };
                    OutputChannels.Add(c);
                }

                System.Console.WriteLine("Beginning audio processing");

                // Begin audio processing.
                if (Inputs.Any() || Outputs.Any())
                    stream = Device.Open(ProcessSamples, Inputs, Outputs);
                else
                    stream = new NullStream(ProcessSamples);
            
                System.Console.WriteLine("Rebuilding solution");
            
                }catch (Exception Ex){
                    System.Console.WriteLine(Ex);
                }
            }
        

        private void OnElementAdded(object sender, ElementEventArgs e)
        {
            if (e.Element is Symbol && ((Symbol)e.Element).Component is Probe)
            {
                Probe probe = (Probe)((Symbol)e.Element).Component;
                probe.Signal = new Signal()
                {
                    Name = probe.V.ToString(),
                };
                lock (sync)
                {
                    probes.Add(probe);
                    if (simulation != null)
                        simulation.Output = probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray();
                }
            }
        }

        private void OnElementRemoved(object sender, ElementEventArgs e)
        {
            if (e.Element is Symbol && ((Symbol)e.Element).Component is Probe)
            {
                Probe probe = (Probe)((Symbol)e.Element).Component;
                lock (sync)
                {
                    probes.Remove(probe);
                    if (simulation != null)
                        simulation.Output = probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray();
                }
            }
        }


        protected int iterations = 8;
        /// <summary>
        /// Max iterations for numerical algorithms.
        /// </summary>
        public int Iterations
        {
            get { return iterations; }
            set { iterations = value; RebuildSolution(); NotifyChanged(nameof(Iterations)); }
        }

        private void NotifyChanged(string p)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(p));
        }
        public event PropertyChangedEventHandler PropertyChanged;

        public int Oversample
        {
            get { return oversample; }
            set { oversample = value; RebuildSolution(); NotifyChanged(nameof(Oversample)); }
        }

        private void ProcessSamples(int Count, SampleBuffer[] In, SampleBuffer[] Out, double Rate)
        {
            // The time covered by these samples.
            double timespan = Count / Rate;

            // Apply input gain.
            for (int i = 0; i < In.Length; ++i)
            {
                Channel ch = InputChannels[i];
                double peak = In[i].Amplify(inputGain);
                ch.SampleSignalLevel(peak, timespan);
            }

            // Run the simulation.
            lock (sync)
            {
                if (simulation != null)
                    RunSimulation(Count, In, Out, Rate);
                else
                    foreach (SampleBuffer i in Out)
                        i.Clear();
            }

            // Apply output gain.
            for (int i = 0; i < Out.Length; ++i)
            {
                Channel ch = OutputChannels[i];
                double peak = Out[i].Amplify(outputGain);
                ch.SampleSignalLevel(peak, timespan);
            }
        }

        // These lists only ever grow, but they should never contain more than 10s of items.
        readonly List<double[]> inputBuffers = new List<double[]>();
        readonly List<double[]> outputBuffers = new List<double[]>();
        private void RunSimulation(int Count, SampleBuffer[] In, SampleBuffer[] Out, double Rate)
        {
            try
            {
                // If the sample rate changed, we need to kill the simulation and let the foreground rebuild it.
                if (Rate != (double)simulation.SampleRate)
                {
                    simulation = null;
                    Thread RebuildSolutionThread = new Thread(new ThreadStart(this.RebuildSolution));
                    RebuildSolutionThread.Start();
                    return;
                }

                inputBuffers.Clear();
                foreach (Channel i in inputs.Values)
                {
                    if (i is InputChannel input)
                        inputBuffers.Add(In[input.Index].Samples);
                    else if (i is SignalChannel channel)
                        inputBuffers.Add(channel.Buffer(Count, simulation.Time, simulation.TimeStep));
                }

                outputBuffers.Clear();
                foreach (Probe i in probes)
                    outputBuffers.Add(i.AllocBuffer(Count));
                for (int i = 0; i < Out.Length; ++i)
                    outputBuffers.Add(Out[i].Samples);

                // Process the samples!
                simulation.Run(Count, inputBuffers, outputBuffers);

                foreach (Probe i in probes)
                    i.Signal.AddSamples(clock, i.Buffer);
            }
            catch (SimulationDiverged Ex)
            {
                Thread RebuildSolutionThread = new Thread(new ThreadStart(this.RebuildSolution));
                // If the simulation diverged more than one second ago, reset it and hope it doesn't happen again.
                simulation = null;
                if ((double)Ex.At > Rate)
                    RebuildSolutionThread.Start();
                foreach (SampleBuffer i in Out)
                    i.Clear();
            }
            catch (Exception Ex)
            {
                // If there was a more serious error, kill the simulation so the user can fix it.
                simulation = null;
                foreach (SampleBuffer i in Out)
                    i.Clear();
            }
        }

        private void RebuildSolution()
        {
            Stream stream = new NullStream(ProcessSamples);

            lock (sync)
            {
                simulation = null;

                try
                {
                    ComputerAlgebra.Expression h = (ComputerAlgebra.Expression)1 / (stream.SampleRate * Oversample);
                    TransientSolution solution = TransientSolution.Solve(circuit.Analyze(), h);

                    simulation = new Simulation(solution)
                    {
                        Input = inputs.Keys.ToArray(),
                        Output = probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray(),
                        Oversample = Oversample,
                        Iterations = Iterations,
                    };
                    System.Console.WriteLine("Solution was rebuilt");

                }
                catch (Exception Ex)
                {
                    System.Console.WriteLine(Ex);
                }

            }
        }
    }

        /// <summary>
    /// Base class for a running audio stream.
    /// </summary>
    public abstract class Stream
    {
        private Channel[] inputs, outputs;

        protected Stream(Channel[] Inputs, Channel[] Outputs) { inputs = Inputs; outputs = Outputs; }

        /// <summary>
        /// Handler for accepting new samples in and writing output samples out.
        /// </summary>
        /// <param name="Samples"></param>
        public delegate void SampleHandler(int Count, SampleBuffer[] In, SampleBuffer[] Out, double Rate);

        public Channel[] InputChannels { get { return inputs; } }
        public Channel[] OutputChannels { get { return outputs; } }

        public abstract double SampleRate { get; }

        public abstract void Stop();
    }

    /// <summary>
    /// Devices describe the supported audio stream properties.
    /// </summary>
    public abstract class Device
    {
        protected string name;
        public string Name { get { return name; } }

        protected Channel[] inputs;
        public Channel[] InputChannels { get { return inputs; } }
        protected Channel[] outputs;
        public Channel[] OutputChannels { get { return outputs; } }

        protected Device() { }
        protected Device(string Name) { name = Name; }

        public abstract Stream Open(Stream.SampleHandler Callback, Channel[] Input, Channel[] Output);
    }

    /// <summary>
    /// This object defines a pinned sample array, suitable for sharing
    /// with native code.
    /// </summary>
    public class SampleBuffer : IDisposable
    {
        /// <summary>
        /// Number of samples contained in this buffer.
        /// </summary>
        public uint Count => (uint)Samples.Length;

        /// <summary>
        /// Samples in this buffer.
        /// </summary>
        public double[] Samples { get; private set; }
        private GCHandle pin;

        /// <summary>
        /// Access samples of this buffer.
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public double this[int i] { get { return Samples[i]; } set { Samples[i] = value; } }

        /// <summary>
        /// Pointer to raw samples in this buffer.
        /// </summary>
        public IntPtr Raw { get { return pin.AddrOfPinnedObject(); } }

        private object tag = null;
        /// <summary>
        /// User defined tag object.
        /// </summary>
        public object Tag { get { return tag; } set { tag = value; } }

        public SampleBuffer(int Count)
        {
            Samples = new double[Count];
            pin = GCHandle.Alloc(Samples, GCHandleType.Pinned);
        }

        ~SampleBuffer() { Dispose(false); }
        public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
        private void Dispose(bool Disposing)
        {
            if (pin.IsAllocated)
                pin.Free();

            Samples = null;
        }

        /// <summary>
        /// Set this buffer to have the zero signal.
        /// </summary>
        public void Clear()
        {
            Util.ZeroMemory(Raw, Count * sizeof(double));
        }

        /// <summary>
        /// Amplify the samples in this buffer.
        /// </summary>
        /// <param name="Gain"></param>
        public double Amplify(double Gain)
        {
            return Util.Amplify(Raw, Count, Gain);
        }
    }


    public static class Util
    {
        // C# is great, but man generics suck compared to templates.
        // TODO: simd?
        public static unsafe void LEi16ToLEf64(IntPtr source, IntPtr destination, uint count)
        {
            short* from = (short*)source.ToPointer();
            double* to = (double*)destination.ToPointer();
            double scale = 1.0 / ((1 << 15) - 1);
            for (int i = 0; i < count; i++)
                to[i] = from[i] * scale;
        }
        public static unsafe void LEi32ToLEf64(IntPtr source, IntPtr destination, uint count)
        {
            int* from = (int*)source.ToPointer();
            double* to = (double*)destination.ToPointer();
            double scale = 1.0 / ((1L << 31) - 1);
            for (int i = 0; i < count; i++)
                to[i] = from[i] * scale;
        }
        public static unsafe void LEf32ToLEf64(IntPtr source, IntPtr destination, uint count)
        {
            float* from = (float*)source.ToPointer();
            double* to = (double*)destination.ToPointer();
            for (int i = 0; i < count; i++)
                to[i] = from[i];
        }

        public static unsafe void LEf64ToLEi16(IntPtr source, IntPtr destination, uint count)
        {
            double* from = (double*)source.ToPointer();
            short* to = (short*)destination.ToPointer();
            double max = (1 << 15) - 1;
            for (int i = 0; i < count; i++)
                to[i] = (short)Math.Max(Math.Min(from[i] * max, max), -max);
        }
        public static unsafe void LEf64ToLEi32(IntPtr source, IntPtr destination, uint count)
        {
            double* from = (double*)source.ToPointer();
            int* to = (int*)destination.ToPointer();
            double max = (1L << 31) - 1;
            for (int i = 0; i < count; i++)
                to[i] = (int)Math.Max(Math.Min(from[i] * max, max), -max);
        }
        public static unsafe void LEf64ToLEf32(IntPtr source, IntPtr destination, uint count)
        {
            double* from = (double*)source.ToPointer();
            float* to = (float*)destination.ToPointer();
            for (int i = 0; i < count; i++)
                to[i] = (float)from[i];
        }

        public static unsafe double Amplify(IntPtr Samples, uint count, double Gain)
        {
            double* s = (double*)Samples.ToPointer();
            double peak = 0.0;
            for (int i = 0; i < count; i++)
            {
                s[i] = s[i] * Gain;
                // TODO: Absolute value of s[i]?
                peak = Math.Max(peak, s[i]);
            }
            return peak;
        }

        public static unsafe void CopyMemory(IntPtr destination, IntPtr source, uint count) => Unsafe.CopyBlock(destination.ToPointer(), source.ToPointer(), count);

        public static unsafe void ZeroMemory(IntPtr startAddress, uint count) => Unsafe.InitBlockUnaligned(startAddress.ToPointer(), 0, count);
    }

     public class NullStream : Stream
    {
        public override double SampleRate { get { return 48000; } }

        private readonly SampleHandler callback;

        private bool run = true;
        private Thread thread;

        public NullStream(SampleHandler Callback) : base(new Channel[] { }, new Channel[] { })
        {
            callback = Callback;
            thread = new Thread(Proc);
            thread.Start();
        }

        private void Proc()
        {
            SampleBuffer[] input = new SampleBuffer[] { };
            SampleBuffer[] output = new SampleBuffer[] { };

            long samples = 0;
            DateTime start = DateTime.Now;
            while (run)
            {
                // Run at ~50 callbacks/second. This doesn't need to be super precise. In
                // practice, Thread.Sleep is going to be +/- 10s of ms, but we'll still deliver
                // the right number of samples on average.
                Thread.Sleep(20);
                double elapsed = (DateTime.Now - start).TotalSeconds;
                int needed_samples = (int)(Math.Round(elapsed * SampleRate) - samples);
                callback(needed_samples, input, output, SampleRate);
                samples += needed_samples;
            }
        }

        public override void Stop()
        {
            run = false;
            thread.Join();
            thread = null;
        }
    }

    public abstract class Channel : INotifyPropertyChanged
    {
        private string name = "";
        public string Name { get { return name; } set { name = value; NotifyChanged(nameof(Name)); } }


        private double signalLevel = 0;
        public double SignalLevel { get { return signalLevel; } }
        /// <summary>
        /// Update the signal level of this channel.
        /// </summary>
        /// <param name="level"></param>
        /// <param name="time"></param>
        public void SampleSignalLevel(double level, double time)
        {
            double a = Frequency.DecayRate(time, 0.25);
            signalLevel = Math.Max(level, level * a + signalLevel * (1 - a));
        }

        public void ResetSignalLevel() { signalLevel = 0; }

        // INotifyPropertyChanged.
        protected void NotifyChanged(string p)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(p));
        }
        public event PropertyChangedEventHandler PropertyChanged;
    }

    /// <summary>
    /// Channel of audio data.
    /// </summary>
    public class InputChannel : Channel
    {
        private int index = 0;
        public int Index { get { return index; } }

        public InputChannel(int Index) { index = Index; }
    }

    /// <summary>
    /// Channel generated by a signal expression.
    /// </summary>
    public class SignalChannel : Channel
    {
        private Func<double, double> signal;

        private double[] buffer = null;
        public double[] Buffer(int Count, double t, double dt)
        {
            if (buffer == null || buffer.Length != Count)
                buffer = new double[Count];
            for (int i = 0; i < Count; ++i, t += dt)
                buffer[i] = signal(t);
            return buffer;
        }

        public SignalChannel(ComputerAlgebra.Expression Signal)
        {
            signal = Signal.Compile<Func<double, double>>(Component.t);
        }
    }

    /// <summary>
    /// Output audio channel.
    /// </summary>
    public class OutputChannel : Channel
    {
        private int index = 0;
        public int Index { get { return index; } }

        private ComputerAlgebra.Expression signal = 0;
        public ComputerAlgebra.Expression Signal { get { return signal; } set { signal = value; NotifyChanged(nameof(Signal)); } }

        public OutputChannel(int Index) { index = Index; }
    }

    public class Frequency
    {
        private static string[] Notes = { "C", "C\u266f", "D", "D\u266f", "E", "F", "F\u266f", "G", "G\u266f", "A", "A\u266f", "B" };
        public static string ToNote(double f, double A4)
        {
            // Halfsteps above C0
            double halfsteps = (Math.Log(f / A4, 2.0) + 5.0) * 12.0 - 3.0;
            if (halfsteps < 0 || double.IsNaN(halfsteps) || double.IsInfinity(halfsteps))
                return "";

            int note = (int)Math.Round(halfsteps) % 12;
            int octave = (int)Math.Round(halfsteps) / 12;
            int cents = (int)Math.Round((halfsteps - Math.Round(halfsteps)) * 100);

            StringBuilder sb = new StringBuilder(Notes[note]);
            sb.Append(IntToSubscript(octave));
            sb.Append(' ');
            if (cents >= 0)
                sb.Append('+');
            sb.Append(cents);
            sb.Append('\u00A2');
            return sb.ToString();
        }

        private static string IntToSubscript(int x)
        {
            string chars = x.ToString();

            StringBuilder ret = new StringBuilder();
            foreach (char i in chars)
            {
                if (i == '-')
                    ret.Append((char)0x208B);
                else
                    ret.Append((char)(0x2080 + i - '0'));
            }
            return ret.ToString();
        }

        public static double Estimate(double[] Samples, int Decimate, out double Phase)
        {
            Complex[] data = DecimateSignal(Samples, Decimate);
            int N = data.Length;
            Fourier.Forward(data);
            // Zero the DC bin.
            data[0] = 0.0;

            double f = 0.0;
            double max = 0.0;
            Phase = 0.0;

            // Find largest frequency in FFT.
            for (int i = 1; i < N / 2 - 1; ++i)
            {
                double x;
                Complex m = LogParabolaMax(data[i - 1], data[i], data[i + 1], out x);

                if (m.Magnitude > max)
                {
                    max = m.Magnitude;
                    f = i + x;
                    Phase = m.Phase;
                }
            }

            // Check if this is a harmonic of another frequency (the fundamental frequency).
            double f0 = f;
            for (int h = 2; h < 5; ++h)
            {
                int i = (int)Math.Round(f / h);
                if (i >= 1)
                {
                    double x;
                    Complex m = LogParabolaMax(data[i - 1], data[i], data[i + 1], out x);

                    if (m.Magnitude * 5.0 > max)
                    {
                        f0 = f / h;
                        Phase = m.Phase;
                    }
                }
            }

            return f0;
        }

        private static double Hann(int i, int N) { return 0.5 * (1.0 - Math.Cos((2.0 * Math.PI * i) / (N - 1))); }

        // Fit parabola to 3 bins and find the maximum.
        private static Complex LogParabolaMax(Complex A, Complex B, Complex C, out double x)
        {
            double a = A.Magnitude;
            double b = B.Magnitude;
            double c = C.Magnitude;

            if (b > a && b > c)
            {
                // Parabola fitting is more accurate in log magnitude.
                a = Math.Log(a);
                b = Math.Log(b);
                c = Math.Log(c);

                // Maximum location.
                x = (a - c) / (2.0 * (a - 2.0 * b + c));

                // Maximum value.
                return Complex.FromPolarCoordinates(
                    Math.Exp(b - x * (a - c) / 4.0),
                    (B - x * (A - C) / 4.0).Phase);
            }
            else
            {
                x = 0.0;
                return B;
            }
        }

        private static Complex[] DecimateSignal(double[] Block, int Decimate)
        {
            int N = Block.Length / Decimate;
            Complex[] data = new Complex[N];

            // Decimate input audio with low pass filter.
            for (int i = 0; i < N; ++i)
            {
                double v = 0.0;
                for (int j = 0; j < Decimate; ++j)
                    v += Block[i * Decimate + j];
                data[i] = new Complex(v * Hann(i, N), 0.0);
            }
            return data;
        }

        /// <summary>
        /// Get the parameter for a first-order IIR filter.
        /// </summary>
        /// <param name="timestep">The time between steps.</param>
        /// <param name="halflife">The time to decay by half.</param>
        public static double DecayRate(double timestep, double halflife)
        {
            return Math.Exp(timestep / halflife * Math.Log(0.5));
        }
    }

      public enum ScopeMode
    {
        Oscilloscope,
        Spectrogram,
    }

     public class SignalEventArgs : EventArgs
    {
        private Signal e;
        public Signal Signal { get { return e; } }

        public SignalEventArgs(Signal E) { e = E; }
    }

    public class TickEventArgs : EventArgs
    {
        private long clock;
        public long Clock { get { return Clock; } }

        public TickEventArgs(long Clock) { clock = Clock; }
    }

    /// <summary>
    /// Collection of Signals.
    /// </summary>
    public class SignalCollection : IEnumerable<Signal>, IEnumerable
    {
        protected List<Signal> x = new List<Signal>();

        public Signal this[int index] { get { lock (x) return x[index]; } }

        public delegate void SignalEventHandler(object sender, SignalEventArgs e);

        private List<SignalEventHandler> itemAdded = new List<SignalEventHandler>();
        protected void OnItemAdded(SignalEventArgs e) { foreach (SignalEventHandler i in itemAdded) i(this, e); }
        public event SignalEventHandler ItemAdded
        {
            add { itemAdded.Add(value); }
            remove { itemAdded.Remove(value); }
        }

        private List<SignalEventHandler> itemRemoved = new List<SignalEventHandler>();
        protected void OnItemRemoved(SignalEventArgs e) { foreach (SignalEventHandler i in itemRemoved) i(this, e); }
        public event SignalEventHandler ItemRemoved
        {
            add { itemRemoved.Add(value); }
            remove { itemRemoved.Remove(value); }
        }

        private long clock = 0;
        public long Clock { get { return clock; } }

        private double sampleRate = 1;
        public double SampleRate { get { return sampleRate; } }

        public void TickClock(int SampleCount, double SampleRate)
        {
            sampleRate = SampleRate;

            int truncate = (int)sampleRate / 4;

            // Remove the signals that we didn't get data for.
            ForEach(i =>
            {
                if (i.Clock < clock)
                    i.Clear();
                else
                    i.Truncate(truncate);
            });

            clock += SampleCount;
        }

        // ICollection<Node>
        public int Count { get { lock (x) return x.Count; } }
        public void Add(Signal item)
        {
            lock (x) x.Add(item);
            OnItemAdded(new SignalEventArgs(item));
        }
        public void AddRange(IEnumerable<Signal> items)
        {
            foreach (Signal i in items)
                Add(i);
        }
        public void Clear()
        {
            Signal[] removed = x.ToArray();
            lock (x) x.Clear();

            foreach (Signal i in removed)
                OnItemRemoved(new SignalEventArgs(i));
        }
        public bool Contains(Signal item) { lock (x) return x.Contains(item); }
        public void CopyTo(Signal[] array, int arrayIndex) { lock (x) x.CopyTo(array, arrayIndex); }
        public bool Remove(Signal item)
        {
            bool ret;
            lock (x) ret = x.Remove(item);
            if (ret)
                OnItemRemoved(new SignalEventArgs(item));
            return ret;
        }
        public void RemoveRange(IEnumerable<Signal> items)
        {
            foreach (Signal i in items)
                Remove(i);
        }

        /// <summary>
        /// This is thread safe.
        /// </summary>
        /// <param name="f"></param>
        public void ForEach(Action<Signal> f)
        {
            lock (x) foreach (Signal i in x)
                    f(i);
        }

        // IEnumerable<Node>
        public IEnumerator<Signal> GetEnumerator() { return x.GetEnumerator(); }

        IEnumerator IEnumerable.GetEnumerator() { return this.GetEnumerator(); }
    }

    public class Signal : IEnumerable<double>
    {
        private List<double> samples = new List<double>();

        private string name;
        /// <summary>
        /// Name of this signal.
        /// </summary>
        public string Name { get { return name; } set { name = value; } }
        public override string ToString() { return name; }

        /// <summary>
        /// Pen to draw this signal with.
        /// </summary>

        private object tag;
        public object Tag { get { return tag; } set { tag = value; } }

        private long clock = 0;
        public long Clock { get { return clock; } }

        /// <summary>
        /// Add new samples to this signal.
        /// </summary>
        /// <param name="Clock"></param>
        /// <param name="Samples"></param>
        public void AddSamples(long Clock, double[] Samples)
        {
            lock (Lock)
            {
                samples.AddRange(Samples);
                clock = Clock;
            }
        }

        /// <summary>
        /// Truncate samples older than NewCount.
        /// </summary>
        /// <param name="Truncate"></param>
        public void Truncate(int NewCount)
        {
            lock (Lock)
            {
                if (samples.Count > NewCount)
                {
                    int remove = samples.Count - NewCount;
                    samples.RemoveRange(0, remove);
                }
            }
        }

        public void Clear() { lock (Lock) { samples.Clear(); clock = 0; } }

        public int Count { get { return samples.Count; } }
        public double this[int i]
        {
            get
            {
                if (0 <= i && i < samples.Count)
                    return samples[(int)i];
                else
                    return double.NaN;
            }
        }

        public object Lock { get { return samples; } }

        // IEnumerable<double> interface
        IEnumerator<double> IEnumerable<double>.GetEnumerator() { return samples.GetEnumerator(); }
        IEnumerator IEnumerable.GetEnumerator() { return samples.GetEnumerator(); }
    }

     /// Component to mark nodes for probing.
    /// </summary>
    class Probe : OneTerminal
    {
        protected EdgeType color;
        public EdgeType Color { get { return color; } set { color = value; } }

        public Signal Signal = null;

        private double[] buffer = null;
        public double[] Buffer { get { return buffer; } }

        public double[] AllocBuffer(int Samples)
        {
            if (buffer == null || buffer.Length != Samples)
                buffer = new double[Samples];
            return buffer;
        }

        private Probe() : this(EdgeType.Red) { }
        public Probe(EdgeType Color) { color = Color; }

        public override void Analyze(Analysis Mna) { }

        public override void LayoutSymbol(SymbolLayout Sym)
        {
            Coord w = new Coord(0, 0);
            Sym.AddTerminal(Terminal, w);

            Coord dw = new Coord(1, 1);
            Coord pw = new Coord(dw.y, -dw.x);

            w += dw * 10;
            Sym.AddWire(Terminal, w);

            Sym.AddLine(color, w - pw * 4, w + pw * 4);
            Sym.AddLoop(color,
                w + pw * 2,
                w + pw * 2 + dw * 10,
                w + dw * 12,
                w - pw * 2 + dw * 10,
                w - pw * 2);

            if (ConnectedTo != null)
                Sym.DrawText(() => V.ToString(), new Point(0, 6), Alignment.Far, Alignment.Near);
        }
    }

    public class SymbolControl : ElementControl
    {

        private bool showText = true;

        protected SymbolLayout layout;

        public SymbolControl(Symbol S) : base(S)
        {
            layout = Component.LayoutSymbol();
        }
        public SymbolControl(Component C) : this(new Symbol(C)) { }

        public Symbol Symbol { get { return (Symbol)element; } }
        public Component Component { get { return Symbol.Component; } }

    }

     public class ElementControl
    {

        protected bool showTerminals = true;
        public bool ShowTerminals { get { return showTerminals; } set { showTerminals = value;} }

        private List<EventHandler> selectedChanged = new List<EventHandler>();
        public event EventHandler SelectedChanged { add { selectedChanged.Add(value); } remove { selectedChanged.Remove(value); } }

        private bool selected = false;
        public bool Selected
        {
            get { return selected; }
            set
            {
                if (selected == value) return;

                selected = value;
                foreach (EventHandler i in selectedChanged)
                    i(this, new EventArgs());
            }
        }

        private bool highlighted = false;
        public bool Highlighted
        {
            get { return highlighted; }
            set
            {
                if (highlighted == value) return;
                highlighted = value;
            }
        }

        protected Element element;
        public Element Element { get { return element; } }

        protected ElementControl(Element E)
        {
            element = E;
            element.Tag = this;
        }

        public static ElementControl New(Element E)
        {
            if (E is Symbol symbol)
                return new SymbolControl(symbol);
            else
                throw new NotImplementedException();
        }


        public static double TerminalSize = 2.0;
        public static double EdgeThickness = 1.0;
    }
}