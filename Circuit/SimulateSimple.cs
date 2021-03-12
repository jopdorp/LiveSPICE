using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Collections.ObjectModel;


namespace Circuit{
  public class SimulateSimple
    {

       
        public static void Main(string[] args) {
            SimpleSim sim = new SimpleSim(args[0]);
        }


    }

    public class SimpleSim{

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
        protected int iterations = 8;

        protected Stream stream = null;
        protected System.Timers.Timer timer;

        readonly List<double[]> inputBuffers = new List<double[]>();
        readonly List<double[]> outputBuffers = new List<double[]>();

        public SimpleSim(string fileName){
            Schematic schema = Schematic.Load(fileName);
            Circuit circuit = schema.Build();

            NullStream stream = new NullStream(ProcessSamples);
            ComputerAlgebra.Expression h = (ComputerAlgebra.Expression)1 / (stream.SampleRate * oversample);
            TransientSolution solution = TransientSolution.Solve(circuit.Analyze(), h);
            TransientSolution.Solve(circuit.Analyze(), h);

            simulation = new Simulation(solution);
        }

        private void ProcessSamples(int Count, SampleBuffer[] In, SampleBuffer[] Out, double Rate)
        {
            // The time covered by these samples.
            double timespan = Count / Rate;

            // Apply input gain.
            for (int i = 0; i < In.Length; ++i)
            {
                System.Console.WriteLine("in:");
                System.Console.WriteLine(In[i].Amplify(1));
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

            for (int i = 0; i < In.Length; ++i)
            {
                System.Console.WriteLine("out:");
                System.Console.WriteLine(Out[i].Amplify(1));
            }
    }

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
                    ComputerAlgebra.Expression h = (ComputerAlgebra.Expression)1 / (stream.SampleRate * oversample);
                    TransientSolution solution = TransientSolution.Solve(circuit.Analyze(), h);

                    simulation = new Simulation(solution)
                    {
                        Input = inputs.Keys.ToArray(),
                        Output = probes.Select(i => i.V).Concat(OutputChannels.Select(i => i.Signal)).ToArray(),
                        Oversample = oversample,
                        Iterations = iterations,
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

}