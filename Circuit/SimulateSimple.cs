using System.Collections.Generic;
using System.Linq;
using ComputerAlgebra;
using System;


namespace Circuit{
  public class SimulateSimple
    {
        public static void Main(string[] args) {
            Schematic schema = Schematic.Load(args[0]);
            Circuit circuit = schema.Build();
            ComputerAlgebra.Expression h = (ComputerAlgebra.Expression)1 / (48000 * 1);
            TransientSolution solution = TransientSolution.Solve(circuit.Analyze(), h);
            System.Console.WriteLine(solution);



            Expression inputExpression = circuit.Components.OfType<Input>().Select(i => i.In).SingleOrDefault();

            IEnumerable<Speaker> speakers = circuit.Components.OfType<Speaker>();

            Expression outputExpression = 0;

            // Output is voltage drop across the speakers
            foreach (Speaker speaker in speakers)
            {
                outputExpression += speaker.Out;
            }

            Simulation simulation = new Simulation(solution)
            {
                Oversample = 1,
                Iterations = 16,
                Input = new[] { inputExpression },
                Output = new[] { outputExpression }
            };

            List<double[]> ins = new List<double[]>();
            List<double[]> outs = new List<double[]>();
            double[] micIn = {1};
            ins.Add(micIn);
            double[] audioOut = {0};
            outs.Add(audioOut);
            Random rand = new Random();
            for(int i = 1; i < 96000; i++){
                if(i%4 == 0){
                    ins[0][0] = rand.NextDouble();
                }
                simulation.Run(1,ins, outs);
            }
            System.Console.WriteLine(ins[0][0].ToString() + " " + outs[0][0]);
        }
    }

}